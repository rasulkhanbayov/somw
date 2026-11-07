"""Bridge to the PhysCtrl code base (cloned on the cluster at $PHYSCTRL_ROOT).

This is the only module that imports PhysCtrl. Everything else talks to the
`Backend` / `SimHandle` interface defined in backends.py, so the same scripts
run against the tiny synthetic backend with --dry-run.

Facts about PhysCtrl this module relies on (read from the repo, Oct 2026; see
notes/physctrl_ref_snapshot/):
  * MPM data: Warp solver `MPM_Simulator_WARP`, config
    src/data_generation/configs/objaverse_mpm.json (substep_dt 1e-4,
    frame_dt 0.0417, frame_num 48, n_grid 125, grid_lim 10, density 1000).
    Objects normalised to size 1 at centre (5,5,5); 2,048 particles.
  * HDF5 keys: x, v, F, C (49 frames incl. initial), vol, floor_height,
    drag_point, drag_mask, drag_force, E, nu, mat_type, gravity.
    mat_type: 0 elastic, 1 plasticine, 2 sand, 3 rigid.
  * Generator: MDM_ST (src/model/spacetime.py) + TrajPipeline (DDIM,
    prediction_type='sample'). Predicts 24 frames = simulator frames
    1,3,...,47 (n_frames_interval 2). Coordinates normalised as (x - 5) / 2.
    eval_base.yaml: 4 DDIM steps, 8 layers, latent 256, elastic only.
    eval_large.yaml: 25 DDIM steps, 12 layers, latent 512, 4 materials,
    gravity embedding.

Silent failure modes guarded here (each would corrupt results without an error):
  1. Checkpoint loaded with strict=False (as PhysCtrl's eval.py does) can load
     nothing if key prefixes differ. We fail if any model key is missing.
  2. The solver's stress update reads particle_F_trial, not particle_F. A
     restart that only imports F is silently restarted from identity. We set both.
  3. Impulse masks are selected from positions *at the time the impulse is
     added*. A restart from a deformed state would select different particles.
     We overwrite the mask with the stored drag_mask.
  4. wp.ScopedTimer(synchronize=True) inside every substep forces 4 device
     syncs per substep. Timing "as_is" keeps them (that is PhysCtrl's real cost);
     "nosync" and "graph" remove them so the simulator gets its best case.
  5. A stored frame is int(frame_dt / substep_dt) substeps, so its simulated length
     can differ from the nominal frame_dt: 417 x 1e-4 = 0.0417 s at PhysCtrl's
     default, but 26 x 1.6e-3 = 0.0416 s in the fidelity sweep. The drag impulse is
     active for exactly frame_num simulated frames, so a solver clock advanced by the
     nominal value would cut the force early and shift restart times. MPMSim.frame_dt
     is the simulated value.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

NORM_FAC = 5.0
N_MODEL_FRAMES = 24
FRAME_INTERVAL = 2
# Largest number of SDPA sequences per launch (CUDA grid limit); see Generator.sample.
MAX_ATTN_SEQUENCES = 65535
DRAG_SIZE = [0.4, 0.4, 0.4]


def model_frame_to_sim_frame(k: int) -> int:
    """Model frame k (0-based) is simulator frame 2k+1."""
    return FRAME_INTERVAL * k + 1


def to_model_coords(x):
    return (x - NORM_FAC) / 2.0


def to_sim_coords(x):
    return x * 2.0 + NORM_FAC


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_PATHS_DONE = False


def setup_physctrl_paths() -> Path:
    global _PATHS_DONE
    root = os.environ.get("PHYSCTRL_ROOT")
    if not root:
        raise RuntimeError("PHYSCTRL_ROOT is not set. Source slurm/cluster.env first.")
    root = Path(root)
    if not (root / "src" / "pipeline_traj.py").exists():
        raise RuntimeError(f"PHYSCTRL_ROOT={root} does not look like a PhysCtrl checkout.")
    if not _PATHS_DONE:
        for p in [root / "src" / "data_generation" / "simulator" / "mpm",
                  root / "src" / "data_generation",
                  root / "src"]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        _PATHS_DONE = True
    return root


# --------------------------------------------------------------------------- #
# Warp setup and timer patching
# --------------------------------------------------------------------------- #
_WARP_READY = False


def init_warp() -> None:
    global _WARP_READY
    if _WARP_READY:
        return
    import warp as wp

    cache = os.environ.get("WARP_CACHE_DIR")
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = cache
    wp.init()
    _WARP_READY = True


class _NoopTimer:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def warp_timers_disabled(enabled: bool = True):
    """Replace wp.ScopedTimer with a no-op (removes per-substep device syncs)."""
    import warp as wp

    if not enabled:
        yield
        return
    original = wp.ScopedTimer
    wp.ScopedTimer = _NoopTimer
    try:
        yield
    finally:
        wp.ScopedTimer = original


# --------------------------------------------------------------------------- #
# HDF5
# --------------------------------------------------------------------------- #
_DEFORM = None


def physctrl_deform_loss(trajs: list, q: dict, clamp: list) -> list[float]:
    """PhysCtrl's own DeformLoss (src/utils/physics.py) on (24, N, 3) trajectories in simulator units, as
    eval.py computes it: model coordinates (x - 5) / 2, outputs clamped to [-2.2, 2.2] (not the ground
    truth), bf16 autocast, frame_interval 2, norm_fac 5, the ground-truth F and C at sim frames 1, 3, ..., 47."""
    global _DEFORM
    import importlib.util
    import torch
    root = setup_physctrl_paths()
    if _DEFORM is None:
        spec = importlib.util.spec_from_file_location("physctrl_physics", root / "src" / "utils" / "physics.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                 # by path: PhysCtrl's "utils" package name is too generic
        _DEFORM = mod.DeformLoss().to("cuda")
    if q.get("F") is None or q.get("C") is None:
        raise RuntimeError(f"{q.get('qid')}: DeformLoss needs the stored F and C")
    dev = "cuda"
    F = torch.from_numpy(np.asarray(q["F"])[1::2][:24].reshape(24, -1, 9)).float().to(dev)[None]
    C = torch.from_numpy(np.asarray(q["C"])[1::2][:24].reshape(24, -1, 9)).float().to(dev)[None]
    vol = torch.from_numpy(np.asarray(q["vol"], np.float32)).to(dev)[None]
    out = []
    for traj, cl in zip(trajs, clamp):
        x = torch.from_numpy(to_model_coords(np.asarray(traj, np.float32))).to(dev)[None]
        if cl:
            x = x.clamp(min=-2.2, max=2.2)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(float(_DEFORM(x=x, vol=vol, F=F, C=C, frame_interval=2, norm_fac=5)))
    return out


def load_h5(path: str | Path) -> dict:
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    return out


def write_h5(path: str | Path, d: dict) -> None:
    """Write a trajectory in PhysCtrl's HDF5 layout (same keys and dtypes)."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        for k in ["x", "v", "F", "C", "vol", "floor_height", "drag_point", "drag_mask",
                  "drag_force", "E", "nu", "mat_type", "gravity"]:
            v = d[k]
            if isinstance(v, list) and len(v) == 0:
                f.create_dataset(k, data=v)
            else:
                f.create_dataset(k, data=np.asarray(v))
        # Phase 4l two-object scenes: which object each particle belongs to, B's translation and
        # the designed first-approach frame. Absent for single-object files.
        for k in ("object_id", "t_b", "k_c"):
            if k in d and d[k] is not None:
                f.create_dataset(k, data=np.asarray(d[k]))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# MPM simulator
# --------------------------------------------------------------------------- #
def mpm_base_params(overrides: dict | None = None):
    """Decode PhysCtrl's objaverse_mpm.json exactly as generate_mpm_data.py does."""
    root = setup_physctrl_paths()
    from simulator.mpm.decode_param import decode_param_json

    cfg_path = root / "src" / "data_generation" / "configs" / "objaverse_mpm.json"
    material_params, bc_params, time_params, _, _ = decode_param_json(str(cfg_path))
    overrides = overrides or {}
    if "substep_dt" in overrides:
        time_params["substep_dt"] = float(overrides["substep_dt"])
    if "n_grid" in overrides:
        material_params["n_grid"] = int(overrides["n_grid"])
    return material_params, time_params


def actual_frame_dt(overrides: dict | None = None) -> float:
    """Simulated seconds per stored frame: int(frame_dt/substep_dt) * substep_dt (0.0417 at default)."""
    _, time_params = mpm_base_params(overrides)
    sub = float(time_params["substep_dt"])
    return int(float(time_params["frame_dt"]) / sub) * sub


def particle_volume(x: np.ndarray, n_grid: int, grid_lim: float) -> np.ndarray:
    """Mirror of PhysCtrl filling.get_particle_volume (non-uniform branch):
    vol_i = dx^3 / (number of particles in i's grid cell)."""
    dx = grid_lim / n_grid
    idx = np.floor(x / dx).astype(np.int64)
    keys = idx[:, 0] * n_grid * n_grid + idx[:, 1] * n_grid + idx[:, 2]
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return (dx ** 3 / counts[inv]).astype(np.float32)


class MPMSim:
    """One elastic MPM scene, built exactly like PhysCtrl's data generator.

    timing_mode: 'as_is' (PhysCtrl's code unchanged), 'nosync' (ScopedTimer
    syncs removed), 'graph' (nosync + one frame of substeps captured as a CUDA
    graph and replayed).
    """

    def __init__(self, query: dict, overrides: dict | None = None, timing_mode: str = "as_is",
                 device: str = "cuda:0"):
        setup_physctrl_paths()
        init_warp()
        import torch
        import warp as wp
        from simulator.mpm.mpm_solver_warp import MPM_Simulator_WARP

        if int(query.get("mat_type", 0)) != 0:
            raise NotImplementedError("Only elastic (mat_type 0) is in scope for MPM probes.")
        self.timing_mode = timing_mode
        self.device = device
        self.wp = wp
        self.torch = torch
        self._graph = None

        material_params, time_params = mpm_base_params(overrides)
        material_params["material"] = "elastic"
        material_params["E"] = float(query["E"])
        material_params["nu"] = float(query["nu"])
        material_params["g"] = [0, -9.8, 0] if int(query["gravity"]) == 1 else [0, 0, 0]
        self.substep_dt = float(time_params["substep_dt"])
        self.nominal_frame_dt = float(time_params["frame_dt"])
        self.frame_num = int(time_params["frame_num"])
        self.substeps_per_frame = int(self.nominal_frame_dt / self.substep_dt)  # same expression as PhysCtrl
        # Simulated time per stored frame (guard 5 in the module docstring). Equals the
        # nominal 0.0417 s at the default substep, but not for every substep in the sweep.
        # The solver clock (which gates the drag impulse) and restart times use this value.
        self.frame_dt = self.substeps_per_frame * self.substep_dt
        self.grid_lim = float(material_params["grid_lim"])
        self.n_grid = int(material_params["n_grid"])

        x0 = np.asarray(query["x0"], dtype=np.float32)
        vol = np.asarray(query["vol"], dtype=np.float32)
        self.n_particles = x0.shape[0]
        x0_t = torch.tensor(x0, device=device).contiguous()
        vol_t = torch.tensor(vol, device=device).contiguous()

        with warp_timers_disabled(timing_mode != "as_is"):
            s = MPM_Simulator_WARP(10)
            s.load_initial_data_from_torch(x0_t, vol_t, n_grid=self.n_grid, grid_lim=self.grid_lim)
            s.set_parameters_dict(material_params)
            s.finalize_mu_lam_bulk()
            gl = self.grid_lim
            s.add_surface_collider([0.2, 0, 0], [1, 0, 0], surface="cut")
            s.add_surface_collider([gl - 0.2, 0, 0], [-1, 0, 0], surface="cut")
            s.add_surface_collider([0, 0, 0.2], [0, 0, 1], surface="cut")
            s.add_surface_collider([0, 0, gl - 0.2], [0, 0, -1], surface="cut")
            s.add_surface_collider([0, float(query["floor_height"]), 0], [0, 1, 0], surface="cut")
            s.add_surface_collider([0, gl - 0.2, 0], [0, -1, 0], surface="cut")

            drag_points = np.asarray(query.get("drag_point", []))
            drag_masks = np.asarray(query.get("drag_mask", []))
            drag_forces = np.asarray(query.get("drag_force", []))
            if drag_forces.size > 0:
                drag_points = drag_points.reshape(-1, 3)
                drag_forces = drag_forces.reshape(-1, 3)
                drag_masks = drag_masks.reshape(len(drag_forces), -1)
                for j in range(len(drag_forces)):
                    s.add_impulse_on_particles(
                        force=drag_forces[j], dt=self.substep_dt, point=drag_points[j],
                        size=DRAG_SIZE, num_dt=self.substeps_per_frame * self.frame_num)
                    # Guard 3: keep the original particle selection on restarts.
                    s.impulse_params[-1].mask = wp.from_numpy(
                        drag_masks[j].astype(np.int32), dtype=int, device=device)
        self.solver = s
        self.time = 0.0
        self.substeps_done = 0

    # ---- state ------------------------------------------------------------
    def set_state(self, x, v=None, F=None, C=None, time: float = 0.0) -> None:
        from warp_utils import torch2warp_mat33  # PhysCtrl simulator/mpm/warp_utils.py

        torch, dev = self.torch, self.device
        n = self.n_particles
        x_t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev).reshape(n, 3)
        v_t = torch.zeros(n, 3, device=dev) if v is None else \
            torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev).reshape(n, 3)
        F_t = torch.eye(3, device=dev).expand(n, 3, 3).contiguous() if F is None else \
            torch.as_tensor(np.asarray(F, dtype=np.float32), device=dev).reshape(n, 3, 3)
        C_t = torch.zeros(n, 3, 3, device=dev) if C is None else \
            torch.as_tensor(np.asarray(C, dtype=np.float32), device=dev).reshape(n, 3, 3)
        s = self.solver
        s.import_particle_x_from_torch(x_t, device=dev)
        s.import_particle_v_from_torch(v_t, device=dev)
        s.import_particle_F_from_torch(F_t, device=dev)
        s.import_particle_C_from_torch(C_t, device=dev)
        # Guard 2: the stress update reads F_trial.
        s.mpm_state.particle_F_trial = torch2warp_mat33(F_t.clone(), dvc=dev)
        s.time = float(time)
        self.time = float(time)
        self._graph = None  # captured graphs bake in array pointers; recapture after a restart

    def export_state(self) -> dict:
        s = self.solver
        return {
            "x": s.export_particle_x_to_torch().detach().cpu().numpy().copy(),
            "v": s.export_particle_v_to_torch().detach().cpu().numpy().copy(),
            "F": s.export_particle_F_to_torch().detach().cpu().numpy().copy(),
            "C": s.export_particle_C_to_torch().detach().cpu().numpy().copy(),
        }

    # ---- stepping -----------------------------------------------------------
    def _check_graph_valid(self):
        s = self.solver
        end_needed = self.frame_dt * self.frame_num - 1e-9
        for p in s.impulse_params:
            if not (p.start_time <= 1e-12 and p.end_time >= end_needed):
                raise RuntimeError("graph mode invalid: an impulse does not span the whole rollout")
        for b in getattr(s, "modify_bc", []):
            if b is not None:
                raise RuntimeError("graph mode invalid: Python-side boundary callback present")

    def _one_frame(self):
        s = self.solver
        if self.timing_mode == "graph":
            wp = self.wp
            if self._graph is None:
                self._check_graph_valid()
                with warp_timers_disabled(True):
                    with wp.ScopedCapture(device=self.device) as cap:
                        for _ in range(self.substeps_per_frame):
                            s.p2g2p(0, self.substep_dt, device=self.device)
                self._graph = cap.graph
                # Capture recorded but did not execute; s.time advanced in Python only.
                s.time = self.time
            wp.capture_launch(self._graph)
        else:
            with warp_timers_disabled(self.timing_mode == "nosync"):
                for _ in range(self.substeps_per_frame):
                    s.p2g2p(0, self.substep_dt, device=self.device)
        self.time += self.frame_dt
        s.time = self.time
        self.substeps_done += self.substeps_per_frame

    def step_frames(self, n_frames: int, record: bool = True) -> np.ndarray | None:
        """Advance n_frames simulator frames; return positions after each frame."""
        out = []
        for _ in range(n_frames):
            self._one_frame()
            if record:
                out.append(self.solver.export_particle_x_to_torch().detach().cpu().numpy().copy())
        return np.stack(out, 0) if record else None


def is_unstable(positions: np.ndarray, grid_lim: float = 10.0) -> bool:
    return (not np.all(np.isfinite(positions))) or positions.min() < 0 or positions.max() > grid_lim


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class Generator:
    """PhysCtrl trajectory diffusion model ('base' or 'large')."""

    def __init__(self, which: str, ckpt_dir: str | Path, device: str = "cuda",
                 compile_model: bool = False, num_inference_steps: int | None = None):
        root = setup_physctrl_paths()
        import torch
        from diffusers import DDIMScheduler
        from model.spacetime import MDM_ST
        from omegaconf import OmegaConf
        from options import TestingConfig
        from pipeline_traj import TrajPipeline
        from safetensors.torch import load_file

        if which not in ("base", "large"):
            raise ValueError(which)
        cfg = OmegaConf.load(root / "src" / "configs" / f"eval_{which}.yaml")
        ckpt = Path(ckpt_dir) / f"physctrl_{which}.safetensors"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        cfg.resume = str(ckpt)
        cfg.vis_dir = "unused"
        cfg = OmegaConf.merge(OmegaConf.structured(TestingConfig), cfg)
        self.cfg = cfg
        self.which = which
        self.device = device
        self.num_inference_steps = int(num_inference_steps or cfg.num_inference_steps)

        model = MDM_ST(cfg.pc_size, cfg.train_dataset.n_training_frames, n_feats=3,
                       model_config=cfg.model_config)
        state = load_file(str(ckpt), device="cpu")
        model_keys = set(model.state_dict().keys())
        # Guard 1: never accept a load that silently matched nothing.
        for prefix in ("", "_orig_mod.", "module."):
            cand = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)} if prefix else state
            missing = model_keys - set(cand.keys())
            if not missing:
                state = cand
                break
        missing = model_keys - set(state.keys())
        unexpected = set(state.keys()) - model_keys
        if missing:
            raise RuntimeError(f"{len(missing)} model keys missing from {ckpt}, e.g. {sorted(missing)[:5]}")
        model.load_state_dict(state, strict=True if not unexpected else False)
        self.unexpected_keys = sorted(unexpected)
        model.eval().requires_grad_(False).to(device)
        if compile_model:
            model = torch.compile(model)
        scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False)
        self.pipeline = TrajPipeline(model=model, scheduler=scheduler)
        self.pipeline.set_progress_bar_config(disable=True)
        self.n_params = sum(p.numel() for p in model.parameters())

    @staticmethod
    def max_batch(n_points: int) -> int:
        """Most samples one call can generate: 31 for PhysCtrl's 2,048 points."""
        return MAX_ATTN_SEQUENCES // int(n_points)

    def dataset(self, folder: str | Path, filenames: list[str]):
        """PhysCtrl's own TrajDataset, pointed at an explicit file list.

        TrajDataset truncates/shuffles lists in its constructor; we replace the
        list afterwards so every requested file is used, in the given order.
        """
        import tempfile

        from dataset.traj_dataset import TrajDataset
        from omegaconf import OmegaConf

        ds_cfg = OmegaConf.to_container(self.cfg.train_dataset, resolve=True)
        ds_cfg["dataset_path"] = str(folder)
        with tempfile.NamedTemporaryFile("w", suffix="_test_list.json", delete=False) as f:
            json.dump(list(filenames)[:1], f)
            ds_cfg["dataset_list"] = f.name
        ds = TrajDataset("val", OmegaConf.create(ds_cfg))
        ds.split_lst = list(filenames)
        ds.split_lst_save = list(filenames)
        ds.prepare_data_lst()
        return ds

    def sample(self, items: list[dict], n_samples: int, seed: int) -> np.ndarray:
        """items: dataset __getitem__ dicts. Returns (len(items), n_samples, 24, N, 3) in SIM coords."""
        import torch
        from torch.utils.data import default_collate

        rows = []
        for it in items:
            rows.extend([it] * n_samples)
        # The DiT's temporal attention runs one sequence per (sample, point). Past 65,535
        # sequences the SDPA kernel launch fails ("invalid configuration argument"):
        # 32 samples x 2,048 points = 65,536 did, on A100 and H200 (Oct 10, 2026).
        n_seq = len(rows) * int(np.asarray(items[0]["points_src"]).shape[-2])
        if n_seq > MAX_ATTN_SEQUENCES:
            raise ValueError(f"batch of {len(rows)} samples is {n_seq} attention sequences "
                             f"(> {MAX_ATTN_SEQUENCES}); use max_batch() samples or fewer per call")
        batch = default_collate([{k: v for k, v in r.items() if k != "model"} for r in rows])
        cfg = self.cfg
        y = None if cfg.model_config.get("num_mat", 0) == 0 else batch["mat_type"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.pipeline(
                batch["points_src"], batch["force"], batch["E"], batch["nu"], batch["mask"][..., :1],
                batch["drag_point"], batch["floor_height"], batch["gravity"], batch["base_drag_coeff"],
                y=y, device=self.device, batch_size=len(rows),
                generator=torch.Generator().manual_seed(int(seed)),
                n_frames=cfg.train_dataset.n_training_frames,
                num_inference_steps=self.num_inference_steps)
        out = out.float().cpu().numpy()
        out = out.reshape(len(items), n_samples, *out.shape[1:])
        return to_sim_coords(out)

    def sample_constrained(self, items: list[dict], n_samples: int, seed: int,
                           known: np.ndarray | None = None, known_mask: np.ndarray | None = None,
                           adjust: list[dict] | None = None, shared_noise: bool = False,
                           fc=None) -> np.ndarray:
        """Training-free constrained re-sampling (Phase 4b repair).

        The same DDIM loop as PhysCtrl's TrajPipeline.__call__ at guidance 1. The scheduler
        uses prediction_type='sample', so the model output is the clean-trajectory estimate
        x0. At every step, x0 at the verified frames (known_mask, model frames) is overwritten
        with the simulator's frames (`known`, SIM coords, (24, N, 3)) before the scheduler
        step, and the final output holds them exactly. Frames between and around the
        constraints are re-generated to agree with them: the "replacement" method for
        diffusion inpainting. With known=None this must reproduce sample() for the same seed;
        p4b_repair.py checks that on the GPU before doing anything else.

        adjust (Phase 4c conditioning repair): one dict per output row, {"dlogE": d, "force_scale": s}.
        It shifts the model's log10(E) input by d and scales its force input (drag rows, and the
        gravity row when present) by s. shared_noise draws one initial noise for all rows, so the
        rows differ only in conditioning.

        fc (Phase 4g): a fc_model.FrameCond wrapping this generator's model. The known frames are
        then also written clean into the noisy input before every step, and the wrapper's
        known-frame embedding marks them, as in its training.
        Returns (len(items), n_samples, 24, N, 3) in SIM coords."""
        import torch
        from torch.utils.data import default_collate

        rows = []
        for it in items:
            rows.extend([it] * n_samples)
        if adjust is not None:
            if len(adjust) != len(rows):
                raise ValueError(f"{len(adjust)} adjustments for {len(rows)} rows")
            rows = [dict(r, E=r["E"] + float(a.get("dlogE", 0.0)),
                         force=r["force"] * float(a.get("force_scale", 1.0))) for r, a in zip(rows, adjust)]
        n_seq = len(rows) * int(np.asarray(items[0]["points_src"]).shape[-2])
        if n_seq > MAX_ATTN_SEQUENCES:
            raise ValueError(f"batch of {len(rows)} samples is {n_seq} attention sequences (> {MAX_ATTN_SEQUENCES})")
        batch = default_collate([{k: v for k, v in r.items() if k != "model"} for r in rows])
        cfg, pipe, dev, bs = self.cfg, self.pipeline, self.device, len(rows)
        y = None if cfg.model_config.get("num_mat", 0) == 0 else batch["mat_type"]
        n_frames = cfg.train_dataset.n_training_frames
        with torch.autocast("cuda", dtype=torch.bfloat16):
            init_pc = batch["points_src"]
            shape = (1 if shared_noise else bs, n_frames, init_pc.shape[2], 3)
            sample = torch.randn(shape, generator=torch.Generator().manual_seed(int(seed))).to(dev)
            if shared_noise:
                sample = sample.expand(bs, -1, -1, -1).contiguous()
            model = pipe.model.to(dev)
            init_pc, force = init_pc.to(dev), batch["force"].to(dev)
            E, nu = batch["E"].to(dev), batch["nu"].to(dev)
            mask = batch["mask"][..., :1].to(dev).to(dtype=sample.dtype)
            drag_point, floor = batch["drag_point"].to(dev), batch["floor_height"].to(dev)
            coeff = batch["base_drag_coeff"].to(dev)
            gravity = batch["gravity"].to(dev) if batch["gravity"] is not None else None
            yy = y.to(dev) if y is not None else None
            pipe.scheduler.set_timesteps(self.num_inference_steps, device=dev)
            null_emb = torch.tensor([1] * bs).to(sample.dtype)[:, None, None].to(dev)
            if known is not None:
                km = torch.from_numpy(np.asarray(known_mask, dtype=bool)).to(dev)
                kn = torch.from_numpy(to_model_coords(np.asarray(known, dtype=np.float32))).to(dev)
            if fc is not None:
                fc.mask = km[None].expand(bs, -1) if known is not None else None
            for t in pipe.scheduler.timesteps:
                tt = torch.tensor([t] * bs, device=dev)
                if fc is not None and known is not None:
                    sample = sample.clone()
                    sample[:, km] = kn[km].to(sample.dtype)
                x0 = model(sample, tt, init_pc, force, E, nu, mask, drag_point, floor_height=floor,
                           gravity_label=gravity, coeff=coeff, y=yy, null_emb=null_emb)
                if known is not None:
                    x0 = x0.clone()
                    x0[:, km] = kn[km].to(x0.dtype)
                sample = pipe.scheduler.step(x0, tt[0], sample).prev_sample
            if known is not None:
                sample[:, km] = kn[km].to(sample.dtype)
            if fc is not None:
                fc.mask = None
        out = sample.float().cpu().numpy().reshape(len(items), n_samples, *sample.shape[1:])
        return to_sim_coords(out)
