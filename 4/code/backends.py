"""Backends: one interface, two implementations.

PhysCtrlBackend  real data, real Warp MPM, real PhysCtrl generator (GPU, cluster).
SyntheticBackend tiny mass-spring stand-in (CPU, seconds) used only by --dry-run
                 and selftest.py. Its numbers mean nothing; every run using it
                 is written under results/SYNTHETIC_*.

Query dict keys (both backends):
  qid, split, x (49,N,3) ground-truth positions or None, v/F/C ground truth or None,
  x0 (N,3), vol (N,), floor_height, drag_point (nf,3), drag_mask (nf,N),
  drag_force (nf,3), E, nu, gravity (0/1), mat_type, path (h5 or None).

Simulator handle interface (MPMSim and SpringSim):
  substeps_per_frame, substep_dt, frame_dt, frame_num, n_particles, substeps_done,
  set_state(x, v, F, C, time), step_frames(n) -> (n, N, 3), export_state().
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from common import load_json, require

def _h(s: str) -> int:
    """Stable across processes (unlike hash(), which is salted per run)."""
    return zlib.crc32(s.encode())


N_SIM_FRAMES = 48          # frames after the initial one; arrays hold 49
FRAME_DT = 0.0417


def query_from_h5(path: str | Path, split: str, qid: str | None = None) -> dict:
    from physctrl_bridge import load_h5

    d = load_h5(path)
    q = {
        "qid": qid or Path(path).stem,
        "split": split,
        "path": str(path),
        "x": d["x"].astype(np.float32),
        "v": d.get("v"),
        "F": d.get("F"),
        "C": d.get("C"),
        "x0": d["x"][0].astype(np.float32),
        "vol": d["vol"].astype(np.float32),
        "floor_height": float(np.asarray(d["floor_height"])),
        "drag_point": np.asarray(d.get("drag_point", [])),
        "drag_mask": np.asarray(d.get("drag_mask", [])),
        "drag_force": np.asarray(d.get("drag_force", [])),
        "E": float(np.asarray(d["E"])),
        "nu": float(np.asarray(d["nu"])),
        # The released test set stores neither key; PhysCtrl's TrajDataset defaults both
        # to 0 (traj_dataset.py:203-207, 246-251), so these defaults match the reference.
        "gravity": int(np.asarray(d.get("gravity", 0))),
        "mat_type": int(np.asarray(d.get("mat_type", 0))),
    }
    if "object_id" in d:                     # Phase 4l two-object scene
        q["object_id"] = np.asarray(d["object_id"]).astype(np.int64)
    return q


class PhysCtrlBackend:
    synthetic = False

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._generators = {}

    # -- data ------------------------------------------------------------------
    def manifest(self, manifest_path: str | Path, split: str | None = None,
                 mat_type: int | None = 0, limit: int | None = None) -> list[dict]:
        entries = load_json(manifest_path)
        if split is not None:
            entries = [e for e in entries if e["split"] == split]
        if mat_type is not None:
            entries = [e for e in entries if int(e["mat_type"]) == mat_type]
        entries = sorted(entries, key=lambda e: e["qid"])
        return entries[:limit] if limit else entries

    def load(self, entry: dict) -> dict:
        return query_from_h5(entry["path"], entry["split"], entry["qid"])

    # -- simulator ---------------------------------------------------------------
    def frame_dt(self) -> float:
        from physctrl_bridge import actual_frame_dt

        return actual_frame_dt()

    def make_sim(self, query: dict, overrides: dict | None = None, timing_mode: str | None = None):
        """Default mode is 'nosync': numerically identical to PhysCtrl's 'as_is' (only the
        per-substep device syncs are removed), and several times faster for probes.
        Timing experiments pass the mode explicitly."""
        from physctrl_bridge import MPMSim

        if overrides is None and self.cfg.get("reference_cfl"):
            # Pilot E1 (Oct 18, 2026): for very stiff materials the fixed reference step (1e-4) breaks the
            # CFL condition, so the reference becomes min(1e-4, reference_cfl x dx / c) on the 125 grid.
            # Off by default: no earlier phase sets reference_cfl, so its reference is unchanged.
            from verifier import resolve_fidelity
            overrides = resolve_fidelity({"substep_dt": 1e-4, "n_grid": 125,
                                          "cfl_target": float(self.cfg["reference_cfl"])}, query) or None
        return MPMSim(query, overrides=overrides, timing_mode=timing_mode or self.cfg.get("sim_mode", "nosync"))

    # -- generator ---------------------------------------------------------------
    def generator(self, which: str, **kw):
        from physctrl_bridge import Generator

        key = (which, tuple(sorted(kw.items())))
        if key not in self._generators:
            self._generators[key] = Generator(which, self.cfg["paths"]["ckpt_dir"], **kw)
        return self._generators[key]

    def generate(self, which: str, queries: list[dict], n_samples: int, seed: int,
                 max_rows: int = 64, **kw) -> np.ndarray:
        """Returns (len(queries), n_samples, 24, N, 3) in simulator coordinates."""
        gen = self.generator(which, **kw)
        out = []
        for q in queries:
            path = Path(q["path"])
            ds = gen.dataset(path.parent, [path.name])
            item = ds[0][0]
            chunks = []
            done = 0
            rows = min(max_rows, gen.max_batch(np.asarray(item["points_src"]).shape[-2]))
            while done < n_samples:
                n = min(rows, n_samples - done)
                chunks.append(gen.sample([item], n, seed + done)[0])
                done += n
            out.append(np.concatenate(chunks, 0))
        return np.stack(out, 0)

    def generate_constrained(self, which: str, query: dict, n_samples: int, seed: int,
                             known=None, known_mask=None, fc_ckpt: str | None = None) -> np.ndarray:
        """(n_samples, 24, N, 3) from Generator.sample_constrained (Phase 4b repair). With
        fc_ckpt, a separate generator instance carries the fine-tuned frame-conditioned weights
        (Phase 4g); the plain one is never modified."""
        if fc_ckpt is None:
            gen, fc = self.generator(which), None
        else:
            key = ("fc", which, str(fc_ckpt))
            if key not in self._generators:
                from fc_model import load_fc
                from physctrl_bridge import Generator

                g = Generator(which, self.cfg["paths"]["ckpt_dir"])
                self._generators[key] = (g, load_fc(g, fc_ckpt))
            gen, fc = self._generators[key]
        path = Path(query["path"])
        item = gen.dataset(path.parent, [path.name])[0][0]
        return gen.sample_constrained([item], n_samples, seed, known, known_mask, fc=fc)[0]

    def generate_conditioned(self, which: str, query: dict, seed: int, adjust: list[dict]) -> np.ndarray:
        """(len(adjust), 24, N, 3): one sample per conditioning adjustment, all from the same
        noise (Phase 4c conditioning repair)."""
        gen = self.generator(which)
        path = Path(query["path"])
        item = gen.dataset(path.parent, [path.name])[0][0]
        return gen.sample_constrained([item], len(adjust), seed, adjust=adjust, shared_noise=True)[0]

    def generate_reference(self, which: str, query: dict, n_samples: int, seed: int) -> np.ndarray:
        """PhysCtrl's own pipeline for the same item and seed (the equivalence check)."""
        gen = self.generator(which)
        path = Path(query["path"])
        item = gen.dataset(path.parent, [path.name])[0][0]
        return gen.sample([item], n_samples, seed)[0]


# =========================================================================== #
# Synthetic backend
# =========================================================================== #
class SpringSim:
    """Damped mass-spring network on a kNN graph. A stand-in with the MPMSim
    interface so the pipeline runs on a CPU in seconds. Not physics we report."""

    def __init__(self, query: dict, overrides: dict | None = None, timing_mode: str = "as_is"):
        overrides = overrides or {}
        self.substep_dt = float(overrides.get("substep_dt", 5e-4))
        self.frame_num = N_SIM_FRAMES
        self.substeps_per_frame = int(FRAME_DT / self.substep_dt)
        self.frame_dt = self.substeps_per_frame * self.substep_dt   # simulated time per frame
        self.timing_mode = timing_mode
        x0 = np.asarray(query["x0"], dtype=np.float64)
        self.n_particles = len(x0)
        self.mass = np.full(self.n_particles, 1.0 / self.n_particles)
        d = np.linalg.norm(x0[:, None] - x0[None], axis=-1)
        nbr = np.argsort(d, axis=1)[:, 1:7]
        edges = {(min(i, j), max(i, j)) for i in range(len(x0)) for j in nbr[i]}
        self.edges = np.array(sorted(edges))
        self.rest = np.linalg.norm(x0[self.edges[:, 0]] - x0[self.edges[:, 1]], axis=1)
        self.ks = 10.0 ** (np.log10(query["E"]) - 3.0)
        self.damp = 2.0
        self.g = np.array([0, -9.8, 0]) if int(query["gravity"]) == 1 else np.zeros(3)
        self.floor = float(query["floor_height"])
        self.fext = np.zeros((self.n_particles, 3))
        dm = np.asarray(query.get("drag_mask", []))
        df = np.asarray(query.get("drag_force", []))
        if df.size:
            for m, f in zip(dm.reshape(len(df.reshape(-1, 3)), -1), df.reshape(-1, 3)):
                self.fext[m.astype(bool)] += f
        self.x = x0.copy()
        self.v = np.zeros_like(x0)
        self.time = 0.0
        self.substeps_done = 0

    def set_state(self, x, v=None, F=None, C=None, time: float = 0.0):
        self.x = np.asarray(x, dtype=np.float64).reshape(-1, 3).copy()
        self.v = np.zeros_like(self.x) if v is None else np.asarray(v, dtype=np.float64).reshape(-1, 3).copy()
        self.time = float(time)

    def export_state(self):
        n = self.n_particles
        return {"x": self.x.copy(), "v": self.v.copy(),
                "F": np.tile(np.eye(3).reshape(1, 9), (n, 1)), "C": np.zeros((n, 9))}

    def _substep(self):
        i, j = self.edges[:, 0], self.edges[:, 1]
        dx = self.x[j] - self.x[i]
        L = np.linalg.norm(dx, axis=1) + 1e-12
        f = (self.ks * (L - self.rest) / L)[:, None] * dx
        force = np.zeros_like(self.x)
        np.add.at(force, i, f)
        np.add.at(force, j, -f)
        acc = force / self.mass[:, None] + self.fext / self.mass[:, None] + self.g - self.damp * self.v
        self.v = self.v + self.substep_dt * acc
        self.x = self.x + self.substep_dt * self.v
        below = self.x[:, 1] < self.floor
        self.x[below, 1] = self.floor
        self.v[below, 1] = np.maximum(self.v[below, 1], 0.0)

    def step_frames(self, n_frames: int, record: bool = True):
        out = []
        for _ in range(n_frames):
            for _ in range(self.substeps_per_frame):
                self._substep()
            self.time += self.frame_dt
            self.substeps_done += self.substeps_per_frame
            if record:
                out.append(self.x.astype(np.float32).copy())
        return np.stack(out, 0) if record else None


SYNTH_SPLITS = {
    # split name: (generator bias level, E multiplier, gravity, force multiplier)
    "test_id": (0.10, 1.0, 0, 1.0),
    "ood_E_low": (0.60, 10 ** -0.5, 0, 1.0),
    "ood_E_high": (0.60, 10 ** 0.5, 0, 1.0),
    "ood_force_x3": (0.60, 1.0, 0, 3.0),
    "ood_gravity_drop": (0.80, 1.0, 1, 0.0),
}


class SyntheticBackend:
    synthetic = True

    def __init__(self, cfg: dict | None = None, n_points: int = 40, per_split: int = 6, seed: int = 0):
        self.n_points = n_points
        self.per_split = per_split
        self.seed = seed
        self._cache = {}

    def manifest(self, manifest_path=None, split: str | None = None, mat_type: int | None = 0,
                 limit: int | None = None) -> list[dict]:
        splits = [split] if split else list(SYNTH_SPLITS)
        entries = [{"qid": f"{s}_{i:03d}", "split": s, "path": None, "mat_type": 0}
                   for s in splits for i in range(self.per_split)]
        return entries[:limit] if limit else entries

    def load(self, entry: dict) -> dict:
        if entry["qid"] in self._cache:
            return self._cache[entry["qid"]]
        bias, e_mult, grav, f_mult = SYNTH_SPLITS[entry["split"]]
        rng = np.random.default_rng(_h(entry["qid"]))
        x0 = rng.normal(size=(self.n_points, 3)) * 0.25 + 5.0
        E = 10 ** rng.uniform(4, 7) * e_mult
        nu = rng.uniform(0.05, 0.45)
        mask = np.zeros((1, self.n_points), dtype=np.int32)
        if f_mult > 0:
            mask[0, np.argsort(x0[:, 0])[-6:]] = 1
            force = (rng.normal(size=3) / 3.0 + np.array([0.5, 0.2, 0.0])) * 0.02 * f_mult
            drag_force = force[None]
            drag_point = x0[mask[0].astype(bool)].mean(0)[None]
        else:
            drag_force = np.zeros((0, 3))
            drag_point = np.zeros((0, 3))
            mask = np.zeros((0, self.n_points), dtype=np.int32)
        q = {"qid": entry["qid"], "split": entry["split"], "path": None, "x0": x0.astype(np.float32),
             "vol": np.full(self.n_points, 1e-3, dtype=np.float32),
             "floor_height": float(x0[:, 1].min() - 0.15) if grav else 0.2,
             "drag_point": drag_point, "drag_mask": mask, "drag_force": drag_force,
             "E": float(E), "nu": float(nu), "gravity": grav, "mat_type": 0, "_bias": bias}
        sim = SpringSim(q)
        traj = sim.step_frames(N_SIM_FRAMES)
        q["x"] = np.concatenate([x0[None].astype(np.float32), traj], 0)
        q["v"] = None
        n = self.n_points
        q["F"] = np.tile(np.eye(3).reshape(1, 1, 9), (N_SIM_FRAMES + 1, n, 1)).astype(np.float32)
        q["C"] = np.zeros((N_SIM_FRAMES + 1, n, 9), dtype=np.float32)
        self._cache[entry["qid"]] = q
        return q

    def frame_dt(self) -> float:
        return int(FRAME_DT / 5e-4) * 5e-4

    def make_sim(self, query: dict, overrides: dict | None = None, timing_mode: str = "as_is"):
        return SpringSim(query, overrides, timing_mode)

    def generate(self, which: str, queries: list[dict], n_samples: int, seed: int, **kw) -> np.ndarray:
        """Biased simulator output plus temporally smooth noise; bias grows OOD."""
        out = []
        for q in queries:
            rng = np.random.default_rng(seed + _h(q["qid"]) % 10000)
            b = abs(rng.normal(q["_bias"], 0.3 * q["_bias"]))
            q_b = dict(q, E=q["E"] * (1 + 3 * b))
            biased = SpringSim(q_b).step_frames(N_SIM_FRAMES)[0::2]       # sim frames 1,3,...,47
            samples = []
            for s in range(n_samples):
                r = np.random.default_rng(seed * 1000 + s + _h(q["qid"]) % 997)
                noise = np.cumsum(r.normal(scale=0.004 * (1 + 3 * b), size=biased.shape), axis=0)
                samples.append(biased + noise)
            out.append(np.stack(samples, 0))
        return np.stack(out, 0).astype(np.float32)

    def generate_constrained(self, which: str, query: dict, n_samples: int, seed: int,
                             known=None, known_mask=None, fc_ckpt: str | None = None) -> np.ndarray:
        """Pipeline stand-in: a fresh sample with the known frames written in. It has no
        diffusion, so repair cannot propagate here; the numbers mean nothing."""
        out = self.generate(which, [query], n_samples, seed)[0]
        if known is not None:
            out[:, np.asarray(known_mask, bool)] = np.asarray(known)[np.asarray(known_mask, bool)]
        return out

    def generate_reference(self, which: str, query: dict, n_samples: int, seed: int) -> np.ndarray:
        return self.generate(which, [query], n_samples, seed)[0]

    def generate_conditioned(self, which: str, query: dict, seed: int, adjust: list[dict]) -> np.ndarray:
        """Stand-in: re-runs the synthetic generator with the adjusted E; the force scale enters
        through its bias. The numbers mean nothing."""
        out = []
        for a in adjust:
            q = dict(query, E=float(query["E"]) * 10 ** float(a.get("dlogE", 0.0)),
                     _bias=float(query["_bias"]) * float(a.get("force_scale", 1.0)))
            out.append(self.generate(which, [q], 1, seed)[0, 0])
        return np.stack(out, 0)


def testset_manifest_path(dry_run: bool):
    """The test-set manifest written by Phase 0 (A100 run preferred)."""
    from common import RESULTS_ROOT

    if dry_run:
        return None
    for tag in ("p0_check_A100", "p0_check_H200", "p0_check"):
        p = RESULTS_ROOT / tag / "testset_manifest.json"
        if p.exists():
            return p
    raise FileNotFoundError("No testset_manifest.json; run Phase 0 (01_p0_check_A100.sh) first.")


def ood_manifest_entries(dry_run: bool, backend, runs: tuple[str, ...] = ("p2_build_ood",)) -> list[dict]:
    """All queries written by p2_build_ood.py runs (default: the Phase 2 OOD build)."""
    import glob

    from common import RESULTS_ROOT

    if dry_run:
        return [e for e in backend.manifest() if e["split"] != "test_id"]
    files = sorted(f for r in runs for f in glob.glob(str(RESULTS_ROOT / r / "shard_*" / "ood_manifest.json")))
    if not files:
        raise FileNotFoundError(f"No manifests for {runs}; run the matching build job first.")
    out = []
    for f in files:
        out.extend(load_json(f))
    return sorted(out, key=lambda e: (e["split"], e["qid"]))


def make_backend(cfg: dict, dry_run: bool):
    if dry_run:
        return SyntheticBackend(cfg)
    require("paths" in cfg, "config needs a 'paths' section for the PhysCtrl backend")
    return PhysCtrlBackend(cfg)
