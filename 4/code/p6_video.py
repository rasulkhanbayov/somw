"""Phase 6a (H200): qualitative videos through PhysCtrl's own image-to-video pipeline, with our
certificate and repair inserted between its trajectory and its renderer. Qualitative; the cases
and the figure-selection rule were fixed in PLAN.md "Phase 6a" (Oct 15, 2026) before any render.

Per example (PhysCtrl preprocessing, unchanged and shared by the example's cases):
  SAM -> SV3D -> LGM                    point_cloud.ply, crop_info, input_mask (PhysCtrl code)
Per case (example, force multiplier):
  query      2,048 farthest-point samples (seeded) of the reconstruction, normalised as PhysCtrl
             (size 1 at (5,5,5)); the drag box (|x - p| < 0.4) and force exactly as inference.py
             builds them; MPM force = direction x coeff x base_drag_coeff (the data generator's rule)
  truth      one reference MPM rollout (written into the query's HDF5)
  generate   base, 8 samples, seed 0, through our Generator on that HDF5
  certify    the 4k gated V2 at tau: spread_event certificate; on reject the chain and a restart
             from the coarse shadow (Phase 4i V2); anchor fails or no chain failure -> reference
  render     generation (sample 0), certified/repaired, ground truth -> PhysCtrl tracking + DAS
Output: $BIG_ROOT/renders/<case>/{query.h5, trajectories.npz, case.json, variants/<v>/result.mp4}

Paper re-render (Oct 19, 2026; configs/p6_video_v2.yaml, 'v2: true'): the same cases with the Phase 9 pipeline
(one-window certificate; M1 shadow + M2 suffix). The case is rebuilt exactly as above and its generation (sample 0) and
ground truth must match the stored Phase 6a trajectories (else it raises). Writes trajectories_v2.npz and case_v2.json
(the Phase 6a records are untouched) and one new variant, certified_v2. When ours-v2 accepts (output = the generation)
or falls back to the reference (output = the ground truth), that existing render is copied instead of rendered again:
the same trajectory, prompt and seed give the same video.

Phase 11 (pre-registered Oct 20, 2026; configs/p11_render.yaml, 'fresh: true'): new cases from code/p11_images.py
(examples_dir, examples: all, force_coeffs absolute). Renders generation, ground truth and certified_v2 (copied when
identical), and records trajectories.npz and case_v2.json per case.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from common import load_config, save_json  # noqa: E402

N_POINTS = 2048
VARIANTS = ("generation", "certified", "ground_truth")


def physctrl_args(cfg, base_dir, out_dir, name, seed):
    r = Path(os.environ["RENDER_CKPT_DIR"])
    return SimpleNamespace(base_dir=str(base_dir), output_dir=str(out_dir), data_name=name, gpu=0, elevation=0.0,
                           seed=seed, sam_ckpt_path=str(r / "sam_vit_h_4b8939.pth"),
                           lgm_ckpt_path=str(r / "lgm_fp16.safetensors"), das_ckpt_path=str(r / "cogshader5B"),
                           tracks_dir="", output_fps=24, point_size=10, len_track=0, num_frames=49, mat_label=0)


def build_query(prep_dir: Path, cfg_json: dict, force_mult: float, seed: int) -> tuple[dict, dict]:
    """PhysCtrl-format query (simulator coordinates) from the LGM reconstruction, plus the files
    PhysCtrl's tracking step needs to map the trajectory back onto the Gaussians."""
    import torch
    import trimesh
    from torch_cluster import fps
    from utils.transform import transform2origin            # PhysCtrl src/
    from physctrl_bridge import particle_volume
    from regimes import _base_drag_coeff

    pts = np.asarray(trimesh.load_mesh(str(prep_dir / "point_cloud.ply")).vertices, dtype=np.float64)
    pts, center, scale = transform2origin(pts, size=1)
    torch.manual_seed(seed)
    pt = torch.tensor(pts, dtype=torch.float32)
    idx = fps(pt, ratio=N_POINTS / len(pts), random_start=False).numpy()[:N_POINTS]
    p = pts[idx]
    if cfg_json["drag_mode"] == "point":
        dp = np.asarray(cfg_json["drag_point"], dtype=np.float64)
    elif cfg_json["drag_mode"] == "max":
        dp = p[int(np.argmax(p[:, cfg_json["drag_axis"]]))]
    else:
        raise ValueError(cfg_json["drag_mode"])
    mask = (np.abs(p - dp) < 0.4).all(axis=-1)
    if not mask.any():
        raise ValueError(f"empty drag box around {dp.tolist()}: the reconstruction has no point within 0.4")
    d = np.asarray(cfg_json["drag_dir"], dtype=np.float64)
    d /= np.linalg.norm(d)
    x0 = (p + 5.0).astype(np.float32)
    vol = particle_volume(x0, 125, 10.0)
    coeff = float(cfg_json["force_coeff"]) * force_mult
    force = d * coeff * _base_drag_coeff(vol, mask)
    q = {"x0": x0, "vol": vol, "floor_height": 0.2, "gravity": 0, "mat_type": 0,
         "E": float(10 ** float(cfg_json["log_E"])), "nu": float(cfg_json["nu"]),
         "drag_point": (dp + 5.0)[None].astype(np.float32), "drag_mask": mask[None].astype(np.int64),
         "drag_force": force[None].astype(np.float32)}
    meta = {"center": center, "scale": scale, "fps_idx": idx, "n_masked": int(mask.sum()), "coeff": coeff}
    return q, meta


def certify_and_repair(backend, q, samples, cfg):
    """The Phase 4k gated V2 on one query. Returns (trajectory (24,N,3), decision dict)."""
    from p2c_select import starts_for
    from p4i_state_repair import restart, splice
    from verifier import generated_frames, nbrs_for, resolve_fidelity, run_window

    tau, spec, lift = float(cfg["tau"]), cfg["defect_spec"], cfg["lift"]
    v2 = bool(cfg.get("v2"))                    # Phase 9 pipeline: one window, M1 shadow, M2 suffix
    g0 = samples[0]
    frames = generated_frames(q["x0"], g0)
    nbrs = nbrs_for(q, lift["k"])
    ov = resolve_fidelity(spec["fidelity_spec"], q) or None
    cache, subs = {}, 0

    def window(s):
        nonlocal subs
        if s not in cache:
            w = run_window(backend, q, frames, s, 2 * spec["m"] - (1 if s == 0 else 0), lift, n_perturb=0,
                           sim_overrides=ov, anchor=(s == 0), nbrs=nbrs)
            cache[s] = np.inf if w["blown"] else w["raw"]
            subs += w["substeps"]
        return cache[s]
    cert = starts_for("spread_event", samples, q, spec["m"])
    if v2:
        cert = cert[:1]                         # the event window comes first
    score = max(window(s) for s in cert)
    dec = {"cert_starts": cert, "cert_score": float(score), "tau": tau, "accepted": bool(score < tau)}
    if score < tau:
        return g0, {**dec, "action": "accept", "probe_substeps": subs}
    gt = q["x"][1::2][:24]
    for i, s in enumerate([0] + [2 * k + 1 for k in cfg["chain_k"]]):
        if not window(s) < tau:
            if i == 0:
                return gt, {**dec, "action": "reference (anchor fails)", "probe_substeps": subs}
            k = cfg["chain_k"][i - 1]
            sh = backend.make_sim(q, overrides=(resolve_fidelity(cfg["shadow_fidelity"], q) or None) if v2 else ov)
            sh.step_frames(2 * k + 1)
            st = {kk: np.asarray(v) for kk, v in sh.export_state().items()}
            rov = (resolve_fidelity(cfg["restart_fidelity"], q) or None) if v2 else None
            sim, sub2, blown = restart(backend, q, frames, k, state=st, sim_overrides=rov)
            if blown:
                return gt, {**dec, "action": f"reference (restart at k={k} blew up)", "probe_substeps": subs}
            return splice(g0, sim, k), {**dec, "action": f"repair from k={k}", "restart_k": k, "probe_substeps": subs,
                                        "shadow_substeps": int(sh.substeps_done), "suffix_substeps": sub2}
    return gt, {**dec, "action": "reference (no chain window fails)", "probe_substeps": subs}


def render_variant(inference, cfg_json, ex_dir: Path, prep_dir: Path, case_dir: Path, name: str, traj: np.ndarray, seed: int):
    """PhysCtrl's tracking + DAS for one trajectory (simulator coordinates, (24, N, 3))."""
    vbase, vout = case_dir / "variants_in", case_dir / "variants"
    (vbase / name).mkdir(parents=True, exist_ok=True)
    od = vout / name
    od.mkdir(parents=True, exist_ok=True)
    for f in ("input.png",):
        if not (vbase / name / f).exists():
            os.symlink(ex_dir / f, vbase / name / f)
    for f in ("point_cloud.ply", "crop_info.npy", "input_mask.png"):
        shutil.copy(prep_dir / f, od / f)
    for f in ("center.npy", "scale.npy", "fps_idx.npy"):
        shutil.copy(case_dir / f, od / f)
    np.save(od / "gen_data.npy", ((traj - 5.0) / 2.0).astype(np.float32))      # model coordinates
    a = physctrl_args(None, vbase, vout, name, seed)
    inference.run_track(a, str(od))
    inference.run_das(a, str(od), cfg_json["prompt"], seed=cfg_json.get("seed", 42))
    return od / "result.mp4"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--no-render", action="store_true", help="trajectories only (skip tracking + DAS)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    physctrl_src = Path(os.environ["PHYSCTRL_ROOT"]) / "src"
    os.chdir(physctrl_src)                                   # PhysCtrl uses relative paths (templates/)
    sys.path[:0] = [str(physctrl_src), str(physctrl_src.parent / "libs"), str(physctrl_src.parent / "libs/LGM"),
                    str(physctrl_src.parent / "libs/das")]
    import inference                                          # PhysCtrl src/inference.py
    from backends import make_backend
    from physctrl_bridge import write_h5
    from utils.seeding import seed_everything

    backend = make_backend(cfg, False)
    ex_root, out_root = Path(physctrl_src.parent / "examples"), Path(cfg["paths"]["render_dir"])
    if cfg.get("examples_dir"):                  # Phase 11: a new image set (code/p11_images.py)
        ex_root = Path(os.path.expandvars(cfg["examples_dir"]))
    examples = cfg["examples"]
    if examples == "all":
        examples = sorted(d.name for d in ex_root.iterdir() if (d / "config.json").exists())
    absolute = bool(cfg.get("force_coeffs"))    # Phase 11: absolute force coefficients instead of multipliers
    cases = [(e, float(f)) for e in examples for f in (cfg["force_coeffs"] if absolute else cfg["force_mults"])]
    if absolute:     # Phase 11: shard by object, so one object's preprocessing never runs on two GPUs at once
        cases = [c for c in cases if examples.index(c[0]) % args.num_shards == args.shard]
    else:
        cases = cases[args.shard::args.num_shards]
    for ex, fm in cases:
        ex_dir = ex_root / ex
        cfg_json = json.loads((ex_dir / "config.json").read_text())
        if absolute:
            cfg_json["force_coeff"], coeff_abs, fm = fm, fm, 1.0
        prep = out_root / "_prep" / ex
        prep.mkdir(parents=True, exist_ok=True)
        seed = int(cfg["seed"])
        if not (prep / "point_cloud.ply").exists():
            seed_everything(seed)
            pa = physctrl_args(cfg, ex_root, out_root / "_prep", ex, seed)
            inference.run_sam(pa, str(prep))
            inference.run_sv3d(pa, str(prep))
            inference.run_LGM(pa, str(prep))
        case = f"{ex}__coeff_{coeff_abs:g}" if absolute else f"{ex}__force_x{fm:g}"
        cdir = out_root / case
        need = ("certified_v2",) + (("generation", "ground_truth") if cfg.get("fresh") else ())
        if cfg.get("v2") and (cdir / "case_v2.json").exists() and all((cdir / "variants" / v / "result.mp4").exists() for v in need):
            print(f"[p6 v2] {case}: already done; skipped (its record and render stay as rendered)", flush=True)
            continue
        if cfg.get("fresh") and (cdir / "case.json").exists() and json.loads((cdir / "case.json").read_text()).get("invalid"):
            print(f"[p6 v2] {case}: recorded as invalid; skipped", flush=True)
            continue
        cdir.mkdir(parents=True, exist_ok=True)
        try:
            q, meta = build_query(prep, cfg_json, fm, seed)
        except ValueError as bad:                             # recorded, never rendered silently
            save_json({"case": case, "example": ex, "force_mult": fm, "invalid": str(bad)}, cdir / "case.json")
            print(f"[p6] {case}: INVALID ({bad})", flush=True)
            continue
        qname = "query_v2.h5" if cfg.get("v2") else "query.h5"      # v2 never overwrites the Phase 6a records
        if not cfg.get("v2") or cfg.get("fresh"):      # fresh (Phase 11) cases are new: the renderer needs these files
            for k in ("center", "scale", "fps_idx"):
                np.save(cdir / f"{k}.npy", np.asarray(meta[k]))
        q.update(qid=case, split="video", path=str(cdir / qname))
        sim = backend.make_sim(q)
        xs, vs, Fs, Cs = [], [], [], []
        st = sim.export_state()
        xs.append(st["x"]); vs.append(st["v"]); Fs.append(st["F"]); Cs.append(st["C"])
        for _ in range(48):
            sim.step_frames(1, record=False)
            st = sim.export_state()
            xs.append(st["x"]); vs.append(st["v"]); Fs.append(st["F"]); Cs.append(st["C"])
        q["x"] = np.stack(xs, 0)
        if cfg.get("fresh") and not np.isfinite(q["x"]).all():      # amended Oct 21: excluded and counted, as in Phases 4m/9
            save_json({"case": case, "example": ex, "force_coeff": cfg_json["force_coeff"], "log_E": cfg_json["log_E"],
                       "invalid": "unstable reference simulation (non-finite ground truth)"}, cdir / "case.json")
            print(f"[p6] {case}: INVALID (unstable reference)", flush=True)
            continue
        write_h5(cdir / qname, {"x": q["x"], "v": np.stack(vs, 0), "F": np.stack(Fs, 0), "C": np.stack(Cs, 0),
                                     **{k: q[k] for k in ("vol", "floor_height", "drag_point", "drag_mask", "drag_force",
                                                          "E", "nu", "mat_type", "gravity")}})
        samples = backend.generate("base", [q], int(cfg["n_samples"]), seed)[0]
        traj_c, dec = certify_and_repair(backend, q, samples, cfg)
        gt = q["x"][1::2][:24]
        if cfg.get("v2") and cfg.get("fresh"):      # Phase 11: new cases, no Phase 6a record to match
            err = {"generation": float(np.linalg.norm(samples[0] - gt, axis=2).mean()),
                   "certified_v2": float(np.linalg.norm(traj_c - gt, axis=2).mean())}
            np.savez_compressed(cdir / "trajectories.npz", generation=samples[0], certified_v2=traj_c, ground_truth=gt)
            save_json({"case": case, "example": ex, "force_coeff": cfg_json["force_coeff"], "log_E": cfg_json["log_E"],
                       "n_masked": meta["n_masked"], "errors": err, "decision": dec}, cdir / "case_v2.json")
            print(f"[p6 p11] {case}: gen {err['generation']:.4f} -> ours-v2 {err['certified_v2']:.4f} ({dec['action']})", flush=True)
            if not args.no_render:
                for v, t in (("generation", samples[0]), ("ground_truth", gt)):
                    if not (cdir / "variants" / v / "result.mp4").exists():
                        render_variant(inference, cfg_json, ex_dir, prep, cdir, v, t, seed)
                dst = cdir / "variants" / "certified_v2" / "result.mp4"
                if not dst.exists():
                    same = next((v for v, t in (("generation", samples[0]), ("ground_truth", gt)) if np.array_equal(traj_c, t)), None)
                    if same:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(cdir / "variants" / same / "result.mp4", dst)
                    else:
                        render_variant(inference, cfg_json, ex_dir, prep, cdir, "certified_v2", traj_c, seed)
            continue
        if cfg.get("v2"):
            old = np.load(cdir / "trajectories.npz")
            for k_, new in (("generation", samples[0]), ("ground_truth", gt)):
                d_ = float(np.abs(old[k_] - new).max())
                print(f"[p6 v2] {case}: rebuilt {k_} max deviation from Phase 6a {d_:.2e}", flush=True)
                if d_ > 2e-3:       # GPU MPM is not bit-reproducible; 2e-3 as in fig_ours_traj.py (penguin x3: 1.0e-3)
                    raise RuntimeError(f"{case}: rebuilt {k_} differs from Phase 6a by {d_:.2e}")
            err = {"generation": float(np.linalg.norm(samples[0] - gt, axis=2).mean()),
                   "certified_v2": float(np.linalg.norm(traj_c - gt, axis=2).mean())}
            np.savez_compressed(cdir / "trajectories_v2.npz", certified_v2=traj_c)
            save_json({"case": case, "errors": err, "decision": dec}, cdir / "case_v2.json")
            print(f"[p6 v2] {case}: gen {err['generation']:.4f} -> ours-v2 {err['certified_v2']:.4f} ({dec['action']})", flush=True)
            dst = cdir / "variants" / "certified_v2" / "result.mp4"
            if not args.no_render and not dst.exists():
                same = next((v for v, t in (("generation", samples[0]), ("ground_truth", gt)) if np.array_equal(traj_c, t)), None)
                if same:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cdir / "variants" / same / "result.mp4", dst)
                    print(f"[p6 v2] {case}: output = {same}; render copied", flush=True)
                else:
                    render_variant(inference, cfg_json, ex_dir, prep, cdir, "certified_v2", traj_c, seed)
            continue
        trajs = {"generation": samples[0], "certified": traj_c, "ground_truth": gt}
        err = {k: float(np.linalg.norm(v - gt, axis=2).mean()) for k, v in trajs.items()}
        np.savez_compressed(cdir / "trajectories.npz", **trajs)
        rec = {"case": case, "example": ex, "force_mult": fm, "log_E": cfg_json["log_E"], "n_masked": meta["n_masked"],
               "coeff": meta["coeff"], "errors": err, "decision": dec}
        save_json(rec, cdir / "case.json")
        print(f"[p6] {case}: gen {err['generation']:.4f} -> certified {err['certified']:.4f} ({dec['action']})", flush=True)
        if not args.no_render:
            for v in VARIANTS:
                if not (cdir / "variants" / v / "result.mp4").exists():
                    render_variant(inference, cfg_json, ex_dir, prep, cdir, v, trajs[v], seed)
            print(f"[p6] {case}: rendered {', '.join(VARIANTS)}", flush=True)


if __name__ == "__main__":
    main()
