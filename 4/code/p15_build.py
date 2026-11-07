"""Phase 15a (A100; pre-registered Oct 22, 2026, clarified Oct 24 before data, PLAN.md "Phase 15"): joint rollouts of
the designed two-object scenes and the model files the generator trains and is evaluated on.

Per valid scene of results/p15a_scenes/scenes.jsonl:
  joint      one rollout of A (its own drag) + B (at rest, translated by t_b) in one grid, 4,096 particles, PhysCtrl's
             reference setting (as Phase 4l); scene (E, nu) = A's draw.
  sanity     invalid if, before model frame k_c - 2 (controls: all frames), the joint rollout deviates from the
             superposition (A's stored rollout + B at rest) by > 0.02 (mean point distance).
  model file 2,048 points (A's 1,024 FPS points, then B's), PhysCtrl's HDF5 layout (TrajDataset reads it unchanged):
             x, F, C of the joint rollout on those points; vol; A's drag on A's points; drag_force = f / c_A with
             base_drag_coeff = c_A (the force input PhysCtrl computes for A alone); object_id, t_b, k_c, sel_a, sel_b.
  rest file  (development scenes only) B alone at rest with zero force for the composed baseline: B's own file with
             drag_force = 0 (its drag box kept as the mask, so the input is well defined).
Output: $BIG_ROOT/p15/{joint,model,rest}/<split>/*.h5; results/p15a_build/shard_<i>/{ood_manifest.json, build.jsonl,
        failures.jsonl}
Falsifier: none (data step); counts of unstable and sanity-invalid scenes are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost, load_config,  # noqa: E402
                    read_jsonl, results_dir, resume_rows, save_json, shard_items, start_run)
from backends import make_backend, query_from_h5, N_SIM_FRAMES  # noqa: E402
from p15_scenes import fps  # noqa: E402

SANITY_TOL = 0.02


def drag_coeff(vol, mask):
    """PhysCtrl's base_drag_coeff for one object (traj_dataset.py): 9.8 * 1000 * total volume / n_masked."""
    vol, mask = np.asarray(vol, np.float64).reshape(-1), np.asarray(mask).reshape(-1).astype(bool)
    return 9.8 * 1000 * vol.sum() / mask.sum()


def write_model_h5(path, d):
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in d.items():
            f.create_dataset(k, data=np.asarray(v))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg["run_name"], cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    from physctrl_bridge import is_unstable

    scenes = [r for r in read_jsonl(results_dir(cfg["scenes_run"]) / "scenes.jsonl") if r["valid"]]
    if args.dry_run:                       # CPU pipeline test: real designs, stand-in simulator, two scenes
        scenes = [next(r for r in scenes if r["kind"] == k) for k in ("contact", "control")] + \
                 [next(r for r in scenes if r["set"] == "dev")]
    scenes = shard_items(scenes, shard, args.num_shards)
    root = Path(os.path.expandvars(cfg["out_root"])) if not args.dry_run else run.path("p15")
    blog, flog = run.path("build.jsonl"), run.path("failures.jsonl")
    done = {r["scene"] for r in resume_rows(blog, args.resume)}
    failed = {r["scene"] for r in resume_rows(flog, args.resume)}
    mpath = run.path("ood_manifest.json")
    manifest = json.load(open(mpath)) if args.resume and mpath.exists() else []
    for rec in scenes:
        sc, split = rec["scene"], rec["split"]
        if sc in done or sc in failed:
            continue
        qa = query_from_h5(rec["a_path"], split, sc)
        qb = query_from_h5(rec["b_path"], split, sc)
        t_b = np.asarray(rec["t_b"], np.float32)
        na, nb = len(qa["x0"]), len(qb["x0"])
        ma = np.asarray(qa["drag_mask"]).reshape(1, na).astype(bool)
        o = {"qid": sc, "split": split, "path": None, "x0": np.concatenate([qa["x0"], qb["x0"] + t_b]).astype(np.float32),
             "vol": np.concatenate([qa["vol"], qb["vol"]]).astype(np.float32), "floor_height": float(qa["floor_height"]),
             "gravity": 0, "mat_type": 0, "E": qa["E"], "nu": qa["nu"],
             "drag_point": np.asarray(qa["drag_point"]).reshape(1, 3),
             "drag_mask": np.concatenate([ma, np.zeros((1, nb), bool)], 1),
             "drag_force": np.asarray(qa["drag_force"]).reshape(1, 3),
             "object_id": np.concatenate([np.zeros(na, np.int64), np.ones(nb, np.int64)])}
        sel_a, sel_b = fps(qa["x0"]), fps(qb["x0"])
        sel = np.concatenate([sel_a, na + sel_b])
        try:
            # dry run: the stand-in spring network blows up at real stiffness, so it gets a soft material
            sim = backend.make_sim(dict(o, E=1e4) if args.dry_run else o, timing_mode=cfg["mode"])
            xs, Fs, Cs = [], [], []
            for i in range(N_SIM_FRAMES + 1):
                if i:
                    sim.step_frames(1, record=False)
                st = sim.export_state()
                xs.append(st["x"])
                Fs.append(np.asarray(st["F"]).reshape(len(o["x0"]), 9)[sel])      # PhysCtrl stores F, C as 9 values
                Cs.append(np.asarray(st["C"]).reshape(len(o["x0"]), 9)[sel])
            x = np.stack(xs, 0)
        except Exception as ex:
            if cuda_context_lost(ex):
                append_jsonl({**rec, "reason": f"crashed (CUDA context lost): {type(ex).__name__}"}, flog)
                exit_cuda_lost(sc, ex)
            raise
        if is_unstable(x) and not args.dry_run:
            append_jsonl({**rec, "valid": False, "reason": "joint rollout unstable"}, blog)
            continue
        xa = np.asarray(qa["x"], np.float64)
        sup = np.concatenate([xa, np.broadcast_to(qb["x0"] + t_b, (len(xa), nb, 3))], 1)
        dev = np.linalg.norm(x - sup, axis=2).mean(axis=1)[1::2][:24]
        upto = 24 if rec["kind"] == "control" else max(rec["k_c"] - 2, 0)
        pre_dev = float(dev[:upto].max()) if upto > 0 else 0.0
        valid = pre_dev <= SANITY_TOL
        out = {**rec, "valid": valid, "pre_dev": pre_dev, "contact_effect": float(dev[-1]),
               "reason": "" if valid else f"pre-contact deviation {pre_dev:.4f} > {SANITY_TOL}"}
        if valid or args.dry_run:          # dry run: write the files anyway, so the file layout is tested
            jp, mp = root / "joint" / split / f"{sc}.h5", root / "model" / split / f"{sc}.h5"
            write_model_h5(jp, {"x": x.astype(np.float32), "vol": o["vol"], "floor_height": o["floor_height"],
                                "drag_point": o["drag_point"], "drag_mask": o["drag_mask"], "drag_force": o["drag_force"],
                                "E": o["E"], "nu": o["nu"], "mat_type": 0, "gravity": 0, "object_id": o["object_id"],
                                "t_b": t_b, "k_c": rec["k_c"] if rec["k_c"] is not None else -1})
            c_a = drag_coeff(qa["vol"], ma)
            write_model_h5(mp, {"x": x[:, sel].astype(np.float32), "F": np.stack(Fs, 0).astype(np.float32),
                                "C": np.stack(Cs, 0).astype(np.float32), "vol": o["vol"][sel],
                                "floor_height": o["floor_height"], "drag_point": o["drag_point"],
                                "drag_mask": o["drag_mask"][:, sel], "drag_force": o["drag_force"] / c_a,
                                "base_drag_coeff": np.array([[c_a]], np.float32), "E": o["E"], "nu": o["nu"],
                                "mat_type": 0, "gravity": 0, "object_id": o["object_id"][sel], "t_b": t_b,
                                "k_c": rec["k_c"] if rec["k_c"] is not None else -1, "sel_a": sel_a, "sel_b": sel_b})
            entry = {"qid": sc, "path": str(mp), "joint_path": str(jp), "split": split, "mat_type": 0,
                     "kind": rec["kind"], "k_c": rec["k_c"], "set": rec["set"], "a_path": rec["a_path"]}
            if rec["set"] == "dev":
                rp = root / "rest" / split / f"{rec['b']}__rest.h5"
                if not rp.exists():
                    b = {k: v for k, v in qb.items() if k in ("vol", "floor_height", "drag_point", "drag_mask", "E", "nu")}
                    write_model_h5(rp, {**b, "x": np.broadcast_to(qb["x0"], (N_SIM_FRAMES + 1, nb, 3)).astype(np.float32),
                                        "F": np.broadcast_to(np.eye(3, dtype=np.float32).reshape(9), (N_SIM_FRAMES + 1, nb, 9)),
                                        "C": np.zeros((N_SIM_FRAMES + 1, nb, 9), np.float32),
                                        "drag_force": np.zeros((1, 3), np.float32), "mat_type": 0, "gravity": 0})
                entry["rest_path"] = str(rp)
            manifest.append(entry)
            save_json(manifest, mpath)
        append_jsonl(out, blog)
        print(f"[p15_build] {sc} {rec['kind']} k_c={rec['k_c']} pre_dev={pre_dev:.4f} effect={out['contact_effect']:.3f}",
              flush=True)
    save_json(manifest, mpath)
    run.finish("ok", n_scenes=len(manifest))


if __name__ == "__main__":
    main()
