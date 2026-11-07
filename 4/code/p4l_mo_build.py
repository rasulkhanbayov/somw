"""Phase 4l (H200): build the two-object contact scenes from the single-object components.

Pre-registered Oct 15, 2026 (PLAN.md "Phase 4l"). Needs the component ground truths
(p2_build_ood.py on mo_comp_<j>) and their base generations (p2b_generate.py). Per scene:
  design     B's translation from the isolated component ground truths (mo_scenes.design)
  ground     one joint rollout, 4,096 particles in one grid (PhysCtrl's settings), both drags
  sanity     invalid if, before model frame k_c - 2 (controls: all frames), the joint ground truth
             deviates from the superposed isolated ground truths by > 0.02 (mean point distance)
  compose    base: sample s of A + sample s of B (translated); gt_super: superposed isolated truths
Writes scene HDF5 (+ object_id, t_b, k_c), the two composed generations, a manifest the scan /
probe / all-k scripts read, and design.jsonl (every scene, valid or not, with its reason).

Output: $ood_dir/mo_<j>/*.h5; $gen_dir/{base,gt_super}/mo_<j>/*.npz;
        results/p4l_scene_build/shard_*/{ood_manifest.json, design.jsonl, failures.jsonl}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, load_json, read_jsonl, results_dir, resume_rows, save_json, shard_items, start_run)
from backends import make_backend, query_from_h5, N_SIM_FRAMES  # noqa: E402
from mo_scenes import compose, design, pairing, scene_physics  # noqa: E402

SANITY_TOL = 0.02


def comp_index(runs, dry_run):
    out = {}
    for r in runs:
        for f in sorted(results_dir(r, dry_run).glob("shard_*/ood_manifest.json")):
            for e in load_json(f):
                out[e["qid"]] = e
    return out


def joint_query(qa: dict, qb: dict, t_b, scene: str, j: int) -> dict:
    na, nb = len(qa["x0"]), len(qb["x0"])
    E, nu = scene_physics(scene)
    t_b = np.asarray(t_b, dtype=np.float32)

    def pad(mask, left):
        m = np.asarray(mask).reshape(-1, na if left else nb)
        z = np.zeros((m.shape[0], nb if left else na), dtype=m.dtype)
        return np.concatenate([m, z], 1) if left else np.concatenate([z, m], 1)
    return {"qid": scene, "split": f"mo_{j}", "path": None,
            "x0": np.concatenate([qa["x0"], qb["x0"] + t_b]).astype(np.float32),
            "vol": np.concatenate([qa["vol"], qb["vol"]]).astype(np.float32),
            "floor_height": float(qa["floor_height"]), "gravity": 0, "mat_type": 0, "E": E, "nu": nu,
            "drag_point": np.concatenate([np.asarray(qa["drag_point"]).reshape(-1, 3),
                                          np.asarray(qb["drag_point"]).reshape(-1, 3) + t_b]),
            "drag_mask": np.concatenate([pad(qa["drag_mask"], True), pad(qb["drag_mask"], False)]),
            "drag_force": np.concatenate([np.asarray(qa["drag_force"]).reshape(-1, 3),
                                          np.asarray(qb["drag_force"]).reshape(-1, 3)]),
            "object_id": np.concatenate([np.zeros(na, np.int64), np.ones(nb, np.int64)])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4l_scene_build"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    from physctrl_bridge import is_unstable, write_h5

    comps = comp_index(cfg["comp_build_runs"], args.dry_run)
    obj_ids = sorted({q.split("__")[0] for q in comps})
    recs = [r for j in cfg["draws"] for r in pairing(obj_ids, j)]
    recs = shard_items(recs, shard, args.num_shards)
    out_dir, groot = Path(cfg["paths"]["ood_dir"]), Path(cfg["paths"]["gen_dir"])
    cgroot = Path(cfg["paths"]["comp_gen_dir"])
    dlog, flog = run.path("design.jsonl"), run.path("failures.jsonl")
    done = {r["scene"] for r in resume_rows(dlog, args.resume)}
    failed = {r["scene"] for r in resume_rows(flog, args.resume)}
    manifest = [m for m in (load_json(run.path("ood_manifest.json")) if args.resume and run.path("ood_manifest.json").exists() else [])]
    for rec in recs:
        sc, j = rec["scene"], rec["j"]
        if sc in done or sc in failed:
            continue
        ca, cb = f"{rec['a']}__mo_comp_{j}", f"{rec['b']}__mo_comp_{j}"
        if ca not in comps or cb not in comps:
            append_jsonl({**rec, "valid": False, "reason": "component missing (unstable build)"}, dlog)
            continue
        qa = query_from_h5(comps[ca]["path"], comps[ca]["split"], ca)        # real files, also in dry runs
        qb = query_from_h5(comps[cb]["path"], comps[cb]["split"], cb)
        d = design(rec, np.asarray(qa["x"], np.float64), np.asarray(qb["x"], np.float64))
        if not d["valid"]:
            append_jsonl(d, dlog)
            continue
        o = joint_query(qa, qb, d["t_b"], sc, j)
        try:
            sim = backend.make_sim(o, timing_mode=cfg["mode"])
            xs, vs, Fs, Cs = [], [], [], []
            st = sim.export_state()
            xs.append(st["x"]); vs.append(st["v"]); Fs.append(st["F"]); Cs.append(st["C"])
            for _ in range(N_SIM_FRAMES):
                sim.step_frames(1, record=False)
                st = sim.export_state()
                xs.append(st["x"]); vs.append(st["v"]); Fs.append(st["F"]); Cs.append(st["C"])
            x = np.stack(xs, 0)
        except Exception as ex:
            if cuda_context_lost(ex):
                append_jsonl({**rec, "reason": f"crashed (CUDA context lost): {type(ex).__name__}"}, flog)
                exit_cuda_lost(sc, ex)
            raise
        if is_unstable(x):
            append_jsonl({**d, "valid": False, "reason": "joint rollout unstable"}, dlog)
            continue
        sup = compose(np.asarray(qa["x"]), np.asarray(qb["x"]), d["t_b"])          # (49, 4096, 3)
        dev = np.linalg.norm(x - sup, axis=2).mean(axis=1)[1::2][:24]             # per model frame
        upto = 24 if rec["kind"] == "control" else max(d["k_c"] - 2, 0)
        pre_dev = float(dev[:upto].max()) if upto > 0 else 0.0
        valid = pre_dev <= SANITY_TOL
        d.update(valid=valid, pre_dev=pre_dev, contact_effect=float(dev[-1]),
                 reason="" if valid else f"pre-contact deviation {pre_dev:.4f} > {SANITY_TOL}")
        path = out_dir / f"mo_{j}" / f"{sc}.h5"
        write_h5(path, {"x": x, "v": np.stack(vs, 0), "F": np.stack(Fs, 0), "C": np.stack(Cs, 0),
                        "vol": o["vol"], "floor_height": o["floor_height"], "drag_point": o["drag_point"],
                        "drag_mask": o["drag_mask"], "drag_force": o["drag_force"], "E": o["E"], "nu": o["nu"],
                        "mat_type": 0, "gravity": 0, "object_id": o["object_id"], "t_b": d["t_b"],
                        "k_c": d["k_c"] if d["k_c"] is not None else -1})
        ga = np.load(cgroot / "base" / f"mo_comp_{j}" / f"{ca}.npz")["samples"]
        gb = np.load(cgroot / "base" / f"mo_comp_{j}" / f"{cb}.npz")["samples"]
        for src, arr in (("base", compose(ga, gb, d["t_b"]).astype(np.float32)),
                         ("gt_super", sup[1::2][:24][None].astype(np.float32))):
            p = groot / src / f"mo_{j}" / f"{sc}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, samples=arr)
        if valid:
            manifest.append({"qid": sc, "path": str(path), "split": f"mo_{j}", "mat_type": 0,
                             "kind": rec["kind"], "k_c": d["k_c"]})
            save_json(manifest, run.path("ood_manifest.json"))
        append_jsonl(d, dlog)
        print(f"[p4l_build] {sc} {rec['kind']} k_c={d['k_c']} pre_dev={pre_dev:.4f} end_effect={d['contact_effect']:.3f}")
    save_json(manifest, run.path("ood_manifest.json"))
    run.finish("ok", n_scenes=len(manifest))


if __name__ == "__main__":
    main()
