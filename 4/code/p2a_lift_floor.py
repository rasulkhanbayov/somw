"""Phase 2a: the lift floor. How large is the window defect on PERFECT trajectories?

Ground-truth trajectories are pushed through exactly the verifier used on
generated ones. Whatever defect remains is error from lifting the state out of
positions (plus solver non-determinism), i.e. the smallest model error the
verifier can detect. Variants isolate each source:

  oracle_state   restart from the stored x, v, F, C           -> solver floor (should be ~0)
  lift           positions at odd frames only (what a generator exposes)
  lift_dense_t   positions at every frame (isolates temporal sampling of v)
  lift_noC       C set to zero
  lift_noF       F set to identity
  lift_k6/k24    neighbourhood size for the F and C fits

Also records lifted-state errors against the stored state:
  |v_lift - v| / |v|, |F_lift - F|_F, |C_lift - C|_F / |C|_F.

Falsifier (checked in p2_summarize.py): the floor for `lift` is not clearly
below the defect measured on generated trajectories (Phase 2b), so the
verifier cannot see model error over its own lifting error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, resume_rows, shard_items, start_run)
from backends import make_backend, ood_manifest_entries, testset_manifest_path  # noqa: E402
from lift import knn, lift_state  # noqa: E402
from verifier import gt_frames, run_window  # noqa: E402

VARIANTS = {
    "oracle_state": None,
    "lift": {"k": 12, "use_F": True, "use_C": True, "dense_t": False},
    "lift_dense_t": {"k": 12, "use_F": True, "use_C": True, "dense_t": True},
    "lift_noC": {"k": 12, "use_F": True, "use_C": False, "dense_t": False},
    "lift_noF": {"k": 12, "use_F": False, "use_C": True, "dense_t": False},
    "lift_k6": {"k": 6, "use_F": True, "use_C": True, "dense_t": False},
    "lift_k24": {"k": 24, "use_F": True, "use_C": True, "dense_t": False},
}


def state_errors(lifted: dict, q: dict, s: int) -> dict:
    out = {}
    if q.get("v") is not None:
        v = np.asarray(q["v"][s])
        out["v_rel"] = float(np.linalg.norm(lifted["v"] - v) / (np.linalg.norm(v) + 1e-12))
    if q.get("F") is not None:
        F = np.asarray(q["F"][s]).reshape(-1, 9)
        out["F_fro_mean"] = float(np.linalg.norm(lifted["F"] - F, axis=1).mean())
    if q.get("C") is not None:
        C = np.asarray(q["C"][s]).reshape(-1, 9)
        out["C_fro_mean"] = float(np.linalg.norm(lifted["C"] - C, axis=1).mean())
        out["C_true_fro_mean"] = float(np.linalg.norm(C, axis=1).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run("p2a_lift_floor", cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("lift_floor.jsonl")
    done = {(r["qid"], r["variant"], r["start_sim_frame"], r["m"])
            for r in resume_rows(out, args.resume)}

    entries = backend.manifest(testset_manifest_path(args.dry_run), split="test_id")[: cfg["n_queries"]]
    if cfg.get("include_ood"):
        try:
            ood = ood_manifest_entries(args.dry_run, backend)
            per = cfg.get("n_ood_per_split", 10)
            for split in sorted({e["split"] for e in ood}):
                entries += [e for e in ood if e["split"] == split][:per]
        except FileNotFoundError as ex:
            print(f"[p2a] OOD skipped: {ex}")
    entries = shard_items(entries, shard, args.num_shards)

    variants = cfg["variants"]
    if args.dry_run:
        variants = [v for v in variants if v != "oracle_state"]   # spring GT stores no velocities
    frame_dt = backend.frame_dt()      # simulated seconds per stored frame
    for e in entries:
        q = backend.load(e)
        rest = np.asarray(q["x0"], dtype=np.float64)
        odd = gt_frames(q["x"], odd_only=True)
        dense = gt_frames(q["x"], odd_only=False)
        nb_cache = {}
        for k_start in cfg["window_starts_model_frames"]:
            s = 2 * k_start + 1
            for m in cfg["window_lengths_model_frames"]:
                if s + 2 * m > 47:
                    continue
                for vname in variants:
                    vc = VARIANTS[vname]
                    row = {"qid": q["qid"], "split": q["split"], "variant": vname,
                           "start_sim_frame": s, "m": m}
                    if (q["qid"], vname, s, m) in done:
                        continue
                    try:
                        if vc is None:
                            state = {"x": q["x"][s], "v": q["v"][s], "F": q["F"][s], "C": q["C"][s]}
                            frames = dense
                        else:
                            frames = dense if vc["dense_t"] else odd
                            if vc["k"] not in nb_cache:
                                nb_cache[vc["k"]] = knn(rest, vc["k"])
                            state = lift_state(frames, s, frame_dt, rest, k=vc["k"],
                                               use_F=vc["use_F"], use_C=vc["use_C"],
                                               nbrs=nb_cache[vc["k"]])
                            row.update(state_errors(state, q, s))
                        w = run_window(backend, q, odd, s, 2 * m, {}, n_perturb=cfg["n_perturb"],
                                       perturb_cfg=cfg["perturb"], state=state, seed=k_start * 100 + m)
                        row.update({k: w[k] for k in ("raw", "raw_p90", "sigma", "norm", "substeps", "finite",
                                                      "blown", "blown_perturb")})
                    except Exception as ex:
                        row["error"] = f"{type(ex).__name__}: {ex}"
                        if cuda_context_lost(ex):
                            row["crashed"] = True
                            append_jsonl(row, out)
                            exit_cuda_lost(f"{q['qid']} {vname} s={s} m={m}", ex)
                    append_jsonl(row, out)
        print(f"[p2a] {q['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
