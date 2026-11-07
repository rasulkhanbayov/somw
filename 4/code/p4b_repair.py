"""Phase 4b (H200): training-free repair by constrained re-sampling.

Pre-registered Oct 11, 2026 (PLAN.md "Phase 4b"), before any repair result. For each Phase 2c
query and each model (sample 0 of the stored generation):

  constraints  the frames verified by simulation:
               - the anchor window from the TRUE initial state (model frames 0 .. m-1);
               - the certificate's spread_event windows (m = 4), each restarted from the
                 state lifted out of the generated trajectory.
               All are run at the reference fidelity; blown windows give no constraint.
  methods      R0   the original generation (no repair)
               R0p  an unconstrained re-sample with the repair seed (re-sampling alone)
               R1   replace: write the verified frames into R0 (the naive baseline)
               R2   constrained re-sample with all constraints (the method)
               R3   constrained re-sample with the anchor window only (ablation)
  metrics      per-frame error to ground truth (24 values) per method; after-repair defect
               of R1 and R2 with the certificate's own signal (coarse_cfl, m = 4,
               spread_event) and acceptance at the Phase 4a lambda (base).
Falsifier (judged in p4b_summarize.py): on base trajectories rejected by the Phase 4a primary
certificate, constrained re-sampling (R2) does not lower mean error compared with replacement
(R1); paired bootstrap, 95% lower bound of mean(R1 - R2) <= 0.

Before any repair, the job checks on the GPU that the unconstrained loop reproduces PhysCtrl's
own pipeline output for the same seed, and stops if it does not.

Output: results/p4b_repair/shard_*/repair.jsonl (+ cuda_crashes.jsonl)
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, read_jsonl, resume_rows, shard_items, start_run)
from backends import make_backend  # noqa: E402
from p2b_generate import all_entries, gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from verifier import generated_frames, nbrs_for, resolve_fidelity, run_window, verified_model_frames  # noqa: E402


def per_frame_err(traj: np.ndarray, gt: np.ndarray) -> list[float]:
    return np.linalg.norm(traj - gt, axis=2).mean(axis=1).astype(float).tolist()


def constraints(backend, q, frames, starts, m, lift, overrides, nbrs):
    """Run the anchor + listed windows; return (known (24,N,3), mask (24,), anchor mask, n_blown)."""
    n = len(q["x0"])
    known, mask, amask = np.zeros((24, n, 3), np.float32), np.zeros(24, bool), np.zeros(24, bool)
    blown = 0
    wins = [(0, 2 * m - 1, True)] + [(s, 2 * m, False) for s in starts]
    for s, L, anchor in wins:
        w = run_window(backend, q, frames, s, L, lift, n_perturb=0, sim_overrides=overrides or None,
                       anchor=anchor, nbrs=nbrs, return_frames=True)
        if w["blown"]:
            blown += 1
            continue
        for k in verified_model_frames(s, L):
            known[k], mask[k] = w["frames"][2 * k + 1], True
            if anchor:
                amask[k] = True
    return known, mask, amask, blown


def verify(backend, q, traj, m, lift, overrides, nbrs) -> float:
    """The certificate's signal on a (repaired) trajectory: max raw defect, +inf if blown."""
    frames = generated_frames(q["x0"], traj)
    vals = []
    for s in starts_for("spread_event", traj[None], q, m):
        w = run_window(backend, q, frames, s, 2 * m, lift, n_perturb=0, sim_overrides=overrides or None, nbrs=nbrs)
        vals.append(np.inf if w["blown"] else w["raw"])
    return float(max(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4b_repair"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("repair.jsonl")
    done = {(r["qid"], r["model"]) for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {(r["qid"], r["model"]) for r in read_jsonl(ledger)} if ledger.exists() else set()
    entries = shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards)
    root, lift, m = gen_dir(cfg, args.dry_run), cfg["lift"], cfg["m"]

    checked = False
    for e in entries:
        q = None
        for model in cfg["models"]:
            if (e["qid"], model) in done:
                continue
            q = q or backend.load(e)
            if not checked:
                ref = backend.generate_reference(model, q, 2, 123)
                mine = backend.generate_constrained(model, q, 2, 123)
                diff = float(np.abs(ref - mine).max())
                print(f"[p4b] equivalence check ({model}, {e['qid']}): max |pipeline - our loop| = {diff:.2e}")
                if diff > cfg["equivalence_tol"]:
                    raise RuntimeError(f"constrained sampler does not reproduce PhysCtrl's pipeline ({diff:.2e})")
                checked = True
            if (e["qid"], model) in crashed:
                append_jsonl({"qid": e["qid"], "split": e["split"], "model": model, "crashed": True}, out)
                continue
            try:
                gt = q["x"][1::2][:24]
                samples = np.load(root / model / e["split"] / f"{e['qid']}.npz")["samples"]
                r0 = samples[0]
                frames = generated_frames(q["x0"], r0)
                nbrs = nbrs_for(q, lift["k"])
                ref_ov = resolve_fidelity(cfg["constraint_fidelity"], q)
                ver_ov = resolve_fidelity(cfg["verify_fidelity"], q)
                starts = starts_for("spread_event", samples, q, m)
                known, mask, amask, n_blown = constraints(backend, q, frames, starts, m, lift, ref_ov, nbrs)
                seed = int(cfg["seed"] + zlib.crc32(e["qid"].encode()) % 100000)
                r1 = r0.copy()
                r1[mask] = known[mask]
                r0p = backend.generate_constrained(model, q, 1, seed)[0]
                r2 = backend.generate_constrained(model, q, 1, seed, known, mask)[0]
                r3 = backend.generate_constrained(model, q, 1, seed, known, amask)[0]
                row = {"qid": e["qid"], "split": e["split"], "model": model,
                       "constrained": np.nonzero(mask)[0].tolist(), "anchor_frames": np.nonzero(amask)[0].tolist(),
                       "n_blown_constraint_windows": n_blown, "seed": seed}
                for name, traj in (("R0", r0), ("R0p", r0p), ("R1", r1), ("R2", r2), ("R3", r3)):
                    row[f"{name}__err_t"] = per_frame_err(traj, gt)
                for name, traj in (("R0", r0), ("R1", r1), ("R2", r2)):
                    row[f"{name}__defect"] = verify(backend, q, traj, m, lift, ver_ov, nbrs)
            except Exception as ex:
                if cuda_context_lost(ex):
                    append_jsonl({"qid": e["qid"], "model": model, "error": f"{type(ex).__name__}: {ex}"}, ledger)
                    exit_cuda_lost(f"{e['qid']} {model}", ex)
                raise
            append_jsonl(row, out)
        print(f"[p4b] {e['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
