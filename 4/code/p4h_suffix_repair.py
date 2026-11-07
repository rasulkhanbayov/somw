"""Phase 4h (A100): certified-prefix repair, re-simulating only the uncertified suffix.

Pre-registered Oct 13, 2026 (PLAN.md "Phase 4h"), before any data. For each Phase 2c base query:
  chain     the anchor (true initial state, model frames 0..3) and windows starting at model
            frames k in {3, 7, 11, 15, 19}, all coarse_cfl and m = 4. Their defects and A100
            seconds are read from the Phase 2c exhaustive scan; no window is re-run here.
  decision  a window passes if its defect < tau. j = the first failing window.
            anchor fails   -> 'full': a reference rollout from the true initial state. That
                              rollout IS the ground truth (Phase 0 replay error ~2e-6), so its
                              error is 0 and its cost is the Phase 1 reference time.
            none fail      -> 'accept': keep the generation.
            otherwise      -> 'suffix': keep generated model frames 0..k_j; restart the
                              simulator from the state lifted at model frame k_j out of the
                              generation (lift_noF) and simulate to the end at the reference
                              setting; model frames > k_j come from that rollout.
  floor     the same suffix restart from the ground-truth positions at k_j (the error of
            restarting with F = I).
The verdict against certify-or-simulate is computed in p4h_summarize.py.

Output: results/p4h_suffix/shard_*/suffix.jsonl (+ cuda_crashes.jsonl)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, read_jsonl, resume_rows, results_dir, shard_items, start_run, timed)
from backends import make_backend  # noqa: E402
from p2b_generate import all_entries, gen_dir  # noqa: E402
from verifier import generated_frames, gt_frames, nbrs_for, run_window  # noqa: E402

N_MODEL = 24


def chain_decision(scan_row: dict, chain_k: list[int], tau: float):
    """(index of the first failing window or None, model frame k_j, seconds of windows evaluated)."""
    win = {int(s): (np.inf if (b or c or raw is None) else float(raw))
           for s, raw, b, c in zip(scan_row["starts"], scan_row["raw"], scan_row["blown"], scan_row["crashed"])}
    sec = {int(s): float(t or 0.0) for s, t in zip(scan_row["starts"], scan_row["seconds"])}
    starts = [0] + [2 * k + 1 for k in chain_k]          # anchor first
    spent = 0.0
    for i, s in enumerate(starts):
        spent += sec[s]
        if not win[s] < tau:
            return i, (0 if i == 0 else chain_k[i - 1]), spent
    return None, None, spent


def suffix_rollout(backend, q, frames, k, lift, nbrs):
    """Restart at model frame k (sim frame 2k+1) and simulate to sim frame 47 at 'ref'."""
    s = 2 * k + 1
    rec = {}
    with timed(rec, "t"):
        w = run_window(backend, q, frames, s, 47 - s, lift, n_perturb=0, nbrs=nbrs, return_frames=True)
    sim = {(f - 1) // 2: w["frames"][f] for f in w["frames"] if f % 2 == 1}
    return sim, rec["t"], int(w["substeps"]), bool(w["blown"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4h_suffix"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("suffix.jsonl")
    done = {r["qid"] for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {r["qid"] for r in read_jsonl(ledger)} if ledger.exists() else set()
    spec, tau, chain_k, lift = cfg["window_spec"], float(cfg["tau"]), list(cfg["chain_k"]), cfg["lift"]
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"], args.dry_run).glob("shard_*/scan.jsonl"))
            for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == spec["fidelity"] and int(r["m"]) == spec["m"]}
    root = gen_dir(cfg, args.dry_run)
    for e in shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards):
        if e["qid"] in done or e["qid"] not in scan:
            continue
        if e["qid"] in crashed:
            append_jsonl({"qid": e["qid"], "split": e["split"], "crashed": True}, out)
            continue
        q = backend.load(e)
        gt = q["x"][1::2][:N_MODEL]
        g0 = np.load(root / "base" / e["split"] / f"{e['qid']}.npz")["samples"][0]
        j, kj, probe_s = chain_decision(scan[e["qid"]], chain_k, tau)
        row = {"qid": e["qid"], "split": e["split"], "j": j, "k_j": kj, "probe_seconds": probe_s,
               "gen_err_t": np.linalg.norm(g0 - gt, axis=2).mean(axis=1).tolist()}
        try:
            if j is None:
                row.update(mode="accept", err_t=row["gen_err_t"], suffix_seconds=0.0)
            elif j == 0:
                row.update(mode="full", err_t=[0.0] * N_MODEL, suffix_seconds=None)
            else:
                nbrs = nbrs_for(q, lift["k"])
                sim, secs, subs, blown = suffix_rollout(backend, q, generated_frames(q["x0"], g0), kj, lift, nbrs)
                rep = g0.copy()
                for k, x in sim.items():
                    if k > kj:
                        rep[k] = x
                fsim, _, _, fblown = suffix_rollout(backend, q, gt_frames(q["x"], odd_only=True), kj, lift, nbrs)
                flo = gt.copy()
                for k, x in fsim.items():
                    if k > kj:
                        flo[k] = x
                row.update(mode="suffix", suffix_seconds=secs, suffix_substeps=subs, suffix_blown=blown,
                           err_t=np.linalg.norm(rep - gt, axis=2).mean(axis=1).tolist(),
                           floor_err_t=np.linalg.norm(flo - gt, axis=2).mean(axis=1).tolist(), floor_blown=fblown)
        except Exception as ex:
            if cuda_context_lost(ex):
                append_jsonl({"qid": e["qid"], "error": f"{type(ex).__name__}: {ex}"}, ledger)
                exit_cuda_lost(e["qid"], ex)
            raise
        append_jsonl(row, out)
        print(f"[p4h] {e['qid']} {row['mode']} j={j}")
    run.finish("ok")


if __name__ == "__main__":
    main()
