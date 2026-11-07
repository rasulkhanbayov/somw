"""Phase 4d data (A100): one full rollout per query from the TRUE initial state, at the fallback
fidelity of the certify-or-simulate pipeline (default: coarse_cfl, the CFL-adaptive cheap setting).

Records each query's per-frame error against the reference-fidelity ground truth (model frames,
i.e. simulator frames 1, 3, ..., 47), substeps, synced wall-clock and stability. These are the
error and cost of the fallback, which p4d_pipeline.py combines with the stored generations,
certificate scores and Phase 1 costs. The queries are the Phase 2c splits and the calibration
set (config 'entry_sets').

Output: results/p4d_fullsim/shard_*/fullsim.jsonl (+ cuda_crashes.jsonl)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, read_jsonl, resume_rows, shard_items, start_run, timed)
from backends import make_backend, N_SIM_FRAMES  # noqa: E402
from p2b_generate import all_entries  # noqa: E402
from verifier import resolve_fidelity  # noqa: E402


def entries_from_sets(cfg, backend, dry_run):
    out, seen = [], set()
    for es in cfg["entry_sets"]:
        for e in all_entries({**cfg, **es}, backend, dry_run):
            if e["qid"] not in seen:
                seen.add(e["qid"])
                out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4d_fullsim"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    from physctrl_bridge import is_unstable

    out = run.path("fullsim.jsonl")
    done = {(r["qid"], r["fidelity"]) for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {(r["qid"], r["fidelity"]) for r in read_jsonl(ledger)} if ledger.exists() else set()
    for e in shard_items(entries_from_sets(cfg, backend, args.dry_run), shard, args.num_shards):
        q = None
        for fid, spec in cfg["fidelities"].items():
            if (e["qid"], fid) in done:
                continue
            row = {"qid": e["qid"], "split": e["split"], "fidelity": fid}
            if (e["qid"], fid) in crashed:
                append_jsonl({**row, "crashed": True, "stable": False}, out)
                continue
            q = q or backend.load(e)
            ov = resolve_fidelity(spec, q)
            try:
                sim = backend.make_sim(q, overrides=ov or None)
                rec = {}
                with timed(rec, "rollout_s"):
                    traj = sim.step_frames(N_SIM_FRAMES)
            except Exception as ex:
                if cuda_context_lost(ex):
                    append_jsonl({**row, "error": f"{type(ex).__name__}: {ex}"}, ledger)
                    exit_cuda_lost(f"{e['qid']} {fid}", ex)
                raise
            gt = q["x"][1::2][:24]
            stable = not is_unstable(traj)
            row.update({"substep_dt": float(ov.get("substep_dt", 1e-4)), "n_grid": int(ov.get("n_grid", 125)),
                        "substeps": int(sim.substeps_done), "rollout_s": rec["rollout_s"], "stable": bool(stable),
                        "err_t": (np.linalg.norm(traj[0::2][:24] - gt, axis=2).mean(axis=1).tolist()
                                  if stable else None)})
            append_jsonl(row, out)
        print(f"[p4d_fullsim] {e['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
