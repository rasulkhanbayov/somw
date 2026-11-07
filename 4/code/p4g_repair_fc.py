"""Phase 4g evaluation (H200): repair with the frame-conditioned generator.

Pre-registered Oct 12, 2026 (PLAN.md "Phase 4g"). The Phase 2c queries and base generations,
with the SAME constraints and seeds as Phase 4b (the anchor from the true initial state plus
the certificate's spread_event windows, simulated at 'ref'):
  R7   fine-tuned model, all constraints
  R7u  fine-tuned model, no constraints (does fine-tuning alone change anything?)
  R8   fine-tuned model, anchor constraints only (extrapolation from an exact start)
plus the certificate's defect of R7 (coarse_cfl, m = 4, spread_event). R0/R1/R2 come from the
Phase 4b rows. Verdict: p4g_summarize.py.

Output: results/p4g_repair/shard_*/repair.jsonl (+ cuda_crashes.jsonl)
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
from p4b_repair import constraints, per_frame_err, verify  # noqa: E402
from verifier import generated_frames, nbrs_for, resolve_fidelity  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4g_repair"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("repair.jsonl")
    done = {(r["qid"], r["model"]) for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {(r["qid"], r["model"]) for r in read_jsonl(ledger)} if ledger.exists() else set()
    ck = None if args.dry_run else str(Path(cfg["paths"]["ckpt_root"]) / cfg["ckpt_name"] / "final.pt")
    if ck is not None and not Path(ck).exists():
        raise FileNotFoundError(f"fine-tuned checkpoint missing: {ck} (run 47_p4g_train first)")
    root, lift, m, model = gen_dir(cfg, args.dry_run), cfg["lift"], cfg["m"], "base"
    for e in shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards):
        if (e["qid"], model) in done:
            continue
        if (e["qid"], model) in crashed:
            append_jsonl({"qid": e["qid"], "split": e["split"], "model": model, "crashed": True}, out)
            continue
        q = backend.load(e)
        try:
            gt = q["x"][1::2][:24]
            samples = np.load(root / model / e["split"] / f"{e['qid']}.npz")["samples"]
            r0 = samples[0]
            nbrs = nbrs_for(q, lift["k"])
            ref_ov = resolve_fidelity(cfg["constraint_fidelity"], q)
            ver_ov = resolve_fidelity(cfg["verify_fidelity"], q)
            starts = starts_for("spread_event", samples, q, m)
            known, mask, amask, n_blown = constraints(backend, q, generated_frames(q["x0"], r0), starts, m,
                                                      lift, ref_ov, nbrs)
            seed = int(cfg["seed"] + zlib.crc32(e["qid"].encode()) % 100000)       # = Phase 4b's seed
            gc = lambda kn, mk: backend.generate_constrained(model, q, 1, seed, kn, mk, fc_ckpt=ck)[0]
            r7, r7u, r8 = gc(known, mask), gc(None, None), gc(known, amask)
            row = {"qid": e["qid"], "split": e["split"], "model": model, "seed": seed,
                   "constrained": np.nonzero(mask)[0].tolist(), "anchor_frames": np.nonzero(amask)[0].tolist(),
                   "n_blown_constraint_windows": n_blown,
                   "R7__err_t": per_frame_err(r7, gt), "R7u__err_t": per_frame_err(r7u, gt),
                   "R8__err_t": per_frame_err(r8, gt),
                   "R7__defect": verify(backend, q, r7, m, lift, ver_ov, nbrs)}
        except Exception as ex:
            if cuda_context_lost(ex):
                append_jsonl({"qid": e["qid"], "model": model, "error": f"{type(ex).__name__}: {ex}"}, ledger)
                exit_cuda_lost(f"{e['qid']} {model}", ex)
            raise
        append_jsonl(row, out)
        print(f"[p4g] {e['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
