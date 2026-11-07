"""Phase 4e (H200): anchored conditioning repair, selection tied to physics truth.

Why (Phase 4c): choosing the conditioning by the window defect was a winner's curse, because the
defect barely varies across conditioning candidates. Here the window defect is not used at all.
The objective is agreement with an exact simulation from the TRUE initial state over the first
L model frames (an "anchor" of length L), which is the only trajectory-level truth available
without a full rollout.

Pre-registered Oct 12, 2026 (PLAN.md "Phase 4e"), before any result. For each Phase 2c query
and model:
  candidates  the same 30 conditioning adjustments, shared noise and seed as Phase 4c
              (p4c_repair_conditioning.grid), so the candidates are identical to 4c's.
  anchors     one rollout from the true initial state at the fallback fidelity (coarse_cfl),
              for 2L - 1 simulator frames, i.e. model frames 0 .. L-1; L in {8 (primary), 12}.
  objective   J_L(c) = mean per-point L2 between c's model frames 0..L-1 and the anchor.
  outputs     R5_L = argmin_c J_L (identity first, so ties go to it);
              R6_L = R5_L with model frames 0..L-1 replaced by the anchor frames.
              Per-frame errors to ground truth, the chosen adjustment, J_L for all 30, the
              anchor's own error, and the cost (anchor substeps).
Falsifier (judged in p4e_summarize.py): on base trajectories rejected by the Phase 4a primary
certificate (the Phase 4b/4c population), mean err(R1) - err(R6_8) has 95% paired-bootstrap
lower bound <= 0.

Output: results/p4e_repair/shard_*/repair.jsonl (+ cuda_crashes.jsonl)
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
from p2b_generate import all_entries  # noqa: E402
from p4b_repair import per_frame_err  # noqa: E402
from p4c_repair_conditioning import grid  # noqa: E402
from verifier import generated_frames, resolve_fidelity, run_window, verified_model_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4e_repair"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("repair.jsonl")
    done = {(r["qid"], r["model"]) for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {(r["qid"], r["model"]) for r in read_jsonl(ledger)} if ledger.exists() else set()
    adj, lift, Ls = grid(cfg), cfg["lift"], sorted(cfg["anchor_lengths"])

    for e in shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards):
        q, anchors = None, None
        for model in cfg["models"]:
            if (e["qid"], model) in done:
                continue
            if (e["qid"], model) in crashed:
                append_jsonl({"qid": e["qid"], "split": e["split"], "model": model, "crashed": True}, out)
                continue
            q = q or backend.load(e)
            try:
                gt = q["x"][1::2][:24]
                if anchors is None:     # the anchor depends only on the query, not on the model
                    ov = resolve_fidelity(cfg["anchor_fidelity"], q)
                    Lmax = Ls[-1]
                    # The frames argument only supplies a window-end target whose defect is not
                    # used; zeros keep ground truth out of the selection path entirely.
                    aw = run_window(backend, q, generated_frames(q["x0"], np.zeros_like(gt)), 0, 2 * Lmax - 1, lift,
                                    n_perturb=0, sim_overrides=ov or None, anchor=True, return_frames=True)
                    ks = verified_model_frames(0, 2 * Lmax - 1)
                    full = np.stack([aw["frames"][2 * k + 1] for k in ks], 0)
                    anchors = {"frames": full, "blown": bool(aw["blown"]), "substeps": int(aw["substeps"]),
                               "err_t": np.linalg.norm(full - gt[: len(ks)], axis=2).mean(axis=1).tolist()}
                seed = int(cfg["seed"] + zlib.crc32(e["qid"].encode()) % 100000)    # = Phase 4b/4c seed
                cands = backend.generate_conditioned(model, q, seed, adj)
                row = {"qid": e["qid"], "split": e["split"], "model": model, "seed": seed,
                       "anchor_blown": anchors["blown"], "anchor_substeps": anchors["substeps"],
                       "anchor_err_t": anchors["err_t"]}
                for L in Ls:
                    a = anchors["frames"][:L]
                    J = [float(np.linalg.norm(c[:L] - a, axis=2).mean()) for c in cands]
                    best = 0 if anchors["blown"] else int(np.argmin(J))
                    r5 = cands[best]
                    r6 = r5.copy()
                    if not anchors["blown"]:
                        r6[:L] = a
                    row.update({f"L{L}__chosen": adj[best], f"L{L}__J": J,
                                f"R5_L{L}__err_t": per_frame_err(r5, gt), f"R6_L{L}__err_t": per_frame_err(r6, gt)})
            except Exception as ex:
                if cuda_context_lost(ex):
                    append_jsonl({"qid": e["qid"], "model": model, "error": f"{type(ex).__name__}: {ex}"}, ledger)
                    exit_cuda_lost(f"{e['qid']} {model}", ex)
                raise
            append_jsonl(row, out)
        print(f"[p4e] {e['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
