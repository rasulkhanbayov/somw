"""Phase 4c (H200): conditioning repair, re-generating with physical inputs chosen by the verifier.

Why (Phase 4b): x0 replacement barely moved the generator's error, which is mostly a GLOBAL
wrong response to stiffness or force (results/diag_p2). This repair changes what the model is
told instead of overwriting frames: it searches the generator's conditioning and keeps the
trajectory the exact simulator finds most consistent.

Pre-registered Oct 11, 2026 (PLAN.md "Phase 4c"), before any result. For each Phase 2c query
and model (the same queries, generations and seeds as Phase 4b):
  candidates  a fixed grid of 30 conditioning adjustments, generated in one batch from SHARED
              noise (seed = Phase 4b's repair seed), so they differ only in conditioning:
              dlog10E in {-1, -0.5, 0, +0.5, +1} x force scale in {0.33, 0.5, 0.75, 1, 1.5, 2}.
  selection   score(c) = max(certificate defect of c [coarse_cfl, m = 4, spread_event],
                             anchor mismatch of c [mean L2 between c's model frames 0..3 and
                             the anchor window simulated from the TRUE initial state at 'ref']).
              Blown windows score +inf. R4 = argmin (ties go to the earlier grid entry; the
              identity comes first). Ground truth is never used.
  outputs     per-frame error to ground truth of R4 and of R4id (the identity candidate, i.e.
              re-sampling alone with the shared noise); the chosen adjustment; all 30 scores;
              a HELD-OUT defect (uniform placement, 'ref', m = 4: not used in selection) for
              R0 and R4.
Falsifier (judged in p4c_summarize.py): on base trajectories rejected by the Phase 4a primary
certificate (R0 defect >= 0.03007, from the Phase 4b rows), conditioning repair does not beat
replacement: mean err(R1) - err(R4) has 95% paired-bootstrap lower bound <= 0.

Output: results/p4c_repair/shard_*/repair.jsonl (+ cuda_crashes.jsonl)
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
from p4b_repair import per_frame_err, verify  # noqa: E402
from verifier import generated_frames, nbrs_for, resolve_fidelity, run_window, verified_model_frames  # noqa: E402


def grid(cfg) -> list[dict]:
    """Identity first, then the rest in a fixed order (ties resolve toward the identity)."""
    g = [{"dlogE": float(d), "force_scale": float(s)} for d in cfg["dlogE"] for s in cfg["force_scale"]]
    ident = [a for a in g if a["dlogE"] == 0.0 and a["force_scale"] == 1.0]
    if len(ident) != 1:
        raise ValueError("the grid must contain the identity exactly once")
    return ident + [a for a in g if a is not ident[0]]


def held_out_defect(backend, q, traj, m, lift, overrides, nbrs) -> float:
    """Uniform placement at the reference setting: a signal the selection never sees."""
    frames = generated_frames(q["x0"], traj)
    vals = []
    for s in starts_for("uniform", traj[None], q, m):
        w = run_window(backend, q, frames, s, 2 * m, lift, n_perturb=0, sim_overrides=overrides or None, nbrs=nbrs)
        vals.append(np.inf if w["blown"] else w["raw"])
    return float(max(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4c_repair"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("repair.jsonl")
    done = {(r["qid"], r["model"]) for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {(r["qid"], r["model"]) for r in read_jsonl(ledger)} if ledger.exists() else set()
    entries = shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards)
    root, lift, m, adj = gen_dir(cfg, args.dry_run), cfg["lift"], cfg["m"], grid(cfg)

    checked = False
    for e in entries:
        q = None
        for model in cfg["models"]:
            if (e["qid"], model) in done:
                continue
            q = q or backend.load(e)
            if not checked:
                ref = backend.generate_reference(model, q, 1, 123)[0]
                mine = backend.generate_conditioned(model, q, 123, [{"dlogE": 0.0, "force_scale": 1.0}])[0]
                diff = float(np.abs(ref - mine).max())
                print(f"[p4c] identity check ({model}, {e['qid']}): max |pipeline - conditioned| = {diff:.2e}")
                if diff > cfg["equivalence_tol"]:
                    raise RuntimeError(f"identity conditioning does not reproduce PhysCtrl's pipeline ({diff:.2e})")
                checked = True
            if (e["qid"], model) in crashed:
                append_jsonl({"qid": e["qid"], "split": e["split"], "model": model, "crashed": True}, out)
                continue
            try:
                gt = q["x"][1::2][:24]
                r0 = np.load(root / model / e["split"] / f"{e['qid']}.npz")["samples"][0]
                nbrs = nbrs_for(q, lift["k"])
                ref_ov = resolve_fidelity(cfg["anchor_fidelity"], q)
                ver_ov = resolve_fidelity(cfg["verify_fidelity"], q)
                aw = run_window(backend, q, generated_frames(q["x0"], r0), 0, 2 * m - 1, lift, n_perturb=0,
                                sim_overrides=ref_ov or None, anchor=True, nbrs=nbrs, return_frames=True)
                ks = verified_model_frames(0, 2 * m - 1)
                anchor = np.stack([aw["frames"][2 * k + 1] for k in ks], 0)
                seed = int(cfg["seed"] + zlib.crc32(e["qid"].encode()) % 100000)    # = Phase 4b's seed
                cands = backend.generate_conditioned(model, q, seed, adj)
                scores, defects, amis = [], [], []
                for c in cands:
                    d = verify(backend, q, c, m, lift, ver_ov, nbrs)
                    a = float(np.linalg.norm(c[ks] - anchor, axis=2).mean()) if not aw["blown"] else 0.0
                    defects.append(d); amis.append(a); scores.append(max(d, a))
                best = int(np.argmin(scores))                    # first minimum: identity wins ties
                r4, r4id = cands[best], cands[0]
                row = {"qid": e["qid"], "split": e["split"], "model": model, "seed": seed,
                       "chosen": adj[best], "chosen_index": best, "anchor_blown": bool(aw["blown"]),
                       "scores": scores, "defects": defects, "anchor_mismatch": amis,
                       "R4__err_t": per_frame_err(r4, gt), "R4id__err_t": per_frame_err(r4id, gt),
                       "R0__heldout_defect": held_out_defect(backend, q, r0, m, lift, ref_ov, nbrs),
                       "R4__heldout_defect": held_out_defect(backend, q, r4, m, lift, ref_ov, nbrs),
                       "R4__cert_defect": defects[best]}
            except Exception as ex:
                if cuda_context_lost(ex):
                    append_jsonl({"qid": e["qid"], "model": model, "error": f"{type(ex).__name__}: {ex}"}, ledger)
                    exit_cuda_lost(f"{e['qid']} {model}", ex)
                raise
            append_jsonl(row, out)
        print(f"[p4c] {e['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
