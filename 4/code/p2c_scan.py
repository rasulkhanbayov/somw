"""Phase 2c (A100): exhaustive window scan, the data behind every placement, window-length
and simulator-fidelity question left open by Phase 2-v2.

For each query and each trajectory source (the base and large generations' sample 0, and the
ground truth itself), at each simulator fidelity and window length m, run EVERY candidate
window (model frames k -> k+m, i.e. simulator frames 2k+1 -> 2k+1+2m) plus the frame-0 anchor
from the true initial state, and record the raw defect of each window. Restarts use the
Phase 2-v2 lift (lift_noF: F = I). No perturbations: Phase 2-v2 showed chaos normalisation
does not change ranks in these splits.

Because every window is stored, any placement strategy (uniform, random, event from the
generation, event from the physics prior, disagreement, the exhaustive maximum) is evaluated
offline by p2c_summarize.py without rerunning the simulator, and the 'gt' source gives the
lift floor at every fidelity and window length.

Falsifiers are pre-registered in p2c_summarize.py and PLAN.md ("Phase 2c"), before this runs.

Fidelity specs may set cfl_target (substep chosen per query from the wave speed), and
cfg cfl_gate skips windows whose CFL number exceeds it, recording them as blown.

Output: results/p2c_scan/shard_*/scan.jsonl, one row per (qid, source, fidelity, m) with
per-window lists; cuda_crashes.jsonl lists windows whose rollout lost the CUDA context.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, read_jsonl, resume_rows, shard_items, start_run, timed)
from backends import make_backend  # noqa: E402
from p2b_generate import all_entries, gen_dir  # noqa: E402
from verifier import (candidate_starts, cfl_number, generated_frames, gt_frames, nbrs_for,  # noqa: E402
                      resolve_fidelity, run_window)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p2c_scan"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("scan.jsonl")
    done = {(r["qid"], r["source"], r["fidelity"], r["m"]) for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {(r["qid"], r["source"], r["fidelity"], r["m"], r["start"])
               for r in read_jsonl(ledger)} if ledger.exists() else set()

    entries = shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards)
    root = gen_dir(cfg, args.dry_run)
    lift = cfg["lift"]
    sources = list(cfg["models"]) + (["gt"] if cfg.get("include_gt", True) else [])

    for e in entries:
        q = None
        for source in sources:
            keys = [(e["qid"], source, f, m) for f in cfg["fidelities"] for m in cfg["ms"]]
            if all(k in done for k in keys):
                continue
            q = q or backend.load(e)
            if source == "gt":
                frames = gt_frames(q["x"], odd_only=True)
            else:
                path = root / source / e["split"] / f"{e['qid']}.npz"
                if not path.exists():
                    print(f"[p2c_scan] missing generation {path}; run p2b_generate first")
                    continue
                frames = generated_frames(q["x0"], np.load(path)["samples"][0])
            nbrs = nbrs_for(q, lift["k"])
            for fid, spec in cfg["fidelities"].items():
                overrides = resolve_fidelity(spec, q)
                cfl = cfl_number(float(q["E"]), float(q["nu"]), float(overrides.get("substep_dt", 1e-4)),
                                 int(overrides.get("n_grid", 125)))
                # CFL gate (added Oct 11, 2026): above cfl_gate the rollout is known to blow
                # up and to corrupt the CUDA context (Phase 2c crash data). Such windows are
                # recorded as blown, which is how the +inf rule scores them, without running.
                gated = cfl > float(cfg.get("cfl_gate", float("inf")))
                for m in cfg["ms"]:
                    if (e["qid"], source, fid, m) in done:
                        continue
                    wins = [(0, 2 * m - 1, True)] + [(2 * k + 1, 2 * m, False) for k in candidate_starts(m)]
                    starts, raws, p90s, blown, subs, secs, is_crash = [], [], [], [], [], [], []
                    for s, L, anchor in wins:
                        key = (e["qid"], source, fid, m, s)
                        starts.append(s)
                        if key in crashed or gated:
                            raws.append(None); p90s.append(None); blown.append(True)
                            subs.append(None); secs.append(None); is_crash.append(True)
                            continue
                        rec = {}
                        try:
                            with timed(rec, "t"):
                                w = run_window(backend, q, frames, s, L, lift, n_perturb=0,
                                               sim_overrides=overrides or None, anchor=anchor, nbrs=nbrs)
                        except Exception as ex:
                            if cuda_context_lost(ex):
                                append_jsonl({"qid": e["qid"], "source": source, "fidelity": fid, "m": m,
                                              "start": s, "error": f"{type(ex).__name__}: {ex}"}, ledger)
                                exit_cuda_lost(f"{e['qid']} {source} {fid} m={m} s={s}", ex)
                            raise
                        raws.append(w["raw"]); p90s.append(w["raw_p90"])
                        blown.append(w["blown"]); subs.append(w["substeps"])
                        secs.append(rec["t"]); is_crash.append(False)
                    append_jsonl({"qid": e["qid"], "split": e["split"], "source": source, "fidelity": fid,
                                  "m": m, "starts": starts, "raw": raws, "raw_p90": p90s, "blown": blown,
                                  "crashed": is_crash, "substeps": subs, "seconds": secs,
                                  "cfl": cfl, "cfl_gated": bool(gated),
                                  "substep_dt": float(overrides.get("substep_dt", 1e-4))}, out)
        print(f"[p2c_scan] {e['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
