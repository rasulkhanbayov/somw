"""Phase 2b (part 2, A100): does the window defect track true trajectory error?

For each query and model, sample 0 is the trajectory under test. We record:
  true error   err_mean: mean over the 24 frames of the mean per-point L2 to
               ground truth; err_final: the same at the last frame.
  baselines    seed_spread: mean distance of the K samples to their mean
               (needs K-1 extra generations, no simulator); seed_spread_final;
               motion: mean final displacement (a "big motions are wrong" control).
  probes       for each placement strategy in the config: windows of m model
               frames, n_windows per trajectory, raw and chaos-normalised
               defects aggregated by max and mean, plus simulator substeps used.
               'oracle' placement (knows the true error) is a ceiling, not a method.

Cost is stored as substeps / substeps of one full rollout, so
p2_summarize.py can report signal quality against simulator budget.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, read_jsonl, shard_items, start_run)
from backends import make_backend  # noqa: E402
from p2b_generate import all_entries, gen_dir  # noqa: E402
from verifier import aggregate, generated_frames, nbrs_for, run_window, select_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p2b_probe"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("probes.jsonl")
    done = set()
    if out.exists():
        done = {(r["qid"], r["model"]) for r in read_jsonl(out)}
    # Strategies whose windows lost the CUDA context. They are not rerun after the restart
    # (a deterministic blow-up would just crash again); the row records them as crashed.
    ledger = run.path("cuda_crashes.jsonl")
    crashed = {(r["qid"], r["model"], r["strategy"]) for r in read_jsonl(ledger)} if ledger.exists() else set()
    entries = shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards)
    root = gen_dir(cfg, args.dry_run)
    pc = cfg["probe"]

    for e in entries:
        q = None
        for which in cfg["models"]:
            if (e["qid"], which) in done:
                continue
            path = root / which / e["split"] / f"{e['qid']}.npz"
            if not path.exists():
                print(f"[p2b_probe] missing generation {path}; run p2b_generate first")
                continue
            q = q or backend.load(e)
            samples = np.load(path)["samples"]
            gt = q["x"][1::2][:24]
            g0 = samples[0]
            l2 = np.linalg.norm(g0 - gt, axis=2).mean(axis=1)
            spread = np.linalg.norm(samples - samples.mean(0, keepdims=True), axis=3).mean(axis=2)  # (K,24)
            row = {
                "qid": q["qid"], "split": q["split"], "model": which,
                "err_mean": float(l2.mean()), "err_final": float(l2[-1]),
                "seed_spread": float(spread.mean()), "seed_spread_final": float(spread[:, -1].mean()),
                "motion": float(np.linalg.norm(g0[-1] - q["x0"], axis=1).mean()),
                "n_samples": int(samples.shape[0]),
            }
            frames = generated_frames(q["x0"], g0)
            nbrs = nbrs_for(q, pc["lift"]["k"])
            sim_probe = backend.make_sim(q)
            full_substeps = sim_probe.substeps_per_frame * sim_probe.frame_num
            for strat in pc["strategies"]:
                if (q["qid"], which, strat) in crashed:
                    row[f"{strat}__error"] = "crashed: CUDA context lost (see cuda_crashes.jsonl)"
                    row[f"{strat}__crashed"] = True
                    continue
                rng = np.random.default_rng(pc["seed"] + zlib.crc32(q["qid"].encode()))
                try:
                    wins = select_windows(strat, samples, q, pc["n_windows"], pc["m"], rng, gt=gt)
                    res = [run_window(backend, q, frames, s, L, pc["lift"], n_perturb=pc["n_perturb"],
                                      perturb_cfg=pc["perturb"], anchor=a, seed=pc["seed"] + i, nbrs=nbrs)
                           for i, (s, L, a) in enumerate(wins)]
                    sub = sum(r["substeps"] for r in res)
                    sub_raw_only = sum(r["length"] for r in res) * sim_probe.substeps_per_frame
                    row[f"{strat}__raw_max"] = aggregate(res, "raw", "max")
                    row[f"{strat}__raw_mean"] = aggregate(res, "raw", "mean")
                    row[f"{strat}__norm_max"] = aggregate(res, "norm", "max")
                    row[f"{strat}__norm_mean"] = aggregate(res, "norm", "mean")
                    row[f"{strat}__cost_frac"] = sub / full_substeps
                    row[f"{strat}__cost_frac_raw_only"] = sub_raw_only / full_substeps
                    row[f"{strat}__windows"] = [(int(s), int(L), bool(a)) for s, L, a in wins]
                    row[f"{strat}__blown"] = sum(r["blown"] for r in res)
                    row[f"{strat}__blown_perturb"] = sum(r["blown_perturb"] for r in res)
                except Exception as ex:
                    if cuda_context_lost(ex):
                        append_jsonl({"qid": q["qid"], "model": which, "strategy": strat,
                                      "error": f"{type(ex).__name__}: {ex}"}, ledger)
                        exit_cuda_lost(f"{q['qid']} {which} {strat}", ex)
                    row[f"{strat}__error"] = f"{type(ex).__name__}: {ex}"
            append_jsonl(row, out)
        print(f"[p2b_probe] {e['qid']} done")
    run.finish("ok")


if __name__ == "__main__":
    main()
