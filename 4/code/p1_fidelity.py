"""Phase 1b: how cheap can the simulator get before it is as wrong as the generator?

A reviewer's first objection to "simulation is expensive" is that PhysCtrl's
simulator settings (substep 1e-4 s, 125^3 grid) may be overkill. This sweep
runs the same elastic test queries at coarser substeps and grids, and records
error against the stored ground truth (which was generated at the default
settings), cost, and stability. It also records the generators' error on the
same queries, so p1_summarize.py can find the cheapest simulator whose error
is no worse than the generator's: the matched-accuracy comparison.

Errors use only the frames the generator outputs (simulator frames 1,3,...,47).
Note: PhysCtrl runs int(0.0417 / substep_dt) substeps per frame, so coarse settings
also shorten the frame slightly (26 x 1.6e-3 = 0.0416 s). That timing drift is part
of what a coarse simulator gets wrong and is counted in its error, as it would be in
any pipeline using that setting.
Sharded over queries for Slurm arrays.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, resume_rows, shard_items, start_run, timed)
from backends import make_backend, testset_manifest_path, N_SIM_FRAMES  # noqa: E402


def row_key(r: dict) -> tuple:
    if r["kind"] == "sim":
        return ("sim", r["qid"], r["substep_dt"], r.get("n_grid"))
    return ("gen", r["qid"], r["model"])


def traj_error(pred_odd: np.ndarray, gt_odd: np.ndarray) -> tuple[float, float]:
    l2 = np.linalg.norm(pred_odd - gt_odd, axis=2).mean(axis=1)
    return float(l2.mean()), float(l2[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--tag", default="A100")
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(f"p1_fidelity_{args.tag}", cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("fidelity.jsonl")
    done = {row_key(r) for r in resume_rows(out, args.resume)}

    entries = backend.manifest(testset_manifest_path(args.dry_run), split="test_id")[: cfg["n_queries"]]
    entries = shard_items(entries, shard, args.num_shards)
    sweep = cfg["sweep"]
    configs = [{"substep_dt": dt, "n_grid": ng} for dt in sweep["substep_dt"] for ng in sweep["n_grid"]]
    if args.dry_run:
        configs = [{"substep_dt": dt} for dt in (5e-4, 1e-3, 2e-3)]

    from physctrl_bridge import is_unstable

    for e in entries:
        q = backend.load(e)
        gt_odd = q["x"][1::2][:24]
        for c in configs:
            row = {"kind": "sim", "qid": q["qid"], **c, "mode": cfg["mode"]}
            if row_key(row) in done:
                continue
            try:
                rec = {}
                sim = backend.make_sim(q, overrides=c, timing_mode=cfg["mode"])
                sim.step_frames(1, record=False)                      # JIT / capture warm-up
                sim = backend.make_sim(q, overrides=c, timing_mode=cfg["mode"])
                with timed(rec, "rollout_s"):
                    traj = sim.step_frames(N_SIM_FRAMES)
                row["rollout_s"] = rec["rollout_s"]
                row["substeps"] = sim.substeps_done
                row["stable"] = not is_unstable(traj)
                if row["stable"]:
                    row["err_mean"], row["err_final"] = traj_error(traj[0::2][:24], gt_odd)
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {ex}"
                if cuda_context_lost(ex):
                    # A blow-up at this setting; counts as unstable in p1_summarize.
                    row.update(stable=False, crashed=True)
                    append_jsonl(row, out)
                    exit_cuda_lost(f"{q['qid']} {c}", ex)
            append_jsonl(row, out)

        for which in cfg["generator_models"]:
            if ("gen", q["qid"], which) in done:
                continue
            rec = {}
            try:
                with timed(rec, "t"):
                    gen = backend.generate(which, [q], n_samples=cfg["generator_samples"], seed=cfg["seed"])[0]
            except Exception as ex:
                if cuda_context_lost(ex):
                    exit_cuda_lost(f"{q['qid']} generate {which}", ex)
                raise
            errs = [traj_error(g, gt_odd) for g in gen]
            append_jsonl({"kind": "gen", "qid": q["qid"], "model": which,
                          "n_samples": len(gen), "seconds_total": rec["t"],
                          "err_mean": float(np.mean([a for a, _ in errs])),
                          "err_final": float(np.mean([b for _, b in errs])),
                          "err_mean_per_sample": [a for a, _ in errs]}, out)
        print(f"[p1_fidelity] {q['qid']} done")

    run.finish("ok")


if __name__ == "__main__":
    main()
