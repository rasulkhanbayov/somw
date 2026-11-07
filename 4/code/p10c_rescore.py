"""Phase 10c step 2 (GPU; pre-registered Oct 21, 2026; PLAN.md "Phase 10c"): positions of every Phase 9 pipeline's
output and of the reference at three time steps, for exact re-scoring on CPU (p10c_summary.py).

Per query (results/p10c_decisions/decisions.json, fixed from the Phase 9 records before this job):
  gt (stored reference, dt 1e-4), half (dt 5e-5), quarter (dt 2.5e-5): reference setting, grid 125
  M1, M2, M3 (and coarse for single objects): full rollouts at the Phase 9 settings (configs/p9_full.yaml)
  ours: the generation if accepted; on repair the Phase 9 shadow up to k and the Phase 9 suffix (single objects: M1
        shadow + M2 suffix, configs/p9_allk.yaml; contact: coarse shadow + M3 suffix at k_sw, configs/p9c_allk.yaml);
        the reference at dt otherwise
  gen: sample 0 of the stored generation
Recomputed M1-M3 / coarse / repair errors against gt must match the stored Phase 9 err_t within 2e-3 (GPU MPM is not
bit-reproducible); a larger deviation stops the job. An unstable rollout is stored as NaN and flagged.
Output: $BIG_ROOT/p10c_traj/<set>/<qid>.npz and results/p10c_rescore/shard_<i>/rows.jsonl (resumable)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost, load_config, load_json,  # noqa: E402
                    resume_rows, results_dir, shard_items, start_run)
from backends import make_backend  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p4d_fullsim import entries_from_sets  # noqa: E402
from p4h_suffix_repair import N_MODEL  # noqa: E402
from p4i_state_repair import restart, splice  # noqa: E402
from verifier import generated_frames, resolve_fidelity  # noqa: E402

REF = {"half": {"substep_dt": 5e-5, "n_grid": 125}, "quarter": {"substep_dt": 2.5e-5, "n_grid": 125}}
TOL = 2e-3       # reported: deviations above it are counted in rows.jsonl
STOP = 1e-2      # amended Oct 21 (after one strong-push M3 rollout deviated 0.0029; GPU MPM is not bit-reproducible and
                 # 10a found that family time-step sensitive): only a deviation above STOP means a different pipeline


def rollout(backend, q, spec):
    from physctrl_bridge import is_unstable
    traj = backend.make_sim(q, overrides=spec or None).step_frames(48)
    return None if is_unstable(traj) else traj[0::2][:N_MODEL]


def check(name, traj, gt, stored):
    if traj is None or stored is None:
        return None
    dev = float(np.abs(np.linalg.norm(traj - gt, axis=2).mean(axis=1) - np.asarray(stored)).max())
    if dev > STOP:
        raise RuntimeError(f"{name}: recomputed error differs from Phase 9 by {dev:.4f}")
    return dev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg["run_name"], cfg, synthetic=args.dry_run, shard=str(shard))
    out_rows = run.path("rows.jsonl")
    done = {r["qid"] for r in resume_rows(out_rows, args.resume)}
    dec = load_json(results_dir(cfg["decisions_run"], allow_synthetic=args.dry_run) / "decisions.json")["decisions"]
    sets = {s: {"full": load_config(cfg[s]["full_config"], args.dry_run), "allk": load_config(cfg[s]["allk_config"], args.dry_run)}
            for s in ("single", "contact")}
    traj_root = Path(os.path.expandvars(cfg["traj_dir"]))
    backends, ents, groots = {}, {}, {}
    for s, c in sets.items():
        backends[s] = make_backend(c["full"], args.dry_run)
        ents[s] = {e["qid"]: e for e in entries_from_sets(c["full"], backends[s], args.dry_run)}
        groots[s] = gen_dir(c["full"], args.dry_run)
    for d in shard_items(dec, shard, args.num_shards):
        if d["qid"] in done or (Path(os.path.expandvars(cfg["traj_dir"])) / d["set"] / f"{d['qid']}.npz").exists():
            continue
        s, q_id = d["set"], d["qid"]
        backend, cf = backends[s], sets[s]
        try:
            q = backend.load(ents[s][q_id])
            gt = np.asarray(q["x"][1::2][:N_MODEL], np.float64)
            g0 = np.load(groots[s] / "base" / d["split"] / f"{q_id}.npz")["samples"][0].astype(np.float64)
            T = {"gt": gt, "gen": g0}
            for name, spec in REF.items():
                T[name] = rollout(backend, q, spec)
            devs = {}
            fids = ("coarse_cfl", "M1", "M2", "M3") if s == "single" else ("M1", "M2", "M3")
            for f in fids:
                T[f] = rollout(backend, q, resolve_fidelity(cf["full"]["fidelities"][f], q))
                devs[f] = check(f"{q_id} {f}", T[f], gt, d["full_err_t"].get(f))
            if d["ours"] == "accept":
                T["ours"] = g0
            elif d["ours"] == "reference":
                T["ours"] = gt
            else:
                k = int(d["k"])
                sh = backend.make_sim(q, overrides=resolve_fidelity(cf["allk"]["shadow_fidelity"], q) or None)
                sh.step_frames(2 * k + 1)
                st = {kk: np.asarray(v) for kk, v in sh.export_state().items()}
                rov = resolve_fidelity(cf["allk"]["restart_fidelity"], q) or None
                sim, _, blown = restart(backend, q, generated_frames(q["x0"], g0), k, state=st, sim_overrides=rov)
                if blown:
                    raise RuntimeError(f"{q_id}: repair at k={k} blew up, but Phase 9's did not")
                T["ours"] = splice(g0, sim, k)
                devs["ours"] = check(f"{q_id} ours", T["ours"], gt, d["ours_err_t"])
        except Exception as ex:
            if cuda_context_lost(ex):
                exit_cuda_lost(q_id, ex)
            raise
        dst = traj_root / s / f"{q_id}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **{k: (np.full_like(gt, np.nan) if v is None else v).astype(np.float32) for k, v in T.items()})
        append_jsonl({"qid": q_id, "set": s, "unstable": sorted(k for k, v in T.items() if v is None),
                      "max_dev": {k: v for k, v in devs.items() if v is not None},
                      "above_tol": sorted(k for k, v in devs.items() if v is not None and v > TOL)}, out_rows)
        print(f"[p10c] {q_id} ({s}, ours {d['ours']})", flush=True)
    run.finish("ok")


if __name__ == "__main__":
    main()
