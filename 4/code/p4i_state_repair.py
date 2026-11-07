"""Phase 4i (H200 or A100): full-state restarts for the Phase 4h suffix repair.

Pre-registered Oct 13, 2026 (PLAN.md "Phase 4i"), before any 4i code ran. Phase 4h failed on risk
because (a) restarts with F = I lose the stress state and (b) late restarts begin from generated
positions that have drifted. For every 4h 'suffix' query (same chain decision j / k_j, recomputed
here and checked against the 4h row), two restarts at model frame k_j, simulated to sim frame 47
at the reference setting:
  V1  stable F-lift    lift (x, v, F, C) from the generated positions with F projected to the
                       closest proper deformation with stretches in [s_lo, s_hi] (bounds from
                       p4i_f_bounds.py, training objects only). Floor: the same from ground truth.
  V2  coarse shadow    simulate from the true initial state at coarse_cfl (the Phase 4d setting)
                       to sim frame 2k_j+1 and restart from its full particle state.
Output frames: generated model frames 0..k_j, simulated frames after k_j (as in 4h).
Blow-ups are recorded; p4i_summarize.py applies the pre-registered fallback (full reference).

Falsifier: see PLAN.md. Output: results/p4i_state/shard_*/state.jsonl (+ cuda_crashes.jsonl)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, read_jsonl, resume_rows, results_dir, shard_items, start_run, timed)
from backends import make_backend  # noqa: E402
from lift import lift_state  # noqa: E402
from p2b_generate import all_entries, gen_dir  # noqa: E402
from p4h_suffix_repair import N_MODEL, chain_decision  # noqa: E402
from verifier import blew_up, generated_frames, gt_frames, nbrs_for, resolve_fidelity, run_window  # noqa: E402


def splice(base: np.ndarray, sim: dict, kj: int) -> np.ndarray:
    out = base.copy()
    for k, x in sim.items():
        if k > kj:
            out[k] = x
    return out


def restart(backend, q, frames, kj, lift=None, state=None, nbrs=None, sim_overrides=None):
    """Reference rollout from model frame kj (sim frame 2kj+1) to sim frame 47.
    sim_overrides (Pilot 8a): run the suffix at another fidelity instead of the reference."""
    s = 2 * kj + 1
    w = run_window(backend, q, frames, s, 47 - s, lift or {}, n_perturb=0, state=state, nbrs=nbrs,
                   return_frames=True, sim_overrides=sim_overrides)
    return {(f - 1) // 2: w["frames"][f] for f in w["frames"] if f % 2 == 1}, int(w["substeps"]), bool(w["blown"])


def clamp_fraction(frames, kj, q, lift, nbrs, bounds):
    """Share of singular values of the least-squares F that the V1 projection changes (secondary).
    F does not depend on the frame time, so any frame_dt gives the same F."""
    st = lift_state(frames, 2 * kj + 1, 1.0, rest=np.asarray(q["x0"], dtype=np.float64),
                    k=lift["k"], lam=lift["lam"], use_F=True, use_C=False, nbrs=nbrs)
    S = np.linalg.svd(st["F"].reshape(-1, 3, 3).astype(np.float64), compute_uv=False)
    return float(np.mean((S < bounds[0]) | (S > bounds[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4i_state"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("state.jsonl")
    done = {r["qid"] for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {r["qid"] for r in read_jsonl(ledger)} if ledger.exists() else set()

    b = json.loads((results_dir(cfg["bounds_run"], args.dry_run) / "bounds.json").read_text())
    bounds = (float(b["s_lo"]), float(b["s_hi"]))
    lift1 = {**cfg["lift_v1"], "F_clamp": bounds}
    spec, tau, chain_k = cfg["window_spec"], float(cfg["tau"]), list(cfg["chain_k"])
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"], args.dry_run).glob("shard_*/scan.jsonl"))
            for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == spec["fidelity"] and int(r["m"]) == spec["m"]}
    h4 = {r["qid"]: r for f in sorted(results_dir(cfg["h4_run"], args.dry_run).glob("shard_*/suffix.jsonl"))
          for r in read_jsonl(f) if not r.get("crashed")}
    root = gen_dir(cfg, args.dry_run)
    for e in shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards):
        r4 = h4.get(e["qid"])
        if e["qid"] in done or r4 is None or r4["mode"] != "suffix":
            continue
        if e["qid"] in crashed:
            append_jsonl({"qid": e["qid"], "split": e["split"], "crashed": True}, out)
            continue
        j, kj, _ = chain_decision(scan[e["qid"]], chain_k, tau)
        if (j, kj) != (r4["j"], r4["k_j"]):
            raise RuntimeError(f"{e['qid']}: chain decision {(j, kj)} differs from the 4h row {(r4['j'], r4['k_j'])}")
        q = backend.load(e)
        gt = q["x"][1::2][:N_MODEL]
        g0 = np.load(root / "base" / e["split"] / f"{e['qid']}.npz")["samples"][0]
        gen_f, gt_f = generated_frames(q["x0"], g0), gt_frames(q["x"], odd_only=True)
        nbrs = nbrs_for(q, lift1["k"])
        row = {"qid": e["qid"], "split": e["split"], "j": j, "k_j": kj, "bounds": list(bounds)}
        try:
            # V1: stable F-lift from the generation, and its floor from ground truth
            sim, subs, blown = restart(backend, q, gen_f, kj, lift=lift1, nbrs=nbrs)
            fsim, _, fblown = restart(backend, q, gt_f, kj, lift=lift1, nbrs=nbrs)
            row.update(v1_err_t=np.linalg.norm(splice(g0, sim, kj) - gt, axis=2).mean(axis=1).tolist(),
                       v1_substeps=subs, v1_blown=blown,
                       v1_floor_err_t=np.linalg.norm(splice(gt, fsim, kj) - gt, axis=2).mean(axis=1).tolist(),
                       v1_floor_blown=fblown, v1_clamped_frac=clamp_fraction(gen_f, kj, q, lift1, nbrs, bounds))
            # V2: coarse shadow from the true initial state, then the reference suffix
            sh = backend.make_sim(q, overrides=resolve_fidelity(cfg["shadow_fidelity"], q) or None)
            rec = {}
            with timed(rec, "t"):
                sh_traj = sh.step_frames(2 * kj + 1)
            sh_state = {k: np.asarray(v) for k, v in sh.export_state().items()}
            sh_blown = blew_up(sh_traj[-1], sh)
            row.update(v2_shadow_substeps=int(sh.substeps_done), v2_shadow_seconds=rec["t"],
                       v2_shadow_blown=bool(sh_blown),
                       v2_shadow_err_kj=float(np.linalg.norm(sh_traj[-1] - gt[kj], axis=1).mean()),
                       gen_err_kj=float(np.linalg.norm(g0[kj] - gt[kj], axis=1).mean()))
            if not sh_blown:
                sim2, subs2, blown2 = restart(backend, q, gen_f, kj, state=sh_state)
                row.update(v2_err_t=np.linalg.norm(splice(g0, sim2, kj) - gt, axis=2).mean(axis=1).tolist(),
                           v2_substeps=subs2, v2_blown=blown2)
            else:
                row.update(v2_err_t=None, v2_substeps=0, v2_blown=True)
        except Exception as ex:
            if cuda_context_lost(ex):
                append_jsonl({"qid": e["qid"], "error": f"{type(ex).__name__}: {ex}"}, ledger)
                exit_cuda_lost(e["qid"], ex)
            raise
        append_jsonl(row, out)
        print(f"[p4i] {e['qid']} k_j={kj} v1 {np.mean(row['v1_err_t']):.4f} "
              f"v2 {np.mean(row['v2_err_t']) if row['v2_err_t'] else float('nan'):.4f}")
    run.finish("ok")


if __name__ == "__main__":
    main()
