"""Pilot 8d (GPU): certify the cheap simulator M1 itself (EXPLORATORY, development data).

Defined Oct 18, 2026 in PLAN.md "Pilot 8d" before any 8d code ran. Per query:
  proposal  M1 rollout (cfg 'proposal_fidelity') from the true initial state to sim frame 48; full state exported
            at model frames k in cfg 'check_k'
  windows   from M1's own full state at each k: cfg 'window_m' model frames at cfg 'window_fidelity';
            defect d_k = mean distance between the window's last frame and M1's frame k + m (sim 2(k+m)+1)
  suffixes  (skipped for splits matching cfg 'windows_only_prefix') reference rollout from M1's state at k to the
            end, spliced after M1's frames 0..k
Records per-frame errors against the stored ground truth and substeps for every part; p8d_pilot.py builds the
CM1 pipeline (novelty threshold from the windows-only calibration queries) on the CPU.

Output: results/p8d_certsim/shard_*/certsim.jsonl (+ cuda_crashes.jsonl)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_args, append_jsonl, cuda_context_lost, exit_cuda_lost,  # noqa: E402
                    load_config, read_jsonl, resume_rows, shard_items, start_run)
from backends import make_backend  # noqa: E402
from p4d_fullsim import entries_from_sets  # noqa: E402
from p4h_suffix_repair import N_MODEL  # noqa: E402
from p4i_state_repair import restart, splice  # noqa: E402
from verifier import blew_up, resolve_fidelity  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p8d_certsim"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("certsim.jsonl")
    done = {r["qid"] for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {r["qid"] for r in read_jsonl(ledger)} if ledger.exists() else set()
    K, m = sorted(int(k) for k in cfg["check_k"]), int(cfg["window_m"])
    for e in shard_items(entries_from_sets(cfg, backend, args.dry_run), shard, args.num_shards):
        if e["qid"] in done:
            continue
        if e["qid"] in crashed:
            append_jsonl({"qid": e["qid"], "split": e["split"], "crashed": True}, out)
            continue
        q = backend.load(e)
        gt = q["x"][1::2][:N_MODEL]
        win_only = e["split"].startswith(cfg.get("windows_only_prefix", "\0"))
        row = {"qid": e["qid"], "split": e["split"], "kind": e.get("kind"), "windows_only": win_only, "k": {}}
        try:
            sh = backend.make_sim(q, overrides=resolve_fidelity(cfg["proposal_fidelity"], q) or None)
            pos, states, frame, blown = {0: np.asarray(q["x0"])}, {}, 0, False
            for k in K:
                traj = sh.step_frames(2 * k + 1 - frame)
                pos.update({frame + 1 + i: p for i, p in enumerate(traj)})
                frame = 2 * k + 1
                blown = blown or blew_up(traj[-1], sh)
                states[k] = {kk: np.asarray(v) for kk, v in sh.export_state().items()}
            rest = sh.step_frames(48 - frame)
            pos.update({frame + 1 + i: p for i, p in enumerate(rest)})
            blown = blown or blew_up(rest[-1], sh)
            m1 = np.stack([pos[f] for f in range(1, 48, 2)])
            row.update(m1_err_t=np.linalg.norm(m1 - gt, axis=2).mean(axis=1).tolist(), m1_substeps=int(sh.substeps_done),
                       m1_blown=bool(blown))
            frames = {0: np.asarray(q["x0"]), **{2 * j + 1: m1[j] for j in range(N_MODEL)}}
            for k in K:
                st = states[k]
                r = {}
                w = backend.make_sim(q, overrides=resolve_fidelity(cfg["window_fidelity"], q) or None)
                w.set_state(st["x"], st.get("v"), st.get("F"), st.get("C"), time=(2 * k + 1) * w.frame_dt)
                wt = w.step_frames(2 * m)
                wb = bool(blew_up(wt[-1], w))
                tgt = pos[2 * (k + m) + 1]
                r.update(d=None if wb else float(np.linalg.norm(np.asarray(wt[-1]) - tgt, axis=1).mean()),
                         win_substeps=int(w.substeps_done), win_blown=wb)
                if not win_only:
                    sim, subs, sb = restart(backend, q, frames, k, state=st)
                    r.update(err_t=None if sb else np.linalg.norm(splice(m1, sim, k) - gt, axis=2).mean(axis=1).tolist(),
                             suffix_substeps=int(subs), blown=bool(sb))
                row["k"][str(k)] = r
        except Exception as ex:
            if cuda_context_lost(ex):
                append_jsonl({"qid": e["qid"], "error": f"{type(ex).__name__}: {ex}"}, ledger)
                exit_cuda_lost(e["qid"], ex)
            raise
        append_jsonl(row, out)
        print(f"[p8d] {e['qid']} M1 {np.mean(row['m1_err_t']):.4f} " +
              " ".join(f"d{k}:{v['d'] if v['d'] is None else round(v['d'], 4)}" for k, v in row["k"].items()), flush=True)
    run.finish("ok")


if __name__ == "__main__":
    main()
