"""Phase 4j (H200 or A100): V2 suffix repair at every chain restart point, on the calibration pool.

Pre-registered Oct 13, 2026 (PLAN.md "Phase 4j"), before any 4j code ran. For every base
calibration query (id_new_0..11): one coarse_cfl shadow from the true initial state to sim frame
2*max(k)+1, exporting its full particle state at sim frames 2k+1 for k in chain_k; then, for each
k, a reference rollout from that state to sim frame 47. Output frames: generated model frames
0..k (sample 0), simulated after k. Which k a pipeline uses is decided later, on CPU, in
p4j_confirm.py, so the held-out test and the shuffled-restart ablation share these rollouts.
Phase 4k (Oct 14, 2026): with shadow_to_end the shadow continues to sim frame 48 and its error
is recorded as the coarse-only baseline (coarse_err_t); off by default, so 4j is unchanged.
Phase 4m (Oct 14, 2026): save_coarse_dir also stores the coarse model frames (float32); an empty
chain_k runs the coarse rollout only.
Phase 4l (Oct 15, 2026): event_restart adds a restart at k_e = max(k_g - 1, 1), where k_g is the first
model frame at which the two generated objects come within D_C (two-object scenes only).
Phase 4n (Oct 16, 2026): fidelity_restart_dist adds a restart at k_s = max(k_f - 1, 1), k_f the first
frame the generated objects come within that distance (3 coarse cells: where the coarse grid couples them).
Pilot 8a (Oct 17, 2026): restart_fidelity runs the suffix restarts at that fidelity (default: reference).
Phase 7b (Oct 17, 2026): save_traj_dir keeps the spliced restart outputs and sample 0 (float32);
physctrl_metrics scores them and the ground truth with PhysCtrl's DeformLoss (row key "deform").
Phase 4o (Oct 16, 2026): shadow_switch_dist records the shadow state at every model frame and restarts at
k_sw = max(k_sh - 1, 1), k_sh the first frame the SHADOW's objects come within that distance.

Output: results/p4j_allk/shard_*/allk.jsonl (+ cuda_crashes.jsonl)
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
from p2b_generate import gen_dir  # noqa: E402
from p4d_fullsim import entries_from_sets  # noqa: E402
from p4h_suffix_repair import N_MODEL  # noqa: E402
from p4i_state_repair import restart, splice  # noqa: E402
from verifier import blew_up, generated_frames, resolve_fidelity  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("run_name", "p4j_allk"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    out = run.path("allk.jsonl")
    done = {r["qid"] for r in resume_rows(out, args.resume)}
    ledger = run.path("cuda_crashes.jsonl")
    if not args.resume and ledger.exists():
        ledger.unlink()
    crashed = {r["qid"] for r in read_jsonl(ledger)} if ledger.exists() else set()
    chain_k = sorted(int(k) for k in cfg["chain_k"])
    root = gen_dir(cfg, args.dry_run)
    for e in shard_items(entries_from_sets(cfg, backend, args.dry_run), shard, args.num_shards):
        if e["qid"] in done:
            continue
        if e["qid"] in crashed:
            append_jsonl({"qid": e["qid"], "split": e["split"], "crashed": True}, out)
            continue
        q = backend.load(e)
        gt = q["x"][1::2][:N_MODEL]
        g0 = np.load(root / "base" / e["split"] / f"{e['qid']}.npz")["samples"][0]
        gen_f = generated_frames(q["x0"], g0)
        row = {"qid": e["qid"], "split": e["split"], "chain_k": chain_k,
               "gen_err_t": np.linalg.norm(g0 - gt, axis=2).mean(axis=1).tolist(), "k": {}}
        ks = list(chain_k)
        if cfg.get("event_restart"):
            # Phase 4l: one more restart just before the generated objects first approach,
            # k_e = max(k_g - 1, 1); none when they never approach or k_g <= 1.
            from mo_scenes import gen_contact_frame
            kg = gen_contact_frame(g0, q["object_id"]) if q.get("object_id") is not None else None
            k_e = max(kg - 1, 1) if (kg is not None and kg > 1) else None
            row.update(k_g=kg, k_e=k_e)
            ks = sorted(set(ks + ([k_e] if k_e is not None else [])))
        if cfg.get("fidelity_restart_dist"):
            # Phase 4n: restart just before the cheap simulator's grid couples the objects: k_f is the
            # first frame the generated objects come within d_s (3 coarse cells); k_s = max(k_f - 1, 1).
            from mo_scenes import gen_contact_frame
            kf = gen_contact_frame(g0, q["object_id"], d_c=float(cfg["fidelity_restart_dist"])) \
                if q.get("object_id") is not None else None
            k_s = max(kf - 1, 1) if (kf is not None and kf > 1) else None
            row.update(k_f=kf, k_s=k_s)
            ks = sorted(set(ks + ([k_s] if k_s is not None else [])))
        try:
            sh = backend.make_sim(q, overrides=resolve_fidelity(cfg["shadow_fidelity"], q) or None)
            states, frame, sh_blown, shadow_pos = {}, 0, False, {}
            rec = {}
            # Phase 4o: with shadow_switch_dist the shadow keeps its state at EVERY model frame 1..22, so the
            # restart can be placed where the shadow itself says the coarse grid starts coupling the objects.
            pass_ks = sorted(set(ks) | set(range(1, N_MODEL - 1))) if cfg.get("shadow_switch_dist") else ks
            with timed(rec, "t"):
                for k in pass_ks:                       # one shadow pass, state exported at each restart
                    traj = sh.step_frames(2 * k + 1 - frame)
                    shadow_pos.update({frame + 1 + i: p for i, p in enumerate(traj)})
                    frame = 2 * k + 1
                    sh_blown = sh_blown or blew_up(traj[-1], sh)
                    states[k] = ({kk: np.asarray(v) for kk, v in sh.export_state().items()},
                                 int(sh.substeps_done), bool(sh_blown),
                                 float(np.linalg.norm(traj[-1] - gt[k], axis=1).mean()))
            row["shadow_seconds"] = rec["t"]
            if cfg.get("shadow_switch_dist"):
                from mo_scenes import gen_contact_frame
                sh_traj = np.stack([shadow_pos[2 * k + 1] for k in range(0, max(pass_ks) + 1)])
                ksh = gen_contact_frame(sh_traj, q["object_id"], d_c=float(cfg["shadow_switch_dist"]))
                k_sw = max(ksh - 1, 1) if (ksh is not None and ksh > 1) else None
                row.update(k_sh=ksh, k_sw=k_sw)
                ks = sorted(set(ks + ([k_sw] if k_sw is not None else [])))
            if cfg.get("shadow_to_end"):
                # Phase 4k: continue the same shadow to sim frame 48. That is exactly the Phase 4d
                # coarse-only rollout; its model frames are sim frames 1, 3, ..., 47.
                rest = sh.step_frames(48 - frame)
                shadow_pos.update({frame + 1 + i: p for i, p in enumerate(rest)})
                coarse = np.stack([shadow_pos[f] for f in range(1, 48, 2)])
                row.update(coarse_err_t=np.linalg.norm(coarse - gt, axis=2).mean(axis=1).tolist(),
                           coarse_substeps=int(sh.substeps_done), coarse_blown=bool(blew_up(rest[-1], sh)))
                if cfg.get("save_coarse_dir"):
                    # Phase 4m: keep the coarse model frames for the cross-fidelity agreement baseline
                    cdir = Path(cfg["save_coarse_dir"]) / e["split"]
                    cdir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cdir / f"{e['qid']}.npz", coarse=coarse.astype(np.float32))
            trajs = {}
            # Pilot 8a: suffix at a medium fidelity instead of the reference (None = reference, as before)
            rov = (resolve_fidelity(cfg["restart_fidelity"], q) or None) if cfg.get("restart_fidelity") else None
            for k in ks:
                st, sh_subs, blown_k, sh_err = states[k]
                r = {"shadow_substeps": sh_subs, "shadow_blown": blown_k, "shadow_err_k": sh_err,
                     "gen_err_k": float(np.linalg.norm(g0[k] - gt[k], axis=1).mean())}
                if blown_k:
                    r.update(err_t=None, substeps=0, blown=True)
                else:
                    sim, subs, blown = restart(backend, q, gen_f, k, state=st, sim_overrides=rov)
                    sp_k = splice(g0, sim, k)
                    r.update(err_t=np.linalg.norm(sp_k - gt, axis=2).mean(axis=1).tolist(),
                             substeps=subs, blown=blown)
                    if not blown:
                        trajs[f"k{k}"] = sp_k
                row["k"][str(k)] = r
            if cfg.get("save_traj_dir") or cfg.get("physctrl_metrics"):
                # Phase 7b: keep the spliced outputs (positions were never stored before) and score
                # them, sample 0 and the ground truth with PhysCtrl's own DeformLoss (GPU).
                trajs.update(gen=g0, gt=gt)
                if cfg.get("physctrl_metrics"):
                    from physctrl_bridge import physctrl_deform_loss
                    names = sorted(trajs)
                    vals = physctrl_deform_loss([trajs[nm] for nm in names], q, clamp=[nm != "gt" for nm in names])
                    row["deform"] = dict(zip(names, vals))
                if cfg.get("save_traj_dir"):
                    tdir = Path(cfg["save_traj_dir"]) / e["split"]
                    tdir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(tdir / f"{e['qid']}.npz", **{nm: np.asarray(v, np.float32)
                                                                     for nm, v in trajs.items() if nm != "gt"})
        except Exception as ex:
            if cuda_context_lost(ex):
                append_jsonl({"qid": e["qid"], "error": f"{type(ex).__name__}: {ex}"}, ledger)
                exit_cuda_lost(e["qid"], ex)
            raise
        append_jsonl(row, out)
        print(f"[p4j] {e['qid']} coarse {np.mean(row.get('coarse_err_t', [np.nan])):.4f} " + " ".join(
            f"k{k}:{np.mean(v['err_t']):.4f}" if v["err_t"] else f"k{k}:blown" for k, v in row["k"].items()))
    run.finish("ok")


if __name__ == "__main__":
    main()
