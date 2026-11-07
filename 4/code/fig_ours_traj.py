"""Positions of OUR output trajectory for the deck's qualitative figures (Oct 16, 2026). GPU (small).

The verdict runs (p4m_allk, p4o_allk) stored only per-frame errors of our repairs, not positions. For each
case in results/fig_qualitative/deck_selection.json (picked by the median-error rule written there before
any picture), this recomputes the trajectory our pipeline produced, with the decision it made:
  repair  generated frames 0..k + reference rollout from the coarse shadow's full state at model frame k
  exact   our output is the reference rollout = the ground truth stored in the scene file
It loads each case through its pool's own config (same entry list, shadow setting and generations as the
verdict job) and checks the recomputed per-frame error against the stored err_t; a mismatch raises.
Saves results/fig_qualitative/ours_<name>.npz: ours, gen (sample 0), gt, coarse (4m only), object_id.

Paper figures (Oct 19, 2026): --selection results/fig_qualitative/paper_selection.json --prefix paper_ runs the same
recomputation on the Phase 9 cases (pools p9 / p9c), with the pool's suffix setting (restart_fidelity: M2 single
objects, M3 contact), an 'accept' decision (ours = the generation), and for single objects a full rollout at the tuned
setting M1 ('m1', checked against the stored p9_full err_t), the cheap baseline the paper compares against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir  # noqa: E402
from backends import make_backend  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p4d_fullsim import entries_from_sets  # noqa: E402
from p4h_suffix_repair import N_MODEL  # noqa: E402
from p4i_state_repair import restart, splice  # noqa: E402
from verifier import generated_frames, resolve_fidelity  # noqa: E402

POOL_CFG = {"4m": "configs/p4m_allk.yaml", "4o": "configs/p4o_allk.yaml", "p9": "configs/p9_allk.yaml", "p9c": "configs/p9c_allk.yaml"}
LADDER = {"p9": ("configs/p9_full.yaml", "M1")}          # pool -> (full-rollout config, fidelity) drawn as a baseline


def check(name, got, ref):
    dev = float(np.abs(np.asarray(got) - np.asarray(ref)).max())
    print(f"[fig_ours] {name}: max |err_t - stored| = {dev:.2e}", flush=True)
    if dev > 2e-3:              # MPM on GPU is not bit-reproducible; larger means a different pipeline
        raise RuntimeError(f"{name}: recomputed error differs from the verdict run by {dev:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", default="results/fig_qualitative/deck_selection.json")
    ap.add_argument("--prefix", default="ours_")
    args = ap.parse_args()
    out = Path("results/fig_qualitative")
    sel = json.loads(Path(args.selection).read_text())
    for pool in sorted({c["pool"] for c in sel["cases"]}):
        cfg = load_config(POOL_CFG[pool])
        backend = make_backend(cfg, False)
        ents = {e["qid"]: e for e in entries_from_sets(cfg, backend, False)}
        rows = {r["qid"]: r for f in sorted(results_dir(cfg["run_name"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
        groot = gen_dir(cfg, False)
        for c in [c for c in sel["cases"] if c["pool"] == pool]:
            q = backend.load(ents[c["qid"]])
            gt = q["x"][1::2][:N_MODEL]
            g0 = np.load(groot / "base" / c["split"] / f"{c['qid']}.npz")["samples"][0]
            extra = {}
            if cfg.get("save_coarse_dir"):
                extra["coarse"] = np.load(Path(cfg["save_coarse_dir"]) / c["split"] / f"{c['qid']}.npz")["coarse"]
            if c["decision"] == "exact":
                ours = gt.copy()
            elif c["decision"] == "accept":
                ours = g0.copy()
            else:
                k = int(c["k"])
                sh = backend.make_sim(q, overrides=resolve_fidelity(cfg["shadow_fidelity"], q) or None)
                sh.step_frames(2 * k + 1)
                st = {kk: np.asarray(v) for kk, v in sh.export_state().items()}
                rov = (resolve_fidelity(cfg["restart_fidelity"], q) or None) if cfg.get("restart_fidelity") else None
                sim, _, blown = restart(backend, q, generated_frames(q["x0"], g0), k, state=st, sim_overrides=rov)
                if blown:
                    raise RuntimeError(f"{c['qid']}: restart at k={k} blew up, but the verdict run did not")
                ours = splice(g0, sim, k)
                check(c["name"], np.linalg.norm(ours - gt, axis=2).mean(axis=1), rows[c["qid"]]["k"][str(k)]["err_t"])
            if pool in LADDER:
                lcfg, fid = LADDER[pool]
                spec = load_config(lcfg)["fidelities"][fid]
                traj = backend.make_sim(q, overrides=resolve_fidelity(spec, q) or None).step_frames(48)[0::2][:N_MODEL]
                lad = {r["fidelity"]: r for f in sorted(results_dir(load_config(lcfg)["run_name"]).glob("shard_*/fullsim.jsonl"))
                       for r in read_jsonl(f) if r["qid"] == c["qid"]}
                check(f"{c['name']} always {fid}", np.linalg.norm(traj - gt, axis=2).mean(axis=1), lad[fid]["err_t"])
                extra["m1"] = traj
            np.savez_compressed(out / f"{args.prefix}{c['name']}.npz", ours=ours.astype(np.float32), gen=g0.astype(np.float32),
                                gt=gt.astype(np.float32), x0=np.asarray(q["x0"], np.float32),
                                object_id=np.asarray(q["object_id"]) if q.get("object_id") is not None
                                else np.zeros(gt.shape[1], int), **extra)
            print(f"[fig_ours] {c['name']}: {c['decision']} -> mean error "
                  f"{float(np.linalg.norm(ours - gt, axis=2).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
