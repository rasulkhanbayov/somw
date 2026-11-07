"""Phase 14a (CPU; pre-registered Oct 22, 2026, PLAN.md "Phase 14"): which candidates each arm acquires.

The one-window certificate (Phase 9: coarse_cfl, m = 4, window at the spread_event start of PhysCtrl base's 8 samples;
blown or crashed windows score +inf) is read from the exhaustive scan of the candidate pool. Rejected = score >= tau.
  ACT  all rejected candidates (the certificate decides what is worth simulating)
  RND  a uniformly random subset of the whole pool of the same size (seed 14)
Every candidate was simulated at the reference setting (the pilot's shared pool), so both arms come from one pool.
--wall-free (pilot 14a-v2, PLAN "Phase 14"): the pool is first restricted to candidates whose reference never comes
within 0.5 of the MPM domain boundary (p10e_wall rule); 84% of 14a's ACT arm were wall impacts.
Outputs: results/<run name>/{arm_ACT.json, arm_RND.json, SUMMARY.md, pool.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import spread_event  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p5a_amortise import light_query  # noqa: E402
from p10e_wall import wall_frame  # noqa: E402

TAU, SEED = 0.02614, 14


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p14a_scan.yaml")
    ap.add_argument("--wall-free", action="store_true")
    ap.add_argument("--run-name", default="p14a_acquire")
    args = ap.parse_args()
    cfg = load_config(args.config)
    oi = ood_index_from(cfg["query_runs"])
    groot = gen_dir(cfg, False)
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["run_name"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == "coarse_cfl" and int(r["m"]) == 4}
    pool, n_wall = [], 0
    for qid, path in sorted(oi.items()):
        split = qid.split("__")[-1]
        r = scan.get(qid)
        if r is None:
            raise RuntimeError(f"{qid}: no scan row")
        smp = np.load(groot / "base" / split / f"{qid}.npz")["samples"]
        q = {**light_query(path), "qid": qid}
        ev = spread_event(smp, q, 4)[0]
        w = {t: (np.inf if (bl or cr or x is None) else x) for t, x, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
        if ev not in w:
            raise RuntimeError(f"{qid}: event start {ev} not scanned")
        if args.wall_free:
            import h5py
            with h5py.File(path, "r") as h:
                if wall_frame(np.asarray(h["x"][1:48:2])) is not None:
                    n_wall += 1
                    continue
        pool.append({"qid": qid, "path": str(path), "score": float(w[ev]), "E": float(q["E"]),
                     "force_norm": float(np.linalg.norm(np.asarray(q.get("drag_force", [0, 0, 0])).reshape(-1)[:3]))})
    rej = [p for p in pool if p["score"] >= TAU]
    rng = np.random.default_rng(SEED)
    rnd = [pool[i] for i in sorted(rng.choice(len(pool), size=len(rej), replace=False))]
    run = start_run(args.run_name, cfg, synthetic=False)
    save_json({"tau": TAU, "files": [p["path"] for p in rej], "qids": [p["qid"] for p in rej]}, run.path("arm_ACT.json"))
    save_json({"seed": SEED, "files": [p["path"] for p in rnd], "qids": [p["qid"] for p in rnd]}, run.path("arm_RND.json"))
    save_json(pool, run.path("pool.json"))
    logE = np.log10([p["E"] for p in pool])
    md = ["# Phase 14a acquisition (pre-registered Oct 22, 2026)\n",
          (f"Wall-free restriction: {n_wall} candidates whose reference reaches the domain wall removed.\n" if args.wall_free else ""),
          f"Pool: {len(pool)} candidates (stable reference rollouts). Rejected by the one-window certificate (tau {TAU}): "
          f"{len(rej)} ({100 * len(rej) / len(pool):.0f}%) -> arm ACT; arm RND = {len(rnd)} random candidates (seed {SEED}).",
          f"Overlap ACT ∩ RND: {len({p['qid'] for p in rej} & {p['qid'] for p in rnd})}.", "",
          "| log10 E bin | candidates | rejected share |", "|---|---|---|"]
    for lo, hi in ((3, 4), (4, 7), (7, 8)):
        sel = [p for p, le in zip(pool, logE) if lo <= le < hi or (hi == 8 and le == 8)]
        md.append(f"| [{lo}, {hi}) | {len(sel)} | {np.mean([p['score'] >= TAU for p in sel]):.2f} |" if sel else f"| [{lo}, {hi}) | 0 | — |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    run.finish("ok")


if __name__ == "__main__":
    main()
