"""Phase 10c, EXPLORATORY (Oct 21, 2026; written after the pre-registered verdict in results/p10c_summary, not a test).

Two readings of the same re-scored trajectories ($BIG_ROOT/p10c_traj, decisions from results/p10c_decisions):
  gap:    risk(pipeline) - risk(always simulate at the reference step dt), paired per query, against dt, dt/2 and dt/4.
          It asks whether a pipeline is as safe as running the reference simulator itself, whichever reference is truth.
  robust: P(error > 0.10 against all three references), a failure that no choice of reference step removes.
Object-cluster bootstrap, 2,000 resamples, 95% CIs; queries with an unstable reference at dt/2 or dt/4 are left out of
the columns that need them.
Output: results/p10c_explore/{SUMMARY.md, explore.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, results_dir, save_json, start_run  # noqa: E402
from p10c_summary import EPS, PIPES, REFS, err, outputs  # noqa: E402

B, LV, BASE = 2000, 0.95, "reference (dt)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dec = load_json(results_dir(cfg["decisions_run"]) / "decisions.json")["decisions"]
    root = Path(os.path.expandvars(cfg["traj_dir"]))
    rows = []
    for d in dec:
        T = {k: v.astype(np.float64) for k, v in np.load(root / d["set"] / f"{d['qid']}.npz").items()}
        o = outputs(d, T)
        e = {rn: (None if np.isnan(T[rk]).any() else {p: err(x, T[rk]) for p, x in o.items()}) for rn, rk in REFS.items()}
        grp = "contact" if d["set"] == "contact" else ("in distribution" if d["split"].startswith("id_new") else "shift")
        rows.append({"obj": d["obj"], "grp": grp, "split": d["split"], "e": e})
    sets = {g: [r for r in rows if r["grp"] == g] for g in ("in distribution", "shift", "contact")}
    for fam in sorted({r["split"] for r in rows if r["grp"] == "shift"}):
        sets[fam] = [r for r in rows if r["split"] == fam]

    def stat(sel, fn):
        rs = [(r["obj"], fn(r)) for r in sel]
        rs = [(o, v) for o, v in rs if v is not None]
        if not rs:
            return None
        objs = sorted({o for o, _ in rs})
        by = {o: np.array([v for oo, v in rs if oo == o], float) for o in objs}
        rng = np.random.default_rng(1)
        bs = [np.mean([by[o].mean() for o in rng.choice(objs, len(objs))]) for _ in range(B)]
        return (float(np.mean([by[o].mean() for o in objs])), float(np.quantile(bs, (1 - LV) / 2)), float(np.quantile(bs, 1 - (1 - LV) / 2)))

    def gap(p, rn):
        return lambda r: None if r["e"][rn] is None or p not in r["e"][rn] else \
            float(r["e"][rn][p] > EPS) - float(r["e"][rn][BASE] > EPS)

    def robust(p):
        return lambda r: None if any(r["e"][rn] is None for rn in REFS) or p not in r["e"]["dt"] else \
            float(all(r["e"][rn][p] > EPS for rn in REFS))
    f = lambda c: "—" if c is None else f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"                          # noqa: E731
    md = ["# Phase 10c exploratory: gap to always-simulate at dt, and robust risk (Oct 21, 2026; NOT a pre-registered test)\n",
          f"risk = P(error > {EPS}), object-weighted; {LV:.0%} object-cluster bootstrap CIs.\n"]
    res = {}
    for nm, sel in sets.items():
        md += [f"## {nm} ({len(sel)} queries)", "", "| pipeline | robust risk | " + " | ".join(f"gap vs {rn}" for rn in REFS) + " |",
               "|---|---|" + "---|" * len(REFS)]
        res[nm] = {}
        for p in PIPES:
            rb = stat(sel, robust(p))
            gs = {rn: stat(sel, gap(p, rn)) for rn in REFS}
            if rb is None:
                continue
            res[nm][p] = {"robust": rb, "gap": gs}
            md.append(f"| {p} | {rb[0]:.3f} | " + " | ".join(f(gs[rn]) for rn in REFS) + " |")
        md.append("")
    run = start_run("p10c_explore", cfg, synthetic=False)
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json(res, run.path("explore.json"))
    print(f"[p10c explore] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
