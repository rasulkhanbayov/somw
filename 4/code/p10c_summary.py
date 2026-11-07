"""Phase 10c verdict (CPU; pre-registered Oct 21, 2026; PLAN.md "Phase 10c"): every Phase 9 pipeline re-scored exactly
against the reference at dt, dt/2 and dt/4 (positions from job 101, $BIG_ROOT/p10c_traj).

Pipelines: PhysCtrl (generation), coarse (single objects), M1, M2, M3, always reference (= the dt rollout), ours,
certify-or-simulate, momentum-residual gate and seed-spread deferral (single objects), by their Phase 9 decisions
(results/p10c_decisions). Risk = P(error > 0.10), object-weighted; an unstable pipeline rollout counts as wrong; a query
whose reference rollout is unstable is left out of that reference's column (counted).
Tests (object-cluster bootstrap, 2,000 resamples, 98.3% CIs: Bonferroni over three families):
  T1 risk(PhysCtrl) - risk(ours) > 0 in every set against every reference
  T2 risk(ours) <= 0.03 in every set against every reference (point estimates)
  T3 under shift, risk(M) - risk(ours) > 0 against every reference, for each M in M1, M2, M3
  --sandbox DIR: read trajectories from DIR instead (code test only; output also goes to DIR)
Output: results/p10c_summary/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, results_dir, save_json, start_run  # noqa: E402

EPS, SAFE, B, LV = 0.10, 0.03, 2000, 1 - 0.05 / 3
REFS = {"dt": "gt", "dt/2": "half", "dt/4": "quarter"}
PIPES = ["PhysCtrl", "coarse", "M1", "M2", "M3", "reference (dt)", "residual gate", "seed spread", "certify or simulate", "ours"]


def outputs(d, T):
    g, gt = T["gen"], T["gt"]
    o = {"PhysCtrl": g, "M1": T["M1"], "M2": T["M2"], "M3": T["M3"], "reference (dt)": gt, "ours": T["ours"],
         "certify or simulate": g if d["cert"] == "accept" else gt}
    if d["set"] == "single":
        o["coarse"] = T["coarse_cfl"]
        o["residual gate"] = g if d["residual"] == "accept" else gt
        o["seed spread"] = g if d["seed"] == "accept" else gt
    return o


def err(a, b):
    return np.inf if (np.isnan(a).any() or np.isnan(b).any()) else float(np.linalg.norm(a - b, axis=2).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sandbox", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dec = load_json((Path(args.sandbox) if args.sandbox else results_dir(cfg["decisions_run"])) / "decisions.json")["decisions"]
    root = Path(args.sandbox) if args.sandbox else Path(os.path.expandvars(cfg["traj_dir"]))
    rows = []
    for d in dec:
        f = root / d["set"] / f"{d['qid']}.npz"
        if not f.exists():
            raise RuntimeError(f"missing trajectories for {d['qid']}")
        T = {k: v.astype(np.float64) for k, v in np.load(f).items()}
        o = outputs(d, T)
        e = {}
        for rn, rk in REFS.items():
            ref = T[rk]
            e[rn] = None if np.isnan(ref).any() else {p: err(x, ref) for p, x in o.items()}
        refd = {"dt-dt/2": err(T["gt"], T["half"]), "dt-dt/4": err(T["gt"], T["quarter"]), "dt/2-dt/4": err(T["half"], T["quarter"])}
        grp = "contact" if d["set"] == "contact" else ("in distribution" if d["split"].startswith("id_new") else "shift")
        rows.append({"qid": d["qid"], "obj": d["obj"], "grp": grp, "split": d["split"], "e": e, "refd": refd})
    sets = {g: [r for r in rows if r["grp"] == g] for g in ("in distribution", "shift", "contact")}
    for fam in sorted({r["split"] for r in rows if r["grp"] == "shift"}):
        sets[fam] = [r for r in rows if r["split"] == fam]

    def stat(sel, rn, fn):
        """Object-weighted mean of fn(row) over rows with a stable reference rn; bootstrap CI over objects."""
        rs = [r for r in sel if r["e"][rn] is not None and fn(r) is not None]
        if not rs:
            return (np.nan,) * 3, 0
        objs = sorted({r["obj"] for r in rs})
        by = {o: np.array([fn(r) for r in rs if r["obj"] == o], float) for o in objs}
        pt = float(np.mean([by[o].mean() for o in objs]))
        rng = np.random.default_rng(1)
        bs = [np.mean([by[o].mean() for o in rng.choice(objs, len(objs))]) for _ in range(B)]
        return (pt, float(np.quantile(bs, (1 - LV) / 2)), float(np.quantile(bs, 1 - (1 - LV) / 2))), len(rs)
    bad = lambda p, rn: (lambda r: float(r["e"][rn][p] > EPS) if p in r["e"][rn] else None)              # noqa: E731
    md = ["# Phase 10c: every pipeline re-scored against the reference at dt, dt/2 and dt/4 (pre-registered Oct 21, 2026)\n",
          f"Queries: {len(rows)}; risk = P(error > {EPS}), object-weighted; CIs {LV:.3f} (Bonferroni over 3).\n"]
    res = {}
    for nm, sel in sets.items():
        md += [f"## {nm} ({len(sel)} queries)", "", "| pipeline | " + " | ".join(f"risk vs {rn}" for rn in REFS) + " |",
               "|---|" + "---|" * len(REFS)]
        res[nm] = {}
        for p in PIPES:
            cells = []
            for rn in REFS:
                (pt, _, _), n = stat(sel, rn, bad(p, rn))
                res[nm].setdefault(p, {})[rn] = pt
                cells.append("—" if n == 0 else f"{pt:.3f}")
            if any(c != "—" for c in cells):
                md.append(f"| {p} | " + " | ".join(cells) + " |")
        dd = {k: np.median([r["refd"][k] for r in sel if np.isfinite(r["refd"][k])]) for k in ("dt-dt/2", "dt-dt/4", "dt/2-dt/4")}
        md += ["", "Median distance between references: " + ", ".join(f"{k} {v:.4f}" for k, v in dd.items()),
               "Unstable reference rollouts (left out): " + ", ".join(f"{rn} {sum(r['e'][rn] is None for r in sel)}" for rn in REFS), ""]
    T1, T2, T3 = {}, {}, {}
    for nm in ("in distribution", "shift", "contact"):
        for rn in REFS:
            ci, _ = stat(sets[nm], rn, lambda r: (r["e"][rn]["PhysCtrl"] > EPS) - (r["e"][rn]["ours"] > EPS))
            T1[(nm, rn)] = ci
            T2[(nm, rn)] = res[nm]["ours"][rn]
    for m in ("M1", "M2", "M3"):
        for rn in REFS:
            ci, _ = stat(sets["shift"], rn, lambda r: (r["e"][rn][m] > EPS) - (r["e"][rn]["ours"] > EPS))
            T3[(m, rn)] = ci
    t1 = all(v[1] > 0 for v in T1.values())
    t2 = all(v <= SAFE for v in T2.values())
    t3 = {m: all(T3[(m, rn)][1] > 0 for rn in REFS) for m in ("M1", "M2", "M3")}
    f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"                                                 # noqa: E731
    md += ["## Tests", ""]
    md += [f"- T1 {nm}, vs {rn}: risk(PhysCtrl) - risk(ours) {f(c)}" for (nm, rn), c in T1.items()]
    md += [f"- **T1 (PhysCtrl worse than ours everywhere, every reference): {'PASS' if t1 else 'FAIL'}**", ""]
    md += [f"- T2 {nm}, vs {rn}: risk(ours) {v:.3f}" for (nm, rn), v in T2.items()]
    md += [f"- **T2 (ours <= {SAFE} everywhere, every reference): {'PASS' if t2 else 'FAIL'}**", ""]
    md += [f"- T3 shift, {m} vs {rn}: risk({m}) - risk(ours) {f(c)}" for (m, rn), c in T3.items()]
    md += [f"- **T3 {m} (less safe than ours under shift against every reference): {'PASS' if v else 'FAIL'}**" for m, v in t3.items()]
    out = Path(args.sandbox) / "p10c_summary" if args.sandbox else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
        print("sandbox summary written", out)
        return
    run = start_run("p10c_summary", cfg, synthetic=False)
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"T1": t1, "T2": t2, "T3": t3, "risk": res,
               "T1_ci": {f"{a}|{b}": c for (a, b), c in T1.items()}, "T3_ci": {f"{a}|{b}": c for (a, b), c in T3.items()}},
              run.path("summary.json"))
    print(f"[p10c] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
