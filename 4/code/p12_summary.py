"""Phase 12 verdict (CPU; pre-registered Oct 22, 2026, PLAN.md "Phase 12"): every Phase 9 pipeline re-scored against
the independent simulator S2 (PolyFEM) on the primary single-object sets.

Positions: pipelines and the MPM reference from job 101 ($BIG_ROOT/p10c_traj), decisions from results/p10c_decisions,
S2 from p12_test ($BIG_ROOT/p12_traj/test). Error = mean point distance; risk = P(error > 0.10), object-weighted.
Frames: PolyFEM has no walls, so wherever the MPM reference comes within 0.5 of the domain boundary (p10e rule) only the
frames before it are compared (queries with < 4 such frames dropped and counted); fixed in PLAN before any test query.
Sets: in distribution (id_new_22 + 23), ood3_E_high, ood3_E_low, ood3_force. Object-cluster bootstrap, 2,000 resamples,
98.75% CIs (Bonferroni over the 4 sets).
  X0 (descriptive) MPM-S2 distance per set: median, p95, share > 0.10
  X1 risk_S2(PhysCtrl) - risk_S2(ours) > 0 in every set (CI lower bound > 0)
  X2 Delta_S2 / Delta_MPM >= 0.5 in every set (point estimates), Delta = risk(PhysCtrl) - risk(ours)
  X3 on fallback queries (ours = the MPM reference) pooled: share with d(MPM, S2) < d(PhysCtrl, S2), CI lower > 0.5;
     repairs reported separately (descriptive)
  --sandbox DIR: S2 positions from DIR (code test only; output there too)
Output: results/p12_summary/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, load_json, save_json, start_run  # noqa: E402
from p10c_summary import EPS, PIPES, err, outputs  # noqa: E402
from p10e_wall import MIN_FRAMES, wall_frame  # noqa: E402

B, LV = 2000, 1 - 0.05 / 4
SETS = {"in distribution": ("id_new_22", "id_new_23"), "ood3_E_high": ("ood3_E_high",),
        "ood3_E_low": ("ood3_E_low",), "ood3_force": ("ood3_force",)}


def boot(rows, fn):
    rs = [(r["obj"], fn(r)) for r in rows]
    rs = [(o, v) for o, v in rs if v is not None]
    if not rs:
        return None
    objs = sorted({o for o, _ in rs})
    by = {o: np.array([v for oo, v in rs if oo == o], float) for o in objs}
    rng = np.random.default_rng(12)
    bs = [np.mean([by[o].mean() for o in rng.choice(objs, len(objs))]) for _ in range(B)]
    return (float(np.mean([by[o].mean() for o in objs])), float(np.quantile(bs, (1 - LV) / 2)),
            float(np.quantile(bs, 1 - (1 - LV) / 2)), len(rs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sandbox", default=None)
    args = ap.parse_args()
    root = Path(RESULTS_ROOT)
    dec = [d for d in load_json(root / "p10c_decisions" / "decisions.json")["decisions"]
           if d["set"] == "single" and any(d["split"] in v for v in SETS.values())]
    traj = Path(os.path.expandvars("${BIG_ROOT}/p10c_traj"))
    s2dir = Path(args.sandbox) if args.sandbox else Path(os.path.expandvars("${BIG_ROOT}/p12_traj/test"))
    rows, missing, short = [], 0, 0
    for d in dec:
        f = s2dir / f"{d['qid']}.npz"
        if not f.exists():
            missing += 1
            continue
        s2 = np.load(f)["s2"].astype(np.float64)
        T = {k: v.astype(np.float64) for k, v in np.load(traj / d["set"] / f"{d['qid']}.npz").items()}
        kw = wall_frame(T["gt"])
        k = 24 if kw is None else kw
        if k < MIN_FRAMES:
            short += 1
            continue
        o = outputs(d, T)
        rows.append({"qid": d["qid"], "obj": d["obj"], "split": d["split"], "ours": d["ours"], "frames": k,
                     "s2": {p: err(x[:k], s2[:k]) for p, x in o.items()},
                     "mpm": {p: err(x[:k], T["gt"][:k]) for p, x in o.items()},
                     "d_ms": err(T["gt"][:k], s2[:k])})
    sel = {nm: [r for r in rows if r["split"] in sp] for nm, sp in SETS.items()}
    risk = lambda key, p: (lambda r: float(r[key][p] > EPS) if p in r[key] else None)              # noqa: E731
    gap = lambda key: (lambda r: float(r[key]["PhysCtrl"] > EPS) - float(r[key]["ours"] > EPS))     # noqa: E731
    f = lambda c: "—" if c is None else f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"                       # noqa: E731
    res, md = {}, ["# Phase 12: pipelines re-scored against an independent FEM (pre-registered Oct 22, 2026)\n",
                   f"Queries with S2 positions: {len(rows) + short} of {len(dec)} ({missing} missing: excluded by the mesh "
                   f"rule, solver failures or not run); {short} dropped (< {MIN_FRAMES} frames before the wall). "
                   f"risk = P(error > {EPS}); object-weighted; {LV:.2%} object-cluster CIs.\n",
                   "## X0: MPM vs FEM (descriptive)", "", "| set | queries | median | p95 | share > 0.10 |",
                   "|---|---|---|---|---|"]
    for nm, rs in sel.items():
        v = np.array([r["d_ms"] for r in rs])
        res.setdefault(nm, {})["X0"] = {"n": len(rs), "median": float(np.median(v)) if len(v) else None,
                                        "p95": float(np.quantile(v, 0.95)) if len(v) else None,
                                        "share": float(np.mean(v > EPS)) if len(v) else None}
        x = res[nm]["X0"]
        md.append(f"| {nm} | {len(rs)} | {x['median']:.4f} | {x['p95']:.4f} | {x['share']:.3f} |" if len(rs)
                  else f"| {nm} | 0 | — | — | — |")
    md += ["", "## Risk against FEM and against the MPM reference (same frames)", "",
           "| set | pipeline | risk vs FEM | risk vs MPM |", "|---|---|---|---|"]
    for nm, rs in sel.items():
        for p in PIPES:
            a, b = boot(rs, risk("s2", p)), boot(rs, risk("mpm", p))
            if a is None:
                continue
            res[nm].setdefault("risk", {})[p] = {"s2": a, "mpm": b}
            md.append(f"| {nm} | {p} | {f(a)} | {f(b)} |")
    md += ["", "## Tests", ""]
    x1, x2 = {}, {}
    for nm, rs in sel.items():
        g2, gm = boot(rs, gap("s2")), boot(rs, gap("mpm"))
        x1[nm] = bool(g2 is not None and g2[1] > 0)
        x2[nm] = None if gm is None or gm[0] <= 0 or g2 is None else g2[0] / gm[0]
        res[nm].update(delta_s2=g2, delta_mpm=gm, X1=x1[nm], X2_ratio=x2[nm])
        md.append(f"- {nm}: Δ_FEM = {f(g2)}, Δ_MPM = {f(gm)}; X1 {'PASS' if x1[nm] else 'FAIL'}; X2 ratio "
                  + ("undefined (Δ_MPM <= 0)" if x2[nm] is None else f"{x2[nm]:.2f} {'PASS' if x2[nm] >= 0.5 else 'FAIL'}"))
    fb = [r for r in rows if r["ours"] == "reference"]
    rp = [r for r in rows if r["ours"] == "repair"]
    x3 = boot(fb, lambda r: float(r["d_ms"] < r["s2"]["PhysCtrl"]))
    rep = boot(rp, lambda r: float(r["s2"]["ours"] < r["s2"]["PhysCtrl"]))
    X1, X2 = all(x1.values()), all(v is not None and v >= 0.5 for v in x2.values())
    X3 = bool(x3 is not None and x3[1] > 0.5)
    md += [f"- X3 (fallback queries pooled, n {x3[3] if x3 else 0}): share d(MPM, FEM) < d(PhysCtrl, FEM) = {f(x3)}: "
           f"**{'PASS' if X3 else 'FAIL'}**",
           f"- Repairs (descriptive, n {rep[3] if rep else 0}): share d(ours, FEM) < d(PhysCtrl, FEM) = {f(rep)}",
           f"- **X1 {'PASS' if X1 else 'FAIL'}, X2 {'PASS' if X2 else 'FAIL'}, X3 {'PASS' if X3 else 'FAIL'}**", "",
           "Interpretation (fixed in PLAN before data): " + (
               "X1 and X3 PASS → the advantage is not an artifact of scoring with our own solver (single-object drag)."
               if X1 and X3 else "X3 FAIL → no \"more accurate than PhysCtrl\" claim beyond fidelity." if not X3 else
               "X1 FAIL → fidelity-only claims; discussed with the user before any paper change.")]
    res.update(X1=X1, X2=X2, X3=X3, x3=x3, repairs=rep, n_rows=len(rows), missing=missing, short=short)
    out = Path(args.sandbox) if args.sandbox else None
    if out is None:
        run = start_run("p12_summary", {"source": "p12_test"}, synthetic=False)
        out = run.path("")
    (Path(out) / "SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json(res, Path(out) / "summary.json")
    print("\n".join(md))
    if not args.sandbox:
        run.finish("ok")


if __name__ == "__main__":
    main()
