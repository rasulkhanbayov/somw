"""Phase 10e, EXPLORATORY (Oct 24, 2026; written after a wall-proximity survey, not a pre-registered test).

The MPM grid is [0, 10]^3 and particles cannot leave it: they stop about two cells (0.16) from the edge. Under a strong
push the object reaches that edge inside the clip (survey Oct 24: 99% of Phase 9 ood3_force, 32% of ood3_E_low, 1-4% in
distribution), so the reference then contains a collision with a wall the scene does not have.
Wall reach (rule fixed here, before scoring): the reference at dt has a particle within WALL = 0.5 of the domain
boundary; k_w = the first such model frame.
Two readings of the stored Phase 10c trajectories ($BIG_ROOT/p10c_traj, decisions from results/p10c_decisions):
  wall-free:  risk on the queries whose reference never reaches the wall (the rest dropped);
  truncated:  every query, error = mean point distance over model frames < k_w only (queries with k_w < 4 dropped).
              A shorter horizon lowers every error, so these risks are not comparable with the full-clip ones.
Risk = P(error > 0.10) against the reference at dt, object-weighted, 95% object-cluster bootstrap CIs (2,000).
Output: results/p10e_wall/{SUMMARY.md, wall.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, results_dir, save_json, start_run  # noqa: E402
from p10c_summary import EPS, PIPES, outputs  # noqa: E402

WALL, MIN_FRAMES, B, LV = 0.5, 4, 2000, 0.95


def wall_frame(x):
    hit = ((x < WALL) | (x > 10.0 - WALL)).any(axis=(1, 2))
    return int(np.argmax(hit)) if hit.any() else None


def err(a, b, k=None):
    a, b = a[:k], b[:k]
    return np.inf if (np.isnan(a).any() or np.isnan(b).any()) else float(np.linalg.norm(a - b, axis=2).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p10c_rescore.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dec = load_json(results_dir(cfg["decisions_run"]) / "decisions.json")["decisions"]
    root = Path(os.path.expandvars(cfg["traj_dir"]))
    rows = []
    for d in dec:
        T = {k: v.astype(np.float64) for k, v in np.load(root / d["set"] / f"{d['qid']}.npz").items()}
        o, gt = outputs(d, T), T["gt"]
        kw = wall_frame(gt)
        grp = "contact" if d["set"] == "contact" else ("in distribution" if d["split"].startswith("id_new") else "shift")
        rows.append({"obj": d["obj"], "grp": grp, "split": d["split"], "kw": kw,
                     "full": {p: err(x, gt) for p, x in o.items()},
                     "trunc": None if kw is not None and kw < MIN_FRAMES else {p: err(x, gt, kw) for p, x in o.items()}})
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
        return (float(np.mean([by[o].mean() for o in objs])), float(np.quantile(bs, (1 - LV) / 2)),
                float(np.quantile(bs, 1 - (1 - LV) / 2)), len(rs))

    def risk(p, mode):
        if mode == "full":
            return lambda r: float(r["full"][p] > EPS) if p in r["full"] else None
        if mode == "wall-free":
            return lambda r: float(r["full"][p] > EPS) if p in r["full"] and r["kw"] is None else None
        return lambda r: float(r["trunc"][p] > EPS) if r["trunc"] is not None and p in r["trunc"] else None

    f = lambda c: "—" if c is None else f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]"                          # noqa: E731
    md = ["# Phase 10e exploratory: the MPM domain wall (Oct 24, 2026; NOT a pre-registered test)\n",
          f"Wall reach: a reference (dt) particle within {WALL} of the [0, 10] boundary. risk = P(error > {EPS}) vs the "
          f"reference at dt, object-weighted, {LV:.0%} object-cluster CIs. Truncated = error over model frames before the "
          f"first wall frame (queries with < {MIN_FRAMES} wall-free frames dropped); not comparable with full-clip risks.\n",
          "| set | queries | reach the wall | median first wall frame |", "|---|---|---|---|"]
    for nm, sel in sets.items():
        ks = [r["kw"] for r in sel if r["kw"] is not None]
        md.append(f"| {nm} | {len(sel)} | {len(ks)} ({100 * len(ks) / len(sel):.0f}%) | {np.median(ks) if ks else '—'} |")
    md.append("")
    res = {}
    for nm, sel in sets.items():
        md += [f"## {nm}", "", "| pipeline | full clip | wall-free queries | truncated before the wall |", "|---|---|---|---|"]
        res[nm] = {}
        for p in PIPES:
            c = {m: stat(sel, risk(p, m)) for m in ("full", "wall-free", "truncated")}
            if c["full"] is None:
                continue
            res[nm][p] = c
            md.append(f"| {p} | {f(c['full'])} | {f(c['wall-free'])} (n {c['wall-free'][3] if c['wall-free'] else 0}) | "
                      f"{f(c['truncated'])} (n {c['truncated'][3] if c['truncated'] else 0}) |")
        md.append("")
    run = start_run("p10e_wall", cfg, synthetic=False)
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"rows": [{k: r[k] for k in ("obj", "grp", "split", "kw")} for r in rows], "risk": res}, run.path("wall.json"))
    print("\n".join(md))
    run.finish("ok")


if __name__ == "__main__":
    main()
