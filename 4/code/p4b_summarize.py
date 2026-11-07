"""Phase 4b verdict: does constrained re-sampling repair better than replacement? CPU only.

Pre-registered Oct 11, 2026 (see p4b_repair.py and PLAN.md "Phase 4b").
  Primary: base trajectories REJECTED by the Phase 4a primary certificate (R0 defect >= lambda).
    PASS if mean err(R1) - err(R2) over the 24 frames has 95% paired-bootstrap lower bound > 0.
  Secondary (reported):
    - R2 vs R0p (is it the constraints or just re-sampling?);
    - R3 (anchor only) vs R2;
    - backward propagation: error on frames BEFORE the first non-anchor constraint (and after
      the anchor), where R1 equals R0 by construction; R0 - R2 > 0 means a fix reached back;
    - error on unconstrained frames only;
    - acceptance after repair at the same lambda;
    - the same tables for all base trajectories and for large.

Output: results/p4b_summary/SUMMARY.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bootstrap_ci, load_config, read_jsonl, results_dir, start_run  # noqa: E402


def region_err(r, name, region):
    e = np.array(r[f"{name}__err_t"])
    cons = set(r["constrained"]); anch = set(r["anchor_frames"])
    if region == "all":
        idx = np.arange(24)
    elif region == "free":
        idx = np.array([k for k in range(24) if k not in cons])
    else:  # "before": after the anchor, before the first non-anchor constrained frame
        later = [k for k in cons if k not in anch]
        first = min(later) if later else 24
        idx = np.array([k for k in range(24) if k not in anch and k < first])
    return float(e[idx].mean()) if len(idx) else np.nan


def diff_ci(a, b, n_boot):
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return (np.nan, np.nan, np.nan)
    return bootstrap_ci(lambda x: float(np.mean(x)), [d], n_boot=n_boot)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.allow_synthetic)
    run = start_run(cfg.get("summary_run_name", "p4b_summary"), cfg, synthetic=args.allow_synthetic)
    rows = [r for f in sorted(results_dir(cfg.get("run_name", "p4b_repair"), args.allow_synthetic)
                              .glob("shard_*/repair.jsonl")) for r in read_jsonl(f) if not r.get("crashed")]
    lam, nb = cfg["lambda_base"], cfg["n_boot"]
    md = ["# Phase 4b: repair by constrained re-sampling (pre-registered Oct 11, 2026)\n",
          f"lambda (Phase 4a primary, base) = {lam}. Errors are mean per-point L2 to ground truth, simulator units.\n"]
    verdict = None
    for title, sel in (("base, rejected by the certificate (PRIMARY)",
                        lambda r: r["model"] == "base" and r["R0__defect"] >= lam),
                       ("base, all", lambda r: r["model"] == "base"),
                       ("large, all", lambda r: r["model"] == "large")):
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        md += [f"\n## {title} (n = {len(rs)})\n",
               "| region | R0 original | R0p re-sample | R1 replace | R2 constrained | R3 anchor only | "
               "R1 - R2 [95% CI] | R0p - R2 [95% CI] | R0 - R2 [95% CI] |",
               "|---|---|---|---|---|---|---|---|---|"]
        for region in ("all", "free", "before"):
            v = {n: [region_err(r, n, region) for r in rs] for n in ("R0", "R0p", "R1", "R2", "R3")}
            c12, c02p, c02 = diff_ci(v["R1"], v["R2"], nb), diff_ci(v["R0p"], v["R2"], nb), diff_ci(v["R0"], v["R2"], nb)
            f = lambda t: "n/a" if not np.isfinite(t[0]) else f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"
            md.append(f"| {region} | " + " | ".join(f"{np.nanmean(v[n]):.4f}" for n in ("R0", "R0p", "R1", "R2", "R3"))
                      + f" | {f(c12)} | {f(c02p)} | {f(c02)} |")
            if title.endswith("(PRIMARY)") and region == "all":
                verdict = bool(np.isfinite(c12[1]) and c12[1] > 0)
        if title.startswith("base"):
            acc = {n: np.mean([r[f"{n}__defect"] < lam for r in rs]) for n in ("R0", "R1", "R2")}
            md.append(f"\nAccepted at lambda after repair: R0 {acc['R0']:.2f}, R1 {acc['R1']:.2f}, R2 {acc['R2']:.2f}")
    head = [f"- **Primary**: constrained re-sampling beats replacement on rejected base trajectories: "
            f"{'PASS' if verdict else 'FAIL' if verdict is not None else 'n/a'}", ""]
    if args.allow_synthetic:
        head.insert(0, "> SYNTHETIC INPUT. These numbers test the pipeline only and mean nothing.\n")
    run.path("SUMMARY.md").write_text("\n".join(md[:2] + head + md[2:]) + "\n")
    print(f"[p4b_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
