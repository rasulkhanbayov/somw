"""Phase 4g verdict: does the frame-conditioned generator propagate verified windows? CPU only.

Pre-registered Oct 12, 2026 (PLAN.md "Phase 4g").
  Primary, on base trajectories rejected by the Phase 4a primary certificate (R0 defect >=
  lambda, from the Phase 4b rows); PASS only if BOTH hold:
    (1) mean error on UNCONSTRAINED frames, R1 - R7: 95% paired-bootstrap lower bound > 0;
    (2) mean error on all frames, R1 - R7: lower bound > 0.
  Secondary (reported):
    - R7u vs R0 (fine-tuning alone);
    - R8 on frames >= 4 vs R0 (extrapolation from the anchor);
    - R7 vs R2;
    - certificate acceptance of R7 at the object-level lambda;
    - all base trajectories; test_id only.

Output: results/p4g_summary/SUMMARY.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bootstrap_ci, load_config, read_jsonl, results_dir, start_run  # noqa: E402
from p4b_summarize import region_err  # noqa: E402


def ci(a, b, nb):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    return bootstrap_ci(lambda x: float(np.mean(x)), [d], n_boot=nb) if len(d) >= 3 else (np.nan,) * 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.allow_synthetic)
    syn = args.allow_synthetic
    run = start_run(cfg.get("summary_run_name", "p4g_summary"), cfg, synthetic=syn)
    load = lambda name: {(r["qid"], r["model"]): r for f in sorted(results_dir(name, syn).glob("shard_*/repair.jsonl"))
                         for r in read_jsonl(f) if not r.get("crashed")}
    g, b = load(cfg.get("run_name", "p4g_repair")), load(cfg["p4b_run"])
    keys = sorted(set(g) & set(b))
    lam, lam_obj, nb = cfg["lambda_base"], cfg["lambda_object_level"], cfg["n_boot"]
    f = lambda t: "n/a" if not np.isfinite(t[0]) else f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"
    md = ["# Phase 4g: frame-conditioned repair (pre-registered Oct 12, 2026)\n",
          f"{len(keys)} base trajectories joined with Phase 4b. Constraints and seeds are identical to Phase 4b.\n"]
    verdict = None
    for title, sel in (("rejected by the certificate (PRIMARY)", lambda k: b[k]["R0__defect"] >= lam),
                       ("all base", lambda k: True), ("test_id only", lambda k: b[k]["split"] == "test_id")):
        ks = [k for k in keys if sel(k)]
        if not ks:
            continue
        md += [f"\n## {title} (n = {len(ks)})\n",
               "| region | R0 | R1 replace | R2 (4b) | **R7 FC** | R7u FC unconstr. | R8 FC anchor | R1 - R7 [95% CI] | R2 - R7 [95% CI] |",
               "|---|---|---|---|---|---|---|---|---|"]
        res = {}
        for region in ("all", "free", "before"):
            # free/before regions use the constraint layout, which is identical in 4b and 4g
            v = {n: [region_err(b[k], n, region) for k in ks] for n in ("R0", "R1", "R2")}
            v.update({n: [region_err(g[k], n, region) for k in ks] for n in ("R7", "R7u", "R8")})
            res[region] = (ci(v["R1"], v["R7"], nb), ci(v["R2"], v["R7"], nb))
            md.append(f"| {region} | " + " | ".join(f"{np.nanmean(v[n]):.4f}" for n in ("R0", "R1", "R2", "R7", "R7u", "R8"))
                      + f" | {f(res[region][0])} | {f(res[region][1])} |")
        e0 = [np.mean(b[k]["R0__err_t"][4:]) for k in ks]; e8 = [np.mean(g[k]["R8__err_t"][4:]) for k in ks]
        md.append(f"\nExtrapolation from the anchor (frames >= 4): R0 {np.mean(e0):.4f} -> R8 {np.mean(e8):.4f}; "
                  f"R0 - R8 {f(ci(e0, e8, nb))}")
        e0a = [np.mean(b[k]["R0__err_t"]) for k in ks]; e7u = [np.mean(g[k]["R7u__err_t"]) for k in ks]
        md.append(f"Fine-tuning alone (unconstrained): R0 {np.mean(e0a):.4f} vs R7u {np.mean(e7u):.4f}; R0 - R7u {f(ci(e0a, e7u, nb))}")
        md.append(f"Certificate acceptance after repair (lambda {lam_obj}): R7 "
                  f"{np.mean([g[k]['R7__defect'] < lam_obj for k in ks]):.2f}")
        if title == "test_id only":    # 4g-v2 pre-registered secondary: no degradation without constraints
            dgap = float(np.mean(e7u) - np.mean(e0a))
            md.append(f"Non-degradation (|R7u - R0| <= 0.005 on test_id): {dgap:+.4f} -> "
                      f"{'PASS' if abs(dgap) <= 0.005 else 'FAIL'}")
        if title.endswith("(PRIMARY)"):
            verdict = bool(res["free"][0][1] > 0 and res["all"][0][1] > 0)
    head = [f"- **Primary**: the frame-conditioned generator propagates verified windows (R1 - R7 > 0 on "
            f"unconstrained AND all frames): {'PASS' if verdict else 'FAIL' if verdict is not None else 'n/a'}", ""]
    if syn:
        head.insert(0, "> SYNTHETIC INPUT. These numbers test the pipeline only and mean nothing.\n")
    run.path("SUMMARY.md").write_text("\n".join(md[:2] + head + md[2:]) + "\n")
    print(f"[p4g_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
