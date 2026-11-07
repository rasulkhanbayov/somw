"""Phase 4c verdict: conditioning repair vs the Phase 4b methods. CPU only.

Pre-registered Oct 11, 2026 (see p4c_repair_conditioning.py, PLAN.md "Phase 4c").
  Primary: base trajectories rejected by the Phase 4a primary certificate (R0 defect >= lambda,
    taken from the Phase 4b rows). PASS if mean err(R1) - err(R4) over the 24 frames has 95%
    paired-bootstrap lower bound > 0.
  Secondary (reported): R2 - R4, R0 - R4, R4id - R4 (does selection add anything beyond
    re-sampling?); the chosen adjustment per split; held-out defect R0 vs R4 (uniform windows
    at 'ref', never used in selection); certificate-signal acceptance after repair, marked as
    selection-biased; all base and all large trajectories.

Output: results/p4c_summary/SUMMARY.md
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bootstrap_ci, load_config, read_jsonl, results_dir, start_run  # noqa: E402


def mean_err(r, name):
    return float(np.mean(r[f"{name}__err_t"]))


def ci(a, b, nb):
    d = np.asarray(a) - np.asarray(b)
    return bootstrap_ci(lambda x: float(np.mean(x)), [d], n_boot=nb) if len(d) >= 3 else (np.nan,) * 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.allow_synthetic)
    syn = args.allow_synthetic
    run = start_run(cfg.get("summary_run_name", "p4c_summary"), cfg, synthetic=syn)
    load = lambda name: {(r["qid"], r["model"]): r for f in sorted(results_dir(name, syn).glob("shard_*/repair.jsonl"))
                         for r in read_jsonl(f) if not r.get("crashed")}
    c, b = load(cfg.get("run_name", "p4c_repair")), load(cfg["p4b_run"])
    keys = sorted(set(c) & set(b))
    lam, nb = cfg["lambda_base"], cfg["n_boot"]
    f = lambda t: "n/a" if not np.isfinite(t[0]) else f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"
    md = ["# Phase 4c: conditioning repair (pre-registered Oct 11, 2026)\n",
          f"Joined with Phase 4b on {len(keys)} (query, model) pairs. lambda = {lam}. Errors: mean per-point L2 to "
          "ground truth over 24 frames.\n"]
    verdict = None
    for title, sel in (("base, rejected by the certificate (PRIMARY)", lambda k: k[1] == "base" and b[k]["R0__defect"] >= lam),
                       ("base, all", lambda k: k[1] == "base"), ("large, all", lambda k: k[1] == "large")):
        ks = [k for k in keys if sel(k)]
        if not ks:
            continue
        e = {n: [mean_err(b[k], n) for k in ks] for n in ("R0", "R1", "R2")}
        e.update({n: [mean_err(c[k], n) for k in ks] for n in ("R4", "R4id")})
        d14, d24, d04, did = ci(e["R1"], e["R4"], nb), ci(e["R2"], e["R4"], nb), ci(e["R0"], e["R4"], nb), ci(e["R4id"], e["R4"], nb)
        md += [f"\n## {title} (n = {len(ks)})\n",
               "| R0 original | R1 replace | R2 constrained | R4id re-sample | **R4 conditioning** |",
               "|---|---|---|---|---|",
               "| " + " | ".join(f"{np.mean(e[n]):.4f}" for n in ("R0", "R1", "R2", "R4id", "R4")) + " |",
               f"\nR1 - R4 {f(d14)}; R2 - R4 {f(d24)}; R0 - R4 {f(d04)}; R4id - R4 {f(did)}"]
        if title.endswith("(PRIMARY)"):
            verdict = bool(np.isfinite(d14[1]) and d14[1] > 0)
        h0 = np.array([c[k]["R0__heldout_defect"] for k in ks]); h4 = np.array([c[k]["R4__heldout_defect"] for k in ks])
        ok = np.isfinite(h0) & np.isfinite(h4)
        md.append(f"\nHeld-out defect (uniform, 'ref', m = 4; never used in selection): median R0 {np.median(h0[ok]):.4f} -> "
                  f"R4 {np.median(h4[ok]):.4f}; lower for R4 in {np.mean(h4[ok] < h0[ok]):.2f} of trajectories.")
        if title.startswith("base"):
            acc = np.mean([c[k]["R4__cert_defect"] < lam for k in ks])
            md.append(f"Accepted at lambda after repair with the certificate's own signal (selection-biased): R4 {acc:.2f}.")
        chosen = Counter((b[k]["split"], c[k]["chosen"]["dlogE"], c[k]["chosen"]["force_scale"]) for k in ks)
        by_split = {}
        for (sp, d, s), n in chosen.items():
            by_split.setdefault(sp, []).append((n, d, s))
        md.append("\nMost-chosen adjustments per split (count: dlog10E, force scale): " + "; ".join(
            f"{sp}: " + ", ".join(f"{n}: ({d:+g}, x{s:g})" for n, d, s in sorted(v, reverse=True)[:3])
            for sp, v in sorted(by_split.items())))
    head = [f"- **Primary**: conditioning repair beats replacement on rejected base trajectories: "
            f"{'PASS' if verdict else 'FAIL' if verdict is not None else 'n/a'}", ""]
    if syn:
        head.insert(0, "> SYNTHETIC INPUT. These numbers test the pipeline only and mean nothing.\n")
    run.path("SUMMARY.md").write_text("\n".join(md[:2] + head + md[2:]) + "\n")
    print(f"[p4c_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
