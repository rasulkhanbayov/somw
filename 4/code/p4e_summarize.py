"""Phase 4e verdict: anchored conditioning repair. CPU only.

Pre-registered Oct 12, 2026 (see p4e_repair_anchored.py, PLAN.md "Phase 4e").
  Primary: base trajectories rejected by the Phase 4a primary certificate (R0 defect >= lambda,
    from the Phase 4b rows). PASS if mean err(R1) - err(R6_8) has 95% lower bound > 0.
  Secondary (reported):
    - R5_L vs R0 on frames >= L: does matching an exact start fix the generated remainder?
    - R6_L vs R0 and R1 overall;
    - the anchor's own error;
    - chosen adjustments per split;
    - L = 12;
    - all base and all large trajectories.

Output: results/p4e_summary/SUMMARY.md
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bootstrap_ci, load_config, read_jsonl, results_dir, start_run  # noqa: E402


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
    run = start_run(cfg.get("summary_run_name", "p4e_summary"), cfg, synthetic=syn)
    load = lambda name: {(r["qid"], r["model"]): r for f in sorted(results_dir(name, syn).glob("shard_*/repair.jsonl"))
                         for r in read_jsonl(f) if not r.get("crashed")}
    e4, b = load(cfg.get("run_name", "p4e_repair")), load(cfg["p4b_run"])
    keys = sorted(set(e4) & set(b))
    lam, nb, Ls = cfg["lambda_base"], cfg["n_boot"], sorted(cfg["anchor_lengths"])
    f = lambda t: "n/a" if not np.isfinite(t[0]) else f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"
    # A non-finite trajectory (stored as null) is a failure: score it +inf, never drop it.
    def _m(vals):
        return float("inf") if any(v is None for v in vals) else float(np.mean(vals))
    m_all = lambda r, n: _m(r[f"{n}__err_t"])
    m_from = lambda r, n, L: _m(r[f"{n}__err_t"][L:])
    md = ["# Phase 4e: anchored conditioning repair (pre-registered Oct 12, 2026)\n",
          f"Joined with Phase 4b on {len(keys)} pairs. lambda = {lam}. Errors: mean per-point L2 to ground truth.\n"]
    verdict = None
    for title, sel in (("base, rejected by the certificate (PRIMARY)", lambda k: k[1] == "base" and b[k]["R0__defect"] >= lam),
                       ("base, all", lambda k: k[1] == "base"), ("large, all", lambda k: k[1] == "large")):
        ks = [k for k in keys if sel(k)]
        if not ks:
            continue
        R0 = [m_all(b[k], "R0") for k in ks]; R1 = [m_all(b[k], "R1") for k in ks]
        anc = np.mean([np.mean(e4[k]["anchor_err_t"]) for k in ks])
        n_bad = sum(any(v is None for v in e4[k][f"R5_L{L}__err_t"] + e4[k][f"R6_L{L}__err_t"]) for k in ks for L in Ls)
        md += [f"\n## {title} (n = {len(ks)}); anchor's own error {anc:.4f}; non-finite repaired trajectories "
               f"(scored +inf): {n_bad}\n",
               "| L | R0 original | R1 replace (4b) | R5 chosen | R6 chosen + anchor | R1 - R6 [95% CI] | "
               "frames >= L: R0 | R5 | R0 - R5 [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
        for L in Ls:
            R5 = [m_all(e4[k], f"R5_L{L}") for k in ks]; R6 = [m_all(e4[k], f"R6_L{L}") for k in ks]
            R0x = [m_from(b[k], "R0", L) for k in ks]; R5x = [m_from(e4[k], f"R5_L{L}", L) for k in ks]
            d16, d05 = ci(R1, R6, nb), ci(R0x, R5x, nb)
            md.append(f"| {L} | {np.mean(R0):.4f} | {np.mean(R1):.4f} | {np.mean(R5):.4f} | {np.mean(R6):.4f} | {f(d16)} | "
                      f"{np.mean(R0x):.4f} | {np.mean(R5x):.4f} | {f(d05)} |")
            if title.endswith("(PRIMARY)") and L == cfg["primary_L"]:
                verdict = bool(np.isfinite(d16[1]) and d16[1] > 0)
        ch = Counter((b[k]["split"], e4[k][f"L{cfg['primary_L']}__chosen"]["dlogE"],
                      e4[k][f"L{cfg['primary_L']}__chosen"]["force_scale"]) for k in ks)
        by = {}
        for (sp, d, s), n in ch.items():
            by.setdefault(sp, []).append((n, d, s))
        md.append(f"\nMost-chosen adjustments (L = {cfg['primary_L']}): " + "; ".join(
            f"{sp}: " + ", ".join(f"{n}: ({d:+g}, x{s:g})" for n, d, s in sorted(v, reverse=True)[:3])
            for sp, v in sorted(by.items())))
    head = [f"- **Primary** (L = {cfg['primary_L']}): anchored repair beats replacement on rejected base trajectories: "
            f"{'PASS' if verdict else 'FAIL' if verdict is not None else 'n/a'}", ""]
    if syn:
        head.insert(0, "> SYNTHETIC INPUT. These numbers test the pipeline only and mean nothing.\n")
    run.path("SUMMARY.md").write_text("\n".join(md[:2] + head + md[2:]) + "\n")
    print(f"[p4e_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
