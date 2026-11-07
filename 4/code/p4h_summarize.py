"""Phase 4h verdict: certified-prefix repair vs certify-or-simulate. CPU only.

Pre-registered Oct 13, 2026 (PLAN.md "Phase 4h"). On the Phase 2c base queries, both pipelines
use the same lambda. Costs are A100 seconds per query (Phase 1 generation and reference rollout,
scan seconds for probes, measured suffix rollouts).
  suffix   generation + chain windows up to the first failure + (suffix rollout | reference
           rollout if the anchor fails | nothing if all pass)
  cert     generation + the certificate's 3 windows + reference rollout if rejected
PASS if (1) the cost saving cert - suffix has 95% paired-bootstrap lower bound > 0, and
        (2) risk P(err > eps) of suffix <= risk of cert + 0.02.

Output: results/p4h_summary/SUMMARY.md
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bootstrap_ci, load_config, read_jsonl, results_dir, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p2c_summarize import query_inputs  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4d_pipeline import gen_costs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p4h_summary"), cfg, synthetic=False)
    gcost, c_ref = gen_costs()
    # Suffix cost (Oct 13, before any result, so the job can run on any GPU): its share of the
    # measured A100 reference rollout, c_ref x substeps / full_substeps, the same scale as a
    # full fallback. 'measured' uses the job's own seconds (A100 runs only).
    if cfg.get("suffix_cost") == "ref_fraction":
        suffix_cost = lambda r: c_ref * r["suffix_substeps"] / float(cfg["full_substeps"])
        cost_note = f"share of the reference rollout (c_ref x substeps / {cfg['full_substeps']})"
    else:
        suffix_cost = lambda r: r.get("suffix_seconds") or 0.0
        cost_note = "measured seconds"
    cg, tau, eps, spec = gcost["base"]["best"], float(cfg["tau"]), cfg["eps"], cfg["cert_spec"]
    rows = [r for f in sorted(results_dir(cfg.get("run_name", "p4h_suffix")).glob("shard_*/suffix.jsonl"))
            for r in read_jsonl(f) if not r.get("crashed")]
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == spec["fidelity"] and int(r["m"]) == spec["m"]}
    oi = ood_index_from(["p2_build_ood"])
    root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    S = {k: [] for k in ("c_suf", "c_cert", "e_suf", "e_cert", "split", "mode", "j", "floor")}
    for r in rows:
        sr = scan[r["qid"]]
        q = query_inputs(r["qid"], r["split"], root, oi)
        samples = np.load(groot / "base" / r["split"] / f"{r['qid']}.npz")["samples"]
        st = starts_for(spec["placement"], samples, q, spec["m"])
        win = {s: (np.inf if (b or c or raw is None) else raw) for s, raw, b, c in zip(sr["starts"], sr["raw"], sr["blown"], sr["crashed"])}
        secs = dict(zip(sr["starts"], sr["seconds"]))
        rej = max(win[s] for s in st) >= tau
        e_gen = float(np.mean(r["gen_err_t"]))
        S["c_cert"].append(cg + sum(secs[s] or 0.0 for s in st) + (c_ref if rej else 0.0))
        S["e_cert"].append(0.0 if rej else e_gen)
        extra = {"accept": 0.0, "full": c_ref}.get(r["mode"]) if r["mode"] != "suffix" else suffix_cost(r)
        S["c_suf"].append(cg + r["probe_seconds"] + extra)
        S["e_suf"].append(float(np.mean(r["err_t"])))
        S["split"].append(r["split"]); S["mode"].append(r["mode"]); S["j"].append(r["j"])
        S["floor"].append(float(np.mean(r["floor_err_t"])) if r["mode"] == "suffix" else np.nan)
    A = {k: np.array(v, dtype=object if k in ("split", "mode", "j") else float) for k, v in S.items()}
    save = bootstrap_ci(lambda a, b: float(np.mean(a - b)), [A["c_cert"], A["c_suf"]], n_boot=cfg["n_boot"])
    risk_s, risk_c = float(np.mean(A["e_suf"] > eps)), float(np.mean(A["e_cert"] > eps))
    p1, p2 = save[1] > 0, risk_s <= risk_c + 0.02
    md = ["# Phase 4h: certified-prefix repair (pre-registered Oct 13, 2026)\n",
          f"{len(rows)} base queries (Phase 2c). tau = lambda = {tau}. Costs in A100 s/query; reference rollout {c_ref:.3f} s; "
          f"generation {cg:.3f} s. Suffix cost: {cost_note}.\n",
          f"- **(1) cost saving** cert - suffix: {save[0]:+.3f} [{save[1]:+.3f}, {save[2]:+.3f}] s/query: **{'PASS' if p1 else 'FAIL'}**",
          f"- **(2) risk** suffix {risk_s:.3f} vs cert {risk_c:.3f} (limit +0.02): **{'PASS' if p2 else 'FAIL'}**",
          f"- **Verdict: {'PASS' if (p1 and p2) else 'FAIL'}**", "",
          "| pipeline | mean error | risk | cost (s/query) | cost / always-reference |", "|---|---|---|---|---|",
          f"| certify-or-simulate | {A['e_cert'].mean():.4f} | {risk_c:.3f} | {A['c_cert'].mean():.3f} | {A['c_cert'].mean() / c_ref:.2f} |",
          f"| **certified-prefix repair** | {A['e_suf'].mean():.4f} | {risk_s:.3f} | {A['c_suf'].mean():.3f} | {A['c_suf'].mean() / c_ref:.2f} |",
          f"| always reference | 0 | 0 | {c_ref:.3f} | 1.00 |",
          "", "Modes: " + ", ".join(f"{k} {v}" for k, v in Counter(A["mode"]).items()),
          "First failing window (0 = anchor, 1..5 = chain windows at k = 3, 7, 11, 15, 19): "
          + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(x for x in A["j"] if x is not None).items())),
          f"Suffix lift floor (restart from ground-truth positions with F = I): mean error {np.nanmean(A['floor']):.4f} "
          f"vs suffix repair {np.mean([e for e, m in zip(A['e_suf'], A['mode']) if m == 'suffix']):.4f} on the same queries",
          "", "| split | n | error suffix | error cert | risk suffix | risk cert | cost suffix | cost cert |", "|---|---|---|---|---|---|---|---|"]
    for sp in sorted(set(A["split"])):
        k = A["split"] == sp
        md.append(f"| {sp} | {k.sum()} | {A['e_suf'][k].mean():.4f} | {A['e_cert'][k].mean():.4f} | {np.mean(A['e_suf'][k] > eps):.2f} | "
                  f"{np.mean(A['e_cert'][k] > eps):.2f} | {A['c_suf'][k].mean():.2f} | {A['c_cert'][k].mean():.2f} |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    print(f"[p4h_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
