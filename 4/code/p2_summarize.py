"""Phase 2 summary and GO/NO-GO verdicts.

Reads results/p2a_lift_floor/shard_*/lift_floor.jsonl and
results/p2b_probe/shard_*/probes.jsonl, writes results/p2_summary/SUMMARY.md
and summary.json.

Gates (thresholds in configs/p2_summary.yaml):
  G2a lift floor   median raw defect of `lift` on in-distribution ground truth
                   <= floor_ratio_max x median raw defect on generated
                   in-distribution trajectories (same window length m), for the
                   primary model. Otherwise the verifier cannot separate model
                   error from its own lifting error.
  G2b verifier     some non-oracle probe signal has (i) pooled Spearman with
                   err_mean >= rho_min, (ii) a paired-bootstrap advantage over
                   seed_spread whose 95% CI excludes 0 on the pooled OOD
                   queries, and (iii) median simulator cost <= cost_max of a
                   full rollout (raw signals are charged without perturbation
                   runs; normalised ones with them).
Unit of resampling: the query.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (bootstrap_ci, load_config, paired_bootstrap_diff, read_jsonl,  # noqa: E402
                    results_dir, save_json, spearman, start_run)


def load_rows(name: str, fname: str, allow_syn: bool) -> list[dict]:
    rows = []
    for f in glob.glob(str(results_dir(name, allow_syn) / "shard_*" / fname)):
        rows.extend(read_jsonl(f))
    return rows


INF = float("inf")


def num(x) -> bool:
    """Usable value: present and not NaN. +inf (a scored blow-up) is kept."""
    return x is not None and not np.isnan(x)


def med(xs):
    xs = np.asarray([x for x in xs if num(x)], dtype=float)
    return float(np.median(xs)) if len(xs) else float("nan")


def fmt(x, nd=3):
    if x is None or np.isnan(x):
        return "nan"
    return ("+inf" if x > 0 else "-inf") if np.isinf(x) else f"{x:.{nd}f}"


def apply_blowup_rule(lf: list[dict], pr: list[dict], strategies: list[str]) -> None:
    """Pre-registered Oct 10, 2026, before any Phase 2 result existed: a window whose
    rollout blows up (CUDA crash, non-finite, or particles outside the grid) is the worst
    possible defect, an automatic reject. A blown main rollout scores raw = norm = +inf; a
    blown perturbed rollout scores norm = +inf. Excluding blow-ups instead is reported as
    a secondary column only."""
    for r in lf:
        if r.get("crashed") or r.get("blown"):
            r["raw"] = r["raw_p90"] = r["norm"] = INF
            r["_blowup"] = True
        elif r.get("blown_perturb"):
            r["norm"] = INF
            r["_blowup"] = True
    for r in pr:
        for s in strategies:
            if r.get(f"{s}__crashed") or r.get(f"{s}__blown"):
                for k in ("raw_max", "raw_mean", "norm_max", "norm_mean"):
                    r[f"{s}__{k}"] = INF
                r[f"{s}__blowup"] = True
            elif r.get(f"{s}__blown_perturb"):
                r[f"{s}__norm_max"] = r[f"{s}__norm_mean"] = INF
                r[f"{s}__blowup"] = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.allow_synthetic)
    run = start_run(cfg.get("run_name", "p2_summary"), cfg, synthetic=args.allow_synthetic)
    md = []
    m_win = cfg["m"]
    primary = cfg["primary_model"]

    # ---- A. lift floor -----------------------------------------------------------
    lf = load_rows("p2a_lift_floor", "lift_floor.jsonl", args.allow_synthetic)
    pr = load_rows(cfg.get("probe_run", "p2b_probe"), "probes.jsonl", args.allow_synthetic)
    strategies = cfg["strategies"]
    apply_blowup_rule(lf, pr, strategies)
    md.append("## A. Lift floor (defect on ground-truth trajectories)\n")
    md.append("| split | variant | m | median raw | median p90 | median sigma | median norm | v_rel | F_fro | C_fro (true C_fro) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in lf:
        if "error" in r and not r.get("crashed"):
            continue
        sg = "id" if r["split"] == "test_id" else "ood"
        groups[(sg, r["variant"], r["m"])].append(r)
    for (sg, v, m), rs in sorted(groups.items()):
        md.append(f"| {sg} | {v} | {m} | {fmt(med([r.get('raw') for r in rs]), 4)} | "
                  f"{fmt(med([r.get('raw_p90') for r in rs]), 4)} | {fmt(med([r.get('sigma') for r in rs]), 4)} | "
                  f"{fmt(med([r.get('norm') for r in rs]), 2)} | {fmt(med([r.get('v_rel') for r in rs]))} | "
                  f"{fmt(med([r.get('F_fro_mean') for r in rs]))} | {fmt(med([r.get('C_fro_mean') for r in rs]))} "
                  f"({fmt(med([r.get('C_true_fro_mean') for r in rs]))}) |")
    floor_id = med([r.get("raw") for r in groups.get(("id", cfg.get("floor_variant", "lift"), m_win), [])])

    # ---- B/C. probes ---------------------------------------------------------------
    signals = ["seed_spread", "seed_spread_final", "motion"]
    for s in strategies:
        signals += [f"{s}__raw_max", f"{s}__norm_max", f"{s}__raw_mean", f"{s}__norm_mean"]

    # Windows whose rollout blew up hard enough to lose the CUDA context have no defect and
    # drop out of every statistic below. Report them, since a blow-up is itself evidence of
    # a bad trajectory and excluding it can bias the rank correlations.
    lf_blow = sum(1 for r in lf if r.get("_blowup"))
    lf_crash = sum(1 for r in lf if r.get("crashed"))
    pr_blow = {s: sum(1 for r in pr if r.get(f"{s}__blowup")) for s in strategies}
    md.append(f"\nBlow-ups scored as defect = +inf (pre-registered rule; CUDA crashes included): "
              f"lift floor {lf_blow} of {len(lf)} rows ({lf_crash} CUDA crashes); probes "
              + ", ".join(f"{s} {n}" for s, n in pr_blow.items()) + f" of {len(pr)} rows.")

    gen_def_id = med([r.get(f"{cfg['floor_reference_strategy']}__raw_max") for r in pr
                      if r["model"] == primary and r["split"] == "test_id"])
    md.append(f"\nMedian raw defect on generated in-distribution trajectories ({primary}, "
              f"{cfg['floor_reference_strategy']} placement): {fmt(gen_def_id, 4)}; lift floor ({cfg.get('floor_variant', 'lift')}, m={m_win}): "
              f"{fmt(floor_id, 4)}; ratio floor/generated = {fmt(floor_id / gen_def_id if gen_def_id else float('nan'))}.")

    md.append("\n## B. True error by split (sample 0)\n")
    md.append("| model | split | n | median err_mean | median err_final |\n|---|---|---|---|---|")
    by_ms = defaultdict(list)
    for r in pr:
        by_ms[(r["model"], r["split"])].append(r)
    for (m, s), rs in sorted(by_ms.items()):
        md.append(f"| {m} | {s} | {len(rs)} | {fmt(med([r['err_mean'] for r in rs]), 4)} | "
                  f"{fmt(med([r['err_final'] for r in rs]), 4)} |")

    md.append("\n## C. Does the signal rank trajectories by true error? (Spearman with err_mean, 95% bootstrap CI)\n")
    verdict_rows = []
    for model in sorted({r["model"] for r in pr}):
        rows_m = [r for r in pr if r["model"] == model]
        pools = {"id": [r for r in rows_m if r["split"] == "test_id"],
                 "ood": [r for r in rows_m if r["split"] != "test_id"], "all": rows_m}
        md.append(f"\n### {model}\n")
        md.append("| signal | rho id | rho ood | rho all | adv. over seed_spread (ood) | median cost frac "
                  "| rho all, blow-ups excluded (secondary) |")
        md.append("|---|---|---|---|---|---|---|")
        for sig in signals:
            cells, rho_all, adv = [], float("nan"), (float("nan"),) * 3
            for pname in ("id", "ood", "all"):
                rs = [r for r in pools[pname] if num(r.get(sig))]
                if len(rs) < 5:
                    cells.append("n<5")
                    continue
                a = np.array([r[sig] for r in rs])
                b = np.array([r["err_mean"] for r in rs])
                p, lo, hi = bootstrap_ci(spearman, [a, b], n_boot=cfg["n_boot"])
                cells.append(f"{p:.2f} [{lo:.2f},{hi:.2f}]")
                if pname == "all":
                    rho_all = p
            rs = [r for r in pools["ood"] if num(r.get(sig)) and num(r.get("seed_spread"))]
            if len(rs) >= 5 and sig != "seed_spread":
                a = np.array([r[sig] for r in rs])
                s0 = np.array([r["seed_spread"] for r in rs])
                e = np.array([r["err_mean"] for r in rs])
                adv = paired_bootstrap_diff(spearman, [a, e], [s0, e], n_boot=cfg["n_boot"])
            strat = sig.split("__")[0] if "__" in sig else None
            cost = float("nan")
            if strat:
                key = f"{strat}__cost_frac" if "norm" in sig else f"{strat}__cost_frac_raw_only"
                cost = med([r.get(key) for r in rows_m])
            adv_txt = "-" if not np.isfinite(adv[0]) else f"{adv[0]:+.2f} [{adv[1]:+.2f},{adv[2]:+.2f}]"
            fin = [r for r in pools["all"] if num(r.get(sig)) and np.isfinite(r[sig])]
            rho_excl = (f"{spearman(np.array([r[sig] for r in fin]), np.array([r['err_mean'] for r in fin])):.2f}"
                        if len(fin) >= 5 else "n<5")
            md.append(f"| {sig} | {' | '.join(cells)} | {adv_txt} | {fmt(cost, 3)} | {rho_excl} |")
            if strat and strat != "oracle":
                verdict_rows.append({"model": model, "signal": sig, "rho_all": rho_all,
                                     "adv": adv, "cost": cost})

    # ---- D. verdict -----------------------------------------------------------------
    g2a = bool(np.isfinite(floor_id) and num(gen_def_id) and gen_def_id > 0
               and floor_id <= cfg["floor_ratio_max"] * gen_def_id)
    passing = [v for v in verdict_rows
               if np.isfinite(v["rho_all"]) and v["rho_all"] >= cfg["rho_min"]
               and np.isfinite(v["adv"][1]) and v["adv"][1] > 0
               and np.isfinite(v["cost"]) and v["cost"] <= cfg["cost_max"]]
    passing_nocost = [v for v in verdict_rows
                      if np.isfinite(v["rho_all"]) and v["rho_all"] >= cfg["rho_min"]
                      and np.isfinite(v["adv"][1]) and v["adv"][1] > 0]
    g2b = bool(passing)
    if g2a and g2b:
        verdict = "GO"
    elif g2a and passing_nocost:
        verdict = "GO-CERTIFICATE-ONLY (signal works but costs more than cost_max)"
    else:
        verdict = "NO-GO"
    head = [f"# Phase 2 summary: **{verdict}**\n",
            f"- G2a lift floor: {'PASS' if g2a else 'FAIL'} (floor {fmt(floor_id, 4)} vs generated "
            f"{fmt(gen_def_id, 4)}, max ratio {cfg['floor_ratio_max']})",
            f"- G2b verifier: {'PASS' if g2b else 'FAIL'}; passing signals: "
            + (", ".join(f"{v['model']}:{v['signal']} (rho {v['rho_all']:.2f}, cost {v['cost']:.2f})"
                         for v in passing) or "none"), ""]
    if args.allow_synthetic:
        head.insert(0, "> SYNTHETIC INPUT. These numbers test the pipeline only and mean nothing.\n")
    md = head + md
    run.path("SUMMARY.md").write_text("\n".join(md))
    save_json({"verdict": verdict, "g2a": g2a, "g2b": g2b, "floor_id": floor_id,
               "gen_defect_id": gen_def_id, "passing": passing, "candidates": verdict_rows},
              run.path("summary.json"))
    print("\n".join(head))
    print(f"[p2_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok", verdict=verdict)


if __name__ == "__main__":
    main()
