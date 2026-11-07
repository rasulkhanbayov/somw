"""Phase 4a: conformal certificate on the held-out calibration set. CPU only.

Pre-registered Oct 11, 2026 in PLAN.md ("Phase 4a"), before any analysis of the calibration
data. The signals were chosen on the Phase 2c selection set by a rule fixed in advance: the
cheapest configuration within 0.01 AUROC of the best.

Certificate: class-conditional split conformal on BAD trajectories (err_mean > eps). The
calibration set (100 objects x 4 in-distribution physics draws) is split 50/50 by object.
lambda = the k-th smallest score among bad calibration trajectories, k = floor(alpha (n_bad + 1));
accept if score < lambda. Then P(accept | bad) <= alpha under exchangeability.
  P1 validity: false-accept rate among bad test-half trajectories; PASS if its Clopper-Pearson
     95% interval reaches down to alpha or below.
  P2 power (base): acceptance rate among good test-half trajectories, defect certificate minus
     seed-spread certificate, paired bootstrap over objects; PASS if the 95% lower bound > 0.
  P3 shift (empirical, no guarantee claimed): the same lambda on the Phase 2c splits.

Output: results/p4a_certificate/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p2c_summarize import load_scan, query_inputs  # noqa: E402


def ood_index_from(runs):
    idx = {}
    for run in runs:
        for f in sorted(results_dir(run).glob("shard_*/ood_manifest.json")):
            idx.update({e["qid"]: e["path"] for e in json.load(open(f))})
    return idx


def collect(scan_run: str, probe_run: str, models, specs, cfg, ood_index) -> list[dict]:
    """One row per (qid, model): err, seed spread, and each spec's defect."""
    scan, split = load_scan(scan_run, False)
    probes = {(r["qid"], r["model"]): r
              for f in sorted(results_dir(probe_run).glob("shard_*/probes.jsonl")) for r in read_jsonl(f)}
    data_root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    rows = []
    for qid in sorted({k[0] for k in scan}):
        sp = split[qid]
        q = None
        for model in models:
            pr = probes.get((qid, model))
            if pr is None:
                continue
            q = q or query_inputs(qid, sp, data_root, ood_index)
            samples = np.load(groot / model / sp / f"{qid}.npz")["samples"]
            row = {"qid": qid, "obj": qid.split("__")[0], "split": sp, "model": model,
                   "err": pr["err_mean"], "seed_spread": pr["seed_spread"]}
            for name, s in specs.items():
                win = scan.get((qid, model, s["fidelity"], s["m"]))
                if win is not None:
                    row[name] = max(win[st] for st in starts_for(s["placement"], samples, q, s["m"]))
            rows.append(row)
    return rows


def ecdf(ref: np.ndarray):
    ref = np.sort(ref[~np.isnan(ref)])
    return lambda x: np.searchsorted(ref, np.asarray(x, float), side="right") / len(ref)


def threshold(bad_scores: np.ndarray, alpha: float) -> float:
    s = np.sort(bad_scores[~np.isnan(bad_scores)])
    k = int(np.floor(alpha * (len(s) + 1)))
    return -np.inf if k < 1 else float(s[k - 1])


def clopper_pearson(x: int, n: int, a: float = 0.05):
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("run_name", "p4a_certificate"), cfg, synthetic=False)
    models = cfg["models"]
    specs = cfg["defect_specs"]
    ood_index = ood_index_from(["p2_build_ood", cfg["calib_build_run"]])
    cal_rows = collect(cfg["calib_scan_run"], cfg["calib_probe_run"], models, specs, cfg, ood_index)
    shift_rows = collect(cfg["shift_scan_run"], cfg["shift_probe_run"], models, specs, cfg, ood_index)

    objs = sorted({r["obj"] for r in cal_rows})
    rng = np.random.default_rng(cfg["split_seed"])
    cal_objs = set(rng.permutation(objs)[: len(objs) // 2])
    md = ["# Phase 4a: conformal certificate (pre-registered Oct 11, 2026)\n",
          f"Calibration set: {len(cal_rows)} trajectories; {len(objs)} objects split 50/50 by object "
          f"(seed {cfg['split_seed']}). Accept if score < lambda; guarantee P(accept | bad) <= alpha in distribution.\n"]
    out = {}

    for cert in cfg["certificates"]:
        model, name = cert["model"], cert["name"]
        rows = [r for r in cal_rows if r["model"] == model]
        cal = [r for r in rows if r["obj"] in cal_objs]
        tst = [r for r in rows if r["obj"] not in cal_objs]

        def score_fn(rs, kind=cert["score"], cal_ref=cal):
            if kind == "seed_spread":
                return np.array([r["seed_spread"] for r in rs], float)
            if kind.startswith("fused:"):
                d = kind.split(":", 1)[1]
                fs = ecdf(np.array([r["seed_spread"] for r in cal_ref], float))
                fd = ecdf(np.array([r[d] for r in cal_ref], float))
                return (fs([r["seed_spread"] for r in rs]) + fd([r[d] for r in rs])) / 2.0
            return np.array([r[kind] for r in rs], float)

        md += [f"\n## {name} ({model}; score = {cert['score']})\n",
               "| eps | alpha | n bad (cal) | lambda | test bad | false-accept rate [95% CP] | P1 | "
               "test good | good accepted | seed-spread cert: good accepted | defect - seed spread [95% CI] |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for eps in cfg["eps"]:
            for alpha in cfg["alpha"]:
                sc_cal, sc_tst = score_fn(cal), score_fn(tst)
                e_cal = np.array([r["err"] for r in cal]); e_tst = np.array([r["err"] for r in tst])
                lam = threshold(sc_cal[e_cal > eps], alpha)
                acc = sc_tst < lam
                bad, good = e_tst > eps, e_tst <= eps
                far = acc[bad].mean() if bad.any() else np.nan
                lo, hi = clopper_pearson(int(acc[bad].sum()), int(bad.sum()))
                p1 = bool(bad.any() and lo <= alpha)
                # Reference: the same procedure with seed spread as the score.
                ss_cal = np.array([r["seed_spread"] for r in cal]); ss_tst = np.array([r["seed_spread"] for r in tst])
                lam_ss = threshold(ss_cal[e_cal > eps], alpha)
                acc_ss = ss_tst < lam_ss
                objs_t = np.array([r["obj"] for r in tst]); uo = np.unique(objs_t)
                diffs = []
                brng = np.random.default_rng(0)
                for _ in range(cfg["n_boot"]):
                    pick = np.concatenate([np.nonzero(objs_t == o)[0] for o in brng.choice(uo, len(uo))])
                    g = good[pick]
                    if g.any():
                        diffs.append(acc[pick][g].mean() - acc_ss[pick][g].mean())
                d0 = (acc[good].mean() - acc_ss[good].mean()) if good.any() else np.nan
                dlo, dhi = (np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)) if diffs else (np.nan, np.nan)
                key = f"{name}|eps={eps}|alpha={alpha}"
                out[key] = {"lambda": lam, "far": far, "far_ci": [lo, hi], "P1": p1,
                            "good_accept": float(acc[good].mean()) if good.any() else None,
                            "good_accept_seed_spread": float(acc_ss[good].mean()) if good.any() else None,
                            "P2_diff": [d0, dlo, dhi], "P2": bool(dlo > 0)}
                md.append(f"| {eps} | {alpha} | {int((e_cal > eps).sum())} | {lam:.4g} | {int(bad.sum())} | "
                          f"{far:.3f} [{lo:.3f},{hi:.3f}] | {'PASS' if p1 else 'FAIL'} | {int(good.sum())} | "
                          f"{acc[good].mean() if good.any() else float('nan'):.3f} | "
                          f"{acc_ss[good].mean() if good.any() else float('nan'):.3f} | "
                          f"{d0:+.3f} [{dlo:+.3f},{dhi:+.3f}] |")

        # P3: the primary in-distribution lambda applied to the Phase 2c splits.
        eps, alpha = cfg["primary_eps"], cfg["primary_alpha"]
        lam = out[f"{name}|eps={eps}|alpha={alpha}"]["lambda"]
        md += [f"\nP3 (shift, empirical) for eps = {eps}, alpha = {alpha}, lambda = {lam:.4g}:\n",
               "| split | n | bad | false-accept rate among bad | good | acceptance among good |", "|---|---|---|---|---|---|"]
        srows = [r for r in shift_rows if r["model"] == model]
        for sp in sorted({r["split"] for r in srows}):
            rs = [r for r in srows if r["split"] == sp]
            sc = score_fn(rs); e = np.array([r["err"] for r in rs])
            acc = sc < lam
            b, g = e > eps, e <= eps
            md.append(f"| {sp} | {len(rs)} | {int(b.sum())} | {acc[b].mean() if b.any() else float('nan'):.3f} | "
                      f"{int(g.sum())} | {acc[g].mean() if g.any() else float('nan'):.3f} |")
            out[f"{name}|P3|{sp}"] = {"far": float(acc[b].mean()) if b.any() else None,
                                      "good_accept": float(acc[g].mean()) if g.any() else None}

    prim = cfg["primary"]
    pk = f"{prim}|eps={cfg['primary_eps']}|alpha={cfg['primary_alpha']}"
    head = [f"- **Primary ({pk})**: P1 validity {'PASS' if out[pk]['P1'] else 'FAIL'}; "
            f"P2 power vs seed spread {'PASS' if out[pk]['P2'] else 'FAIL'}", ""]
    run.path("SUMMARY.md").write_text("\n".join(md[:2] + head + md[2:]) + "\n")
    save_json(out, run.path("summary.json"))
    print(f"[p4a_certificate] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
