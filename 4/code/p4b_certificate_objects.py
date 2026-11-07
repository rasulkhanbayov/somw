"""Phase 4a, object-level certificate (pre-registered Oct 11, 2026, before the id_new_4..11 data
existed). CPU only.

Why: the 4 (now 12) physics draws of one object are correlated, so trajectory-level split
conformal (p4a_certificate.py) is not exactly valid for a new object. The guarantee targeted
here is P(accept | a bad draw of a NEW object) <= alpha.

Data: the held-out test objects x 12 in-distribution draws (p4_calib_* and p4_calib2_*).
Signals: frozen in PLAN.md "Phase 4a" (chosen on the Phase 2c selection set).
Procedure, repeated over R random 50/50 object splits (seed r = 0..R-1):
  primary   single-draw subsampling: one bad draw chosen uniformly per calibration object;
            lambda = k-th smallest of those scores, k = floor(alpha (n_obj_bad + 1)). Exact.
  secondary object-weighted pooling: every bad draw, each object's draws weighted 1/n_j;
            lambda = the largest score whose cumulative weight <= alpha (n_obj_bad + 1).
            Approximately valid (two-level hierarchical conformal, CDF pooling).
  Test rates are object-weighted: each test object counts once, averaged over its draws.
Pre-registered checks (primary eps = 0.10, alpha = 0.10):
  P1' validity: mean false-accept rate over the R replicates (an estimate of the marginal
      rate) <= alpha + 0.01 for the primary procedure.
  P2' power (base): good-draw acceptance, defect certificate minus seed-spread certificate
      built the same way; PASS if the 2.5th percentile over replicates is > 0.
  P3' shift (empirical): lambda from replicate 0 applied to every Phase 2c split.

Output: results/p4b_certificate_objects/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, save_json, start_run  # noqa: E402
from p4a_certificate import collect, ecdf, ood_index_from, threshold  # noqa: E402


def pooled_threshold(scores: np.ndarray, objs: np.ndarray, alpha: float) -> float:
    uo, inv, cnt = np.unique(objs, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    order = np.argsort(scores, kind="mergesort")
    cum = np.cumsum(w[order])
    ok = np.nonzero(cum <= alpha * (len(uo) + 1))[0]
    return -np.inf if len(ok) == 0 else float(scores[order][ok[-1]])


def object_rate(accept: np.ndarray, objs: np.ndarray) -> float:
    """Mean over objects of the within-object acceptance rate (each object counts once)."""
    if len(objs) == 0:
        return float("nan")
    uo = np.unique(objs)
    return float(np.mean([accept[objs == o].mean() for o in uo]))


def make_scorer(kind: str):
    def score(rows, cal_rows):
        if kind == "seed_spread":
            return np.array([r["seed_spread"] for r in rows], float)
        if kind.startswith("fused:"):
            d = kind.split(":", 1)[1]
            fs = ecdf(np.array([r["seed_spread"] for r in cal_rows], float))
            fd = ecdf(np.array([r[d] for r in cal_rows], float))
            return (fs([r["seed_spread"] for r in rows]) + fd([r[d] for r in rows])) / 2.0
        return np.array([r[kind] for r in rows], float)
    return score


def run_cert(rows, scorer, eps, alpha, R, method):
    objs_all = np.array(sorted({r["obj"] for r in rows}))
    fars, goods, lams = [], [], []
    for rep in range(R):
        rng = np.random.default_rng(rep)
        cal_o = set(rng.permutation(objs_all)[: len(objs_all) // 2])
        cal = [r for r in rows if r["obj"] in cal_o]
        tst = [r for r in rows if r["obj"] not in cal_o]
        sc_c, sc_t = scorer(cal, cal), scorer(tst, cal)
        e_c, e_t = np.array([r["err"] for r in cal]), np.array([r["err"] for r in tst])
        o_c, o_t = np.array([r["obj"] for r in cal]), np.array([r["obj"] for r in tst])
        bad_c = e_c > eps
        if method == "subsample":
            picks = []
            for o in np.unique(o_c[bad_c]):
                idx = np.nonzero(bad_c & (o_c == o))[0]
                picks.append(sc_c[rng.choice(idx)])
            lam = threshold(np.array(picks, float), alpha)
        else:
            lam = pooled_threshold(sc_c[bad_c], o_c[bad_c], alpha)
        acc = sc_t < lam
        bad_t, good_t = e_t > eps, e_t <= eps
        fars.append(object_rate(acc[bad_t], o_t[bad_t]))
        goods.append(object_rate(acc[good_t], o_t[good_t]))
        lams.append(lam)
    return np.array(fars), np.array(goods), np.array(lams)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("run_name", "p4b_certificate_objects"), cfg, synthetic=False)
    models, specs, R = cfg["models"], cfg["defect_specs"], cfg["n_replicates"]
    ood_index = ood_index_from(["p2_build_ood"] + cfg["calib_build_runs"])
    rows = []
    for scan_run, probe_run in zip(cfg["calib_scan_runs"], cfg["calib_probe_runs"]):
        rows += collect(scan_run, probe_run, models, specs, cfg, ood_index)
    shift = collect(cfg["shift_scan_run"], cfg["shift_probe_run"], models, specs, cfg, ood_index)
    n_obj = len({r["obj"] for r in rows})
    md = ["# Phase 4a, object-level certificate (pre-registered Oct 11, 2026)\n",
          f"{len(rows)} calibration-pool trajectories over {n_obj} held-out objects; {R} random 50/50 object "
          "splits. Rates are object-weighted. Accept if score < lambda.\n"]
    out = {}
    for cert in cfg["certificates"]:
        model, name = cert["model"], cert["name"]
        rs = [r for r in rows if r["model"] == model]
        scorer = make_scorer(cert["score"])
        ss_scorer = make_scorer("seed_spread")
        md += [f"\n## {name} ({model}; score = {cert['score']})\n",
               "| eps | alpha | method | bad objects (pool) | mean FAR [5%, 95%] | good accepted (mean) | "
               "seed-spread cert: good accepted | defect - seed spread [2.5%, 97.5%] |",
               "|---|---|---|---|---|---|---|---|"]
        for eps in cfg["eps"]:
            n_bad_obj = len({r["obj"] for r in rs if r["err"] > eps})
            for alpha in cfg["alpha"]:
                for method in ("subsample", "pooled"):
                    far, good, lam = run_cert(rs, scorer, eps, alpha, R, method)
                    _f2, good_ss, _l2 = run_cert(rs, ss_scorer, eps, alpha, R, method)
                    diff = good - good_ss
                    key = f"{name}|eps={eps}|alpha={alpha}|{method}"
                    out[key] = {"far_mean": float(np.nanmean(far)), "far_q": [float(np.nanquantile(far, .05)),
                                float(np.nanquantile(far, .95))], "good_mean": float(np.nanmean(good)),
                                "good_ss_mean": float(np.nanmean(good_ss)),
                                "diff_q": [float(np.nanquantile(diff, .025)), float(np.nanquantile(diff, .975))]}
                    md.append(f"| {eps} | {alpha} | {method} | {n_bad_obj} | {np.nanmean(far):.3f} "
                              f"[{np.nanquantile(far, .05):.3f}, {np.nanquantile(far, .95):.3f}] | "
                              f"{np.nanmean(good):.3f} | {np.nanmean(good_ss):.3f} | {np.nanmean(diff):+.3f} "
                              f"[{np.nanquantile(diff, .025):+.3f}, {np.nanquantile(diff, .975):+.3f}] |")
        # P3': replicate-0 lambda (primary method) on the Phase 2c splits.
        eps, alpha = cfg["primary_eps"], cfg["primary_alpha"]
        _f, _g, lam = run_cert(rs, scorer, eps, alpha, 1, "subsample")
        rng = np.random.default_rng(0)
        cal_o = set(rng.permutation(np.array(sorted({r["obj"] for r in rs})))[: n_obj // 2])
        cal_rows = [r for r in rs if r["obj"] in cal_o]
        md += [f"\nP3' (shift, empirical): eps = {eps}, alpha = {alpha}, lambda = {lam[0]:.4g} (replicate 0)\n",
               "| split | bad | failures accepted | good | good accepted |", "|---|---|---|---|---|"]
        srs = [r for r in shift if r["model"] == model]
        for sp in sorted({r["split"] for r in srs}):
            ss = [r for r in srs if r["split"] == sp]
            sc = scorer(ss, cal_rows)
            e = np.array([r["err"] for r in ss])
            acc = sc < lam[0]
            b, g = e > eps, e <= eps
            md.append(f"| {sp} | {int(b.sum())} | {int(acc[b].sum())} | {int(g.sum())} | {int(acc[g].sum())} |")
            out[f"{name}|P3|{sp}"] = {"bad": int(b.sum()), "bad_accepted": int(acc[b].sum()),
                                      "good": int(g.sum()), "good_accepted": int(acc[g].sum())}

    pk = f"{cfg['primary']}|eps={cfg['primary_eps']}|alpha={cfg['primary_alpha']}|subsample"
    p1 = out[pk]["far_mean"] <= cfg["primary_alpha"] + 0.01
    p2 = out[pk]["diff_q"][0] > 0
    out["P1_prime"], out["P2_prime"] = bool(p1), bool(p2)
    head = [f"- **Primary ({pk})**: P1' validity {'PASS' if p1 else 'FAIL'} (mean FAR {out[pk]['far_mean']:.3f}, "
            f"target <= {cfg['primary_alpha'] + 0.01:.2f}); P2' power vs seed spread {'PASS' if p2 else 'FAIL'}", ""]
    run.path("SUMMARY.md").write_text("\n".join(md[:2] + head + md[2:]) + "\n")
    save_json(out, run.path("summary.json"))
    print(f"[p4b_certificate_objects] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
