"""Pilot 8b: a fused certificate score (EXPLORATORY, development data). CPU only, existing data.

Defined Oct 17, 2026 in PLAN.md "Pilots 8a and 8b" before any pilot code ran.
Signals per query (no extra simulation): the three placed coarse window defects (sorted) and their max, the
momentum residual (7a), seed spread, and the 5a generation statistics. The calibration pool is split by object
into thirds (seed 0): fit (logistic regression on standardised features, bad = error > eps), calibrate (single-
draw object rule at alpha), evaluate. The window-only certificate is recalibrated on the same third.
Also evaluated on the 4 OOD splits of Phase 2c (false-accept rate among bad queries under shift).
Promising if: on the evaluation third, good-trajectory acceptance rises by >= 0.10 over window-only with FAR <= 0.10,
and the 2c-shift FAR rises by <= 0.02.

Output: results/p8b_fused/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p4a_certificate import ood_index_from, threshold  # noqa: E402
from p5a_amortise import features, light_query  # noqa: E402
from p7a_metrics import com_prediction, momentum_inputs, residual  # noqa: E402


def load(pool, cfg):
    spec, root, groot = cfg["defect_spec"], Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    oi = ood_index_from(pool["build_runs"])
    errs = {r["qid"]: r for rn in pool["probe_runs"] for f in sorted(results_dir(rn).glob("shard_*/probes.jsonl"))
            for r in read_jsonl(f) if r["model"] == "base"}
    rows = []
    for rn in pool["scan_runs"]:
        for f in sorted(results_dir(rn).glob("shard_*/scan.jsonl")):
            for r in read_jsonl(f):
                if r["source"] != "base" or r["fidelity"] != spec["fidelity"] or int(r["m"]) != spec["m"]:
                    continue
                if r["qid"] not in errs or (pool.get("splits") and r["split"] not in pool["splits"]):
                    continue
                path = root / "testset" / f"{r['qid']}.h5" if r["split"] == "test_id" else Path(oi[r["qid"]])
                q = {**light_query(path), "qid": r["qid"], "split": r["split"]}
                s = np.load(groot / "base" / r["split"] / f"{r['qid']}.npz")["samples"]
                w = {st: (np.inf if (bl or cr or v is None) else v) for st, v, bl, cr in
                     zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
                win = sorted(w[t] for t in starts_for(spec["placement"], s, q, spec["m"]))
                chat, wt = com_prediction(momentum_inputs(path))
                rows.append({"qid": r["qid"], "obj": r["qid"].split("__")[0], "split": r["split"],
                             "E": float(errs[r["qid"]]["err_mean"]), "S": win[-1], "win": win,
                             "R": residual(s[0].astype(np.float64), chat, wt),
                             "SS": float(errs[r["qid"]]["seed_spread"]), "gen": features(q, s[0])})
    return rows


def feat(r):
    lg = lambda v: float(np.log10(np.clip(v, 1e-5, 10.0)))                                   # noqa: E731  (inf -> 1)
    return [lg(v) for v in r["win"]] + [lg(r["S"]), lg(r["R"]), lg(r["SS"])] + list(r["gen"])


def object_threshold(score, bad, obj, alpha, seed=0):
    """Single-draw object rule: one bad draw per object, uniformly; the floor(alpha (n+1))-th smallest."""
    rng = np.random.default_rng(seed)
    picks = [score[rng.choice(np.nonzero(bad & (obj == o))[0])] for o in sorted(set(obj[bad]))]
    return threshold(np.array(picks, float), alpha), len(picks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, alpha = cfg["eps"], cfg["alpha"]
    cal = load(cfg["calibration_pool"], cfg)
    ood = load(cfg["shift_pool"], cfg)
    O = np.array([r["obj"] for r in cal])
    objs = np.array(sorted(set(O)))
    perm = np.random.default_rng(0).permutation(objs)
    third = {o: i * 3 // len(perm) for i, o in enumerate(perm)}                 # 0 fit, 1 calibrate, 2 evaluate
    part = np.array([third[o] for o in O])
    X = np.array([feat(r) for r in cal]); y = np.array([r["E"] > eps for r in cal])
    S = np.array([r["S"] for r in cal]); S = np.where(np.isfinite(S), S, 1e9)
    sc = StandardScaler().fit(X[part == 0])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[part == 0]), y[part == 0])
    F = clf.predict_proba(sc.transform(X))[:, 1]
    Xo = np.array([feat(r) for r in ood]); yo = np.array([r["E"] > eps for r in ood])
    So = np.array([r["S"] for r in ood]); So = np.where(np.isfinite(So), So, 1e9)
    Fo = clf.predict_proba(sc.transform(Xo))[:, 1]
    res = {}
    for nm, sc_cal, sc_ood in (("window only", S, So), ("fused", F, Fo)):
        c = part == 1
        lam, n_bad = object_threshold(sc_cal[c], y[c], O[c], alpha)
        ev = part == 2
        acc_ev, acc_o = sc_cal[ev] < lam, sc_ood < lam
        res[nm] = {"lambda": float(lam), "n_bad_cal_objects": n_bad,
                   "power_eval": float(acc_ev[~y[ev]].mean()), "far_eval": float(acc_ev[y[ev]].mean()) if y[ev].any() else np.nan,
                   "reject_eval": float(1 - acc_ev.mean()), "far_shift": float(acc_o[yo].mean()) if yo.any() else np.nan,
                   "power_shift": float(acc_o[~yo].mean()) if (~yo).any() else np.nan}
    d_pow = res["fused"]["power_eval"] - res["window only"]["power_eval"]
    d_far = res["fused"]["far_shift"] - res["window only"]["far_shift"]
    promising = d_pow >= 0.10 and res["fused"]["far_eval"] <= 0.10 and d_far <= 0.02
    names = [f"win{i}" for i in range(3)] + ["max window", "residual", "seed spread", "log10 E", "nu", "force coeff",
             "drag fraction", "gravity", "floor gap", "centroid disp", "max disp", "mean speed", "max speed",
             "max centroid acc", "bbox ratio"]
    coef = sorted(zip(names, clf.coef_[0]), key=lambda t: -abs(t[1]))[:6]
    md = ["# Pilot 8b: a fused certificate score (EXPLORATORY, development data; defined Oct 17, 2026)\n",
          f"Calibration pool {len(cal)} queries ({len(objs)} objects) split into thirds by object (fit / calibrate / "
          f"evaluate); 2c shift splits {len(ood)} queries. eps = {eps}, alpha = {alpha}.\n",
          "| score | threshold | bad objects (cal) | good accepted (eval) | false accept (eval) | rejected (eval) | false accept (2c shift) | good accepted (2c shift) |",
          "|---|---|---|---|---|---|---|---|"]
    for nm, v in res.items():
        md.append(f"| {nm} | {v['lambda']:.4g} | {v['n_bad_cal_objects']} | {v['power_eval']:.3f} | {v['far_eval']:.3f} | "
                  f"{v['reject_eval']:.3f} | {v['far_shift']:.3f} | {v['power_shift']:.3f} |")
    md += ["", f"Good-acceptance gain {d_pow:+.3f}; shift false-accept change {d_far:+.3f}. "
           f"**Promising: {'YES' if promising else 'NO'}** (rule: gain >= 0.10, eval FAR <= 0.10, shift FAR change <= +0.02).",
           "", "Largest standardised coefficients: " + ", ".join(f"{n} {c:+.2f}" for n, c in coef),
           "", "Exploratory pilot on development data; one object split; nothing here is a confirmation."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"results": res, "gain": d_pow, "far_shift_change": d_far, "promising": bool(promising)}, run.path("summary.json"))
    print(f"[p8b] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
