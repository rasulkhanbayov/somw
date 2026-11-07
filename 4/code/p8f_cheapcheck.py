"""Pilot E2: cheaper verification (EXPLORATORY, development data). CPU only, existing scans.

Defined Oct 18, 2026 in PLAN.md "Pilots E1-E3" before any E-pilot code ran.
Variants of the window certificate (coarse_cfl): (a) the event window only; (b) event + the middle of the rest;
(c) baseline, three m = 4 spread_event windows; (d) three m = 2 spread_event windows (calibration rows exist for
id_new_0..3 only). Each threshold by the replicate-0 object rule (alpha = eps = 0.10) on the calibration pool;
evaluated on the 2c development pool. Promising if some variant cuts the probe cost by >= 50% vs (c), with pooled
AUROC >= (c) - 0.02 and shift false-accept rate <= 0.10.

Output: results/p8f_cheapcheck/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import spread_event, starts_for  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402
from p4m_confirm import calib_lambda  # noqa: E402
from p5a_amortise import light_query  # noqa: E402
from verifier import candidate_starts  # noqa: E402


def variant_starts(name, s, q):
    if name == "c":
        return starts_for("spread_event", s, q, 4), 4
    if name == "d":
        return starts_for("spread_event", s, q, 2), 2
    ev = spread_event(s, q, 4)[0]
    if name == "a":
        return [ev], 4
    rest = [2 * k + 1 for k in candidate_starts(4) if 2 * k + 1 != ev]
    return [ev, rest[int(round(0.5 * (len(rest) - 1)))]], 4


def load(pool, cfg, fits):
    root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    oi = ood_index_from(pool["build_runs"])
    errs = {r["qid"]: float(r["err_mean"]) for rn in pool["probe_runs"] for f in sorted(results_dir(rn).glob("shard_*/probes.jsonl"))
            for r in read_jsonl(f) if r["model"] == "base"}
    scan = {}
    for rn in pool["scan_runs"]:
        for f in sorted(results_dir(rn).glob("shard_*/scan.jsonl")):
            for r in read_jsonl(f):
                if r["source"] == "base" and r["fidelity"] == "coarse_cfl" and int(r["m"]) in (2, 4) and r["qid"] in errs:
                    if pool.get("splits") and r["split"] not in pool["splits"]:
                        continue
                    scan[(r["qid"], int(r["m"]))] = r
    rows = []
    for qid in sorted({q for q, _ in scan}):
        r4 = scan.get((qid, 4))
        if r4 is None:
            continue
        sp = r4["split"]
        path = root / "testset" / f"{qid}.h5" if sp == "test_id" else Path(oi[qid])
        q = {**light_query(path), "qid": qid, "split": sp}
        s = np.load(groot / "base" / sp / f"{qid}.npz")["samples"]
        row = {"qid": qid, "obj": qid.split("__")[0], "split": sp, "E": errs[qid]}
        for v in "abcd":
            st, m = variant_starts(v, s, q)
            r = scan.get((qid, m))
            if r is None:
                row[v] = None
                continue
            w = {t: (np.inf if (bl or cr or x is None) else x) for t, x, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
            sub = dict(zip(r["starts"], r["substeps"]))
            a, b = fits[m]
            row[v] = (max(w[t] for t in st), sum(a + b * (sub.get(t) or 0) for t in st))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, alpha = cfg["eps"], cfg["alpha"]
    fits = {m: fit_probe_cost(cfg["cost_fit_scan_run"], "coarse_cfl", m)[:2] for m in (2, 4)}
    cal = load(cfg["calibration_pool"], cfg, fits)
    dev = load(cfg["dev_pool"], cfg, fits)
    res = {}
    for v in "abcd":
        c = [r for r in cal if r[v] is not None]
        S = np.array([r[v][0] for r in c]); bad = np.array([r["E"] > eps for r in c]); O = np.array([r["obj"] for r in c])
        lam, nb = calib_lambda(S, bad, O, np.array(sorted(set(O))), alpha)
        d = [r for r in dev if r[v] is not None]
        Sd = np.array([r[v][0] for r in d]); Ed = np.array([r["E"] for r in d]); bd = Ed > eps
        sh = np.array([r["split"] != "test_id" for r in d]); cost = np.array([r[v][1] for r in d])
        acc = Sd < lam
        res[v] = {"n_cal": len(c), "lambda": float(lam), "bad_objects": nb,
                  "auroc": float(roc_auc_score(bd, np.where(np.isfinite(Sd), Sd, 1e9))),
                  "far_shift": float(acc[sh & bd].mean()) if (sh & bd).any() else float("nan"),
                  "good_acc_id": float(acc[~sh & ~bd].mean()), "cost_s": float(cost.mean())}
    base = res["c"]
    ok = {v: res[v]["cost_s"] <= 0.5 * base["cost_s"] and res[v]["auroc"] >= base["auroc"] - 0.02 and res[v]["far_shift"] <= 0.10
          for v in "abd"}
    names = {"a": "(a) event window only", "b": "(b) event + middle", "c": "(c) baseline: 3 x m = 4", "d": "(d) 3 x m = 2"}
    md = ["# Pilot E2: cheaper verification (EXPLORATORY, development data; defined Oct 18, 2026)\n",
          "Thresholds: replicate-0 object rule on the calibration pool (variant d: id_new_0..3 only). Evaluation: the 2c pool.\n",
          "| variant | cal queries | threshold | AUROC (pooled) | false accept (shift) | good accepted (test_id) | probe cost (A100 s) | promising |",
          "|---|---|---|---|---|---|---|---|"]
    for v in "abcd":
        r = res[v]
        md.append(f"| {names[v]} | {r['n_cal']} | {r['lambda']:.4g} | {r['auroc']:.3f} | {r['far_shift']:.3f} | {r['good_acc_id']:.3f} | "
                  f"{r['cost_s']:.3f} | {'—' if v == 'c' else ('YES' if ok[v] else 'no')} |")
    md += ["", f"**E2 promising: {'YES' if any(ok.values()) else 'NO'}** (rule: cost <= 0.5 x baseline, AUROC >= baseline - 0.02, "
           "shift false accept <= 0.10).", "", "Exploratory pilot on development data."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"res": res, "promising": {k: bool(v) for k, v in ok.items()}}, run.path("summary.json"))
    print(f"[p8f] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
