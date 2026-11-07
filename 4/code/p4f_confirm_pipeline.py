"""Phase 4f: confirm certify-or-simulate with a REFERENCE fallback on the calibration pool.
CPU only. Pre-registered Oct 12, 2026 (PLAN.md "Phase 4f"), before any analysis of it.

Over R random 50/50 splits of the held-out objects: calibrate lambda on the calibration half
(object-level single-draw subsampling, eps = alpha = 0.10), then run the pipeline on the test
half. Accepted queries keep the generation; rejected ones are simulated at PhysCtrl's own
setting, which IS the ground truth (error 0) and costs one reference rollout.
  F1 risk     mean P(output err > eps) of cert_ref <= alpha + 0.01
  F2 compute  cost(cert_ref) / cost(sim_ref): mean < 1 and 97.5th percentile < 1
  F3 value    risk(random_ref) - risk(cert_ref): 2.5th percentile > 0
All rates are object-weighted (each test object counts once).
Probe cost on A100 = a x n_windows + b x substeps, fitted by least squares on the A100-timed
Phase 2c coarse_cfl m = 4 windows, because the calibration scans ran on H200.

Output: results/p4f_confirm/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p2c_summarize import query_inputs  # noqa: E402
from p4a_certificate import ood_index_from, threshold  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402
from p4d_pipeline import gen_costs  # noqa: E402


def fit_probe_cost(scan_run, fid, m):
    """seconds per window = a + b x substeps, from A100-timed scan rows."""
    X, y = [], []
    for f in sorted(results_dir(scan_run).glob("shard_*/scan.jsonl")):
        for r in read_jsonl(f):
            if r["fidelity"] != fid or int(r["m"]) != m:
                continue
            for sub, sec, b in zip(r["substeps"], r["seconds"], r["blown"]):
                if sub and sec and not b:
                    X.append([1.0, float(sub)]); y.append(float(sec))
    (a, b), *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return float(a), float(b), len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("run_name", "p4f_confirm"), cfg, synthetic=False)
    spec, model, eps, alpha, R = cfg["defect_spec"], cfg["model"], cfg["eps"], cfg["alpha"], cfg["n_replicates"]
    gcost, c_ref = gen_costs()
    a, b, n_fit = fit_probe_cost(cfg["cost_fit_scan_run"], spec["fidelity"], spec["m"])
    ood_index = ood_index_from(cfg["calib_build_runs"])
    data_root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)

    rows = []
    for scan_run, err_run in zip(cfg["calib_scan_runs"], cfg["calib_probe_runs"]):
        errs = {(r["qid"], r["model"]): r for f in sorted(results_dir(err_run).glob("shard_*/probes.jsonl"))
                for r in read_jsonl(f)}
        for f in sorted(results_dir(scan_run).glob("shard_*/scan.jsonl")):
            for r in read_jsonl(f):
                if r["source"] != model or r["fidelity"] != spec["fidelity"] or int(r["m"]) != spec["m"]:
                    continue
                pr = errs.get((r["qid"], model))
                if pr is None:
                    continue
                q = query_inputs(r["qid"], r["split"], data_root, ood_index)
                samples = np.load(groot / model / r["split"] / f"{r['qid']}.npz")["samples"]
                st = starts_for(spec["placement"], samples, q, spec["m"])
                win = {s: (np.inf if (bl or cr or raw is None) else raw)
                       for s, raw, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
                sub = dict(zip(r["starts"], r["substeps"]))
                rows.append({"obj": r["qid"].split("__")[0], "err": float(pr["err_mean"]),
                             "score": max(win[s] for s in st),
                             "c_probe": len(st) * a + b * sum((sub.get(s) or 0) for s in st)})
    objs_all = np.array(sorted({r["obj"] for r in rows}))
    O = np.array([r["obj"] for r in rows]); E = np.array([r["err"] for r in rows])
    S = np.array([r["score"] for r in rows]); CP = np.array([r["c_probe"] for r in rows])

    res = {k: [] for k in ("risk_cert", "risk_rand", "risk_gen", "err_cert", "err_gen", "fallback",
                           "cost_cert", "cost_ratio", "cost_ratio_b1")}
    for rep in range(R):
        rng = np.random.default_rng(rep)
        cal_o = set(rng.permutation(objs_all)[: len(objs_all) // 2])
        cal = np.array([o in cal_o for o in O])
        bad_c = cal & (E > eps)
        picks = [S[rng.choice(np.nonzero(bad_c & (O == o))[0])] for o in np.unique(O[bad_c])]
        lam = threshold(np.array(picks, float), alpha)
        t = ~cal
        acc = S[t] < lam
        e_t, o_t = E[t], O[t]
        out_err = np.where(acc, e_t, 0.0)                      # rejected -> reference rollout = ground truth
        p = 1 - object_rate(acc.astype(float), o_t)
        rand_risk = (1 - p) * (e_t > eps)                      # random fallback at the same rate (expectation)
        cost = lambda cg: cg + CP[t] + (~acc) * c_ref
        res["risk_cert"].append(object_rate((out_err > eps).astype(float), o_t))
        res["risk_rand"].append(object_rate(rand_risk.astype(float), o_t))
        res["risk_gen"].append(object_rate((e_t > eps).astype(float), o_t))
        res["err_cert"].append(object_rate(out_err, o_t)); res["err_gen"].append(object_rate(e_t, o_t))
        res["fallback"].append(p)
        res["cost_cert"].append(object_rate(cost(gcost[model]["best"]), o_t))
        res["cost_ratio"].append(res["cost_cert"][-1] / c_ref)
        res["cost_ratio_b1"].append(object_rate(cost(gcost[model]["b1"]), o_t) / c_ref)
    res = {k: np.array(v) for k, v in res.items()}
    q = lambda v, lo, hi: f"{v.mean():.3f} [{np.quantile(v, lo):.3f}, {np.quantile(v, hi):.3f}]"
    f1 = res["risk_cert"].mean() <= alpha + 0.01
    f2 = res["cost_ratio"].mean() < 1 and np.quantile(res["cost_ratio"], 0.975) < 1
    d3 = res["risk_rand"] - res["risk_cert"]
    f3 = np.quantile(d3, 0.025) > 0
    md = ["# Phase 4f: certify-or-simulate with a reference fallback (pre-registered Oct 12, 2026)\n",
          f"Base model; {len(rows)} calibration-pool trajectories over {len(objs_all)} held-out objects; {R} object "
          f"splits. Values: mean [2.5%, 97.5%] over splits, object-weighted. Reference rollout {c_ref:.3f} s; generation "
          f"{gcost[model]['best']:.3f} s (batch 1: {gcost[model]['b1']:.3f} s); probe cost {a:.3f} s/window + "
          f"{b * 1e3:.4f} ms/substep (fit on {n_fit} A100 windows).\n",
          f"- **F1 risk** P(err > {eps}) of cert_ref: {q(res['risk_cert'], .025, .975)}; target <= {alpha + 0.01:.2f}: "
          f"**{'PASS' if f1 else 'FAIL'}**",
          f"- **F2 compute** cost(cert_ref) / cost(sim_ref): {q(res['cost_ratio'], .025, .975)}: "
          f"**{'PASS' if f2 else 'FAIL'}** (with batch-1 generation: {q(res['cost_ratio_b1'], .025, .975)})",
          f"- **F3 value of verification** risk(random_ref) - risk(cert_ref): {q(d3, .025, .975)}: "
          f"**{'PASS' if f3 else 'FAIL'}**",
          "", "| quantity | mean [2.5%, 97.5%] |", "|---|---|",
          f"| fallback rate | {q(res['fallback'], .025, .975)} |",
          f"| mean error: generate only | {q(res['err_gen'], .025, .975)} |",
          f"| mean error: cert_ref | {q(res['err_cert'], .025, .975)} |",
          f"| risk: generate only | {q(res['risk_gen'], .025, .975)} |",
          f"| risk: random_ref | {q(res['risk_rand'], .025, .975)} |",
          f"| cost cert_ref (s/query) | {q(res['cost_cert'], .025, .975)} |"]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"F1": bool(f1), "F2": bool(f2), "F3": bool(f3),
               **{k: [float(v.mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in res.items()},
               "probe_fit": [a, b, n_fit], "c_ref": c_ref}, run.path("summary.json"))
    print(f"[p4f] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
