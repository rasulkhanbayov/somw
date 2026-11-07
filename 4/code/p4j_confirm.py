"""Phase 4j verdict: held-out confirmation of 4i V2 and the shuffled-restart ablation. CPU only.

Pre-registered Oct 13, 2026 (PLAN.md "Phase 4j"). Base calibration pool, the same 500 object
splits and lambda as Phase 4f (identical seeds, so cert_ref reproduces 4f). On each test half:
  cert_ref  accept if the 3 spread_event windows pass lambda, else a reference rollout (4f)
  V2        the 4h chain with tau = lambda: all pass -> accept; anchor fails -> reference rollout;
            else restart at the first failure k_j from the coarse-shadow state (4i V2)
  shuffled  V2 with the k_j of the test half's suffix queries randomly permuted (ablation)
  earliest  V2 restarting at k = 3 for every suffix query (secondary)
  coarse    a full coarse_cfl rollout, no generation (Phase 4d; secondary)
Tests: J1 mean risk(V2) <= alpha + 0.01 and mean[risk(V2) - risk(cert_ref)] <= 0.02;
       J2 cost(V2)/cost(cert_ref) mean < 1 and 97.5th percentile < 1;
       J3 mean error(shuffled) - mean error(V2): 2.5th percentile > 0.
Confirmed if J1 and J2 pass; the timing claim needs J3 too.

Output: results/p4j_confirm/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402

SUBSTEPS_PER_FRAME = 417


def err(e):
    m = float(np.mean(e)) if e is not None else np.inf
    return m if np.isfinite(m) else np.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p4j_confirm"), cfg, synthetic=False)
    spec, model, eps, alpha, R = cfg["defect_spec"], cfg["model"], cfg["eps"], cfg["alpha"], cfg["n_replicates"]
    chain_k = [int(k) for k in cfg["chain_k"]]
    chain_starts = [0] + [2 * k + 1 for k in chain_k]
    gcost, c_ref = gen_costs()
    cg = gcost[model]["best"]
    share = lambda subs: c_ref * subs / float(cfg["full_substeps"])     # noqa: E731
    a, b, n_fit = fit_probe_cost(cfg["cost_fit_scan_run"], spec["fidelity"], spec["m"])
    ood_index = ood_index_from(cfg["calib_build_runs"])
    data_root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)

    allk = {r["qid"]: r for f in sorted(results_dir(cfg.get("run_name", "p4j_allk")).glob("shard_*/allk.jsonl"))
            for r in read_jsonl(f)}
    d4 = {r["qid"]: r for f in sorted(results_dir(cfg["shadow_cost_run"]).glob("shard_*/fullsim.jsonl"))
          for r in read_jsonl(f) if r["fidelity"] == "coarse_cfl"}
    rate_split = defaultdict(list)
    for r in d4.values():
        if r.get("stable") and r.get("rollout_s") and r.get("substeps"):
            rate_split[r["split"]].append(r["rollout_s"] / r["substeps"])
    rate_split = {k: float(np.median(v)) for k, v in rate_split.items()}

    rows, n_gen_mismatch, n_crash, n_blown = [], 0, 0, 0
    for scan_run, err_run in zip(cfg["calib_scan_runs"], cfg["calib_probe_runs"]):
        errs = {(r["qid"], r["model"]): r for f in sorted(results_dir(err_run).glob("shard_*/probes.jsonl"))
                for r in read_jsonl(f)}
        for f in sorted(results_dir(scan_run).glob("shard_*/scan.jsonl")):
            for r in read_jsonl(f):
                if r["source"] != model or r["fidelity"] != spec["fidelity"] or int(r["m"]) != spec["m"]:
                    continue
                pr = errs.get((r["qid"], model))
                if pr is None:                      # same population as Phase 4f
                    continue
                x = allk.get(r["qid"])
                if x is None:
                    raise RuntimeError(f"no 4j row for {r['qid']}; the GPU job is incomplete")
                q = query_inputs(r["qid"], r["split"], data_root, ood_index)
                samples = np.load(groot / model / r["split"] / f"{r['qid']}.npz")["samples"]
                st = starts_for(spec["placement"], samples, q, spec["m"])
                win = {s: (np.inf if (bl or cr or raw is None) else raw)
                       for s, raw, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
                sub = dict(zip(r["starts"], r["substeps"]))
                wcost = lambda s: a + b * (sub.get(s) or 0)                 # noqa: E731
                d = d4.get(r["qid"])
                rate = (d["rollout_s"] / d["substeps"]) if (d and d.get("stable") and d.get("substeps")) \
                    else rate_split[r["split"]]
                ek, ck = [], []
                for k in chain_k:
                    s = 2 * k + 1
                    if x.get("crashed"):                # charged the planned shadow + suffix + c_ref
                        sh_plan = (d["substeps"] * s / 48.0) if d else 0.0
                        ek.append(0.0); ck.append(rate * sh_plan + share((47 - s) * SUBSTEPS_PER_FRAME) + c_ref)
                        continue
                    v = x["k"][str(k)]
                    c = rate * v["shadow_substeps"] + share(v["substeps"])
                    if v["shadow_blown"] or v["blown"]:
                        n_blown += 1
                        ek.append(0.0); ck.append(c + c_ref)
                    else:
                        ek.append(err(v["err_t"])); ck.append(c)
                n_crash += bool(x.get("crashed"))
                if not x.get("crashed") and abs(np.mean(x["gen_err_t"]) - pr["err_mean"]) > 1e-4:
                    n_gen_mismatch += 1
                rows.append({"obj": r["qid"].split("__")[0], "err": float(pr["err_mean"]),
                             "score": max(win[s] for s in st), "c_probe": len(st) * a + b * sum((sub.get(s) or 0) for s in st),
                             "W": [win[s] for s in chain_starts], "CW": np.cumsum([wcost(s) for s in chain_starts]).tolist(),
                             "EK": ek, "CK": ck,
                             "coarse_err": err(d["err_t"]) if (d and d.get("stable") and d.get("err_t")) else np.inf,
                             "coarse_cost": d["rollout_s"] if (d and d.get("rollout_s")) else rate * 48 * 52})
    O = np.array([r["obj"] for r in rows]); E = np.array([r["err"] for r in rows])
    S = np.array([r["score"] for r in rows]); CP = np.array([r["c_probe"] for r in rows])
    W = np.array([r["W"] for r in rows], float); CW = np.array([r["CW"] for r in rows])
    EK = np.array([r["EK"] for r in rows]); CK = np.array([r["CK"] for r in rows])
    CE = np.array([r["coarse_err"] for r in rows]); CC = np.array([r["coarse_cost"] for r in rows])
    objs_all = np.array(sorted(set(O)))
    n = len(rows)
    idx = np.arange(n)

    def chain_outcome(jv, j_probe, t):
        """Error and cost on test rows t for first-failure indices jv (-1 accept, 0 anchor, i>=1 restart at chain_k[i-1]).
        Probe cost always follows the TRUE decision j_probe (the windows that were actually evaluated)."""
        jt, jp = jv[t], j_probe[t]
        probe = CW[idx[t], np.where(jp < 0, len(chain_starts) - 1, jp)]
        kk = np.clip(jt - 1, 0, len(chain_k) - 1)
        e = np.where(jt < 0, E[t], np.where(jt == 0, 0.0, EK[idx[t], kk]))
        c = cg + probe + np.where(jt < 0, 0.0, np.where(jt == 0, c_ref, CK[idx[t], kk]))
        return e, c

    keys = ("risk_cert", "risk_v2", "risk_shuf", "risk_early", "risk_coarse", "risk_gen", "err_cert", "err_v2",
            "err_shuf", "err_early", "err_coarse", "err_gen", "cost_cert", "cost_v2", "cost_shuf", "cost_early",
            "cost_coarse", "ratio_v2_cert", "ratio_v2_ref", "d_risk", "d_err_shuf", "frac_accept", "frac_full",
            "frac_suffix", "lam")
    res = {k: [] for k in keys}
    for rep in range(R):
        rng = np.random.default_rng(rep)                       # identical to Phase 4f
        cal_o = set(rng.permutation(objs_all)[: len(objs_all) // 2])
        cal = np.array([o in cal_o for o in O])
        bad_c = cal & (E > eps)
        picks = [S[rng.choice(np.nonzero(bad_c & (O == o))[0])] for o in np.unique(O[bad_c])]
        lam = threshold(np.array(picks, float), alpha)
        t = ~cal
        o_t = O[t]
        rate = lambda v: object_rate(np.asarray(v, float), o_t)      # noqa: E731
        # cert_ref (4f)
        acc = S[t] < lam
        e_c = np.where(acc, E[t], 0.0)
        c_c = cg + CP[t] + (~acc) * c_ref
        # chain decision with tau = lambda
        fail = W >= lam
        j = np.where(fail.any(axis=1), fail.argmax(axis=1), -1)
        e_v, c_v = chain_outcome(j, j, t)
        rng2 = np.random.default_rng(100_000 + rep)
        js = j.copy()
        sfx = np.nonzero(t & (j >= 1))[0]
        js[sfx] = rng2.permutation(j[sfx])
        e_s, c_s = chain_outcome(js, j, t)
        je = np.where(j >= 1, 1, j)
        e_e, c_e = chain_outcome(je, j, t)
        res["risk_cert"].append(rate(e_c > eps)); res["risk_v2"].append(rate(e_v > eps))
        res["risk_shuf"].append(rate(e_s > eps)); res["risk_early"].append(rate(e_e > eps))
        res["risk_coarse"].append(rate(CE[t] > eps)); res["risk_gen"].append(rate(E[t] > eps))
        res["err_cert"].append(rate(e_c)); res["err_v2"].append(rate(e_v)); res["err_shuf"].append(rate(e_s))
        res["err_early"].append(rate(e_e)); res["err_coarse"].append(rate(np.minimum(CE[t], 10.0)))
        res["err_gen"].append(rate(E[t]))
        res["cost_cert"].append(rate(c_c)); res["cost_v2"].append(rate(c_v)); res["cost_shuf"].append(rate(c_s))
        res["cost_early"].append(rate(c_e)); res["cost_coarse"].append(rate(CC[t]))
        res["ratio_v2_cert"].append(res["cost_v2"][-1] / res["cost_cert"][-1])
        res["ratio_v2_ref"].append(res["cost_v2"][-1] / c_ref)
        res["d_risk"].append(res["risk_v2"][-1] - res["risk_cert"][-1])
        res["d_err_shuf"].append(res["err_shuf"][-1] - res["err_v2"][-1])
        jt = j[t]
        res["frac_accept"].append(rate(jt < 0)); res["frac_full"].append(rate(jt == 0)); res["frac_suffix"].append(rate(jt >= 1))
        res["lam"].append(lam)
    res = {k: np.array(v, float) for k, v in res.items()}
    q = lambda v: f"{v.mean():.3f} [{np.quantile(v, .025):.3f}, {np.quantile(v, .975):.3f}]"   # noqa: E731
    j1 = res["risk_v2"].mean() <= alpha + 0.01 and res["d_risk"].mean() <= 0.02
    j2 = res["ratio_v2_cert"].mean() < 1 and np.quantile(res["ratio_v2_cert"], 0.975) < 1
    j3 = np.quantile(res["d_err_shuf"], 0.025) > 0
    f4 = results_dir("p4f_confirm") / "summary.json"
    ref4f = json.loads(f4.read_text()) if f4.exists() else {}
    md = ["# Phase 4j: held-out confirmation of V2 + shuffled-restart ablation (pre-registered Oct 13, 2026)\n",
          f"Base calibration pool: {n} queries, {len(objs_all)} objects, {R} object splits (the Phase 4f seeds). "
          f"Values: mean [2.5%, 97.5%] over splits, object-weighted. Reference {c_ref:.3f} s, generation {cg:.3f} s, "
          f"probe fit {a:.3f} s/window + {b * 1e3:.4f} ms/substep ({n_fit} A100 windows).\n",
          f"Blown shadow/suffix restarts (fallback to reference): {n_blown} of {n * len(chain_k)}; crashed queries: {n_crash}; "
          f"queries whose sample-0 error differs from the 4f error: {n_gen_mismatch}.\n",
          f"Check: cert_ref risk {res['risk_cert'].mean():.4f}, cost ratio {np.mean(res['cost_cert']) / c_ref:.4f} "
          f"(Phase 4f: {ref4f.get('risk_cert', ['-'])[0]}, {ref4f.get('cost_ratio', ['-'])[0]}).\n",
          f"- **J1 risk:** V2 {q(res['risk_v2'])} (target <= {alpha + 0.01:.2f}); V2 - cert_ref {q(res['d_risk'])} "
          f"(target mean <= 0.02): **{'PASS' if j1 else 'FAIL'}**",
          f"- **J2 compute:** cost(V2) / cost(cert_ref) {q(res['ratio_v2_cert'])}: **{'PASS' if j2 else 'FAIL'}**",
          f"- **J3 localisation:** error(shuffled) - error(V2) {q(res['d_err_shuf'])}: **{'PASS' if j3 else 'FAIL'}**",
          f"- **Confirmation: {'CONFIRMED' if (j1 and j2) else 'NOT CONFIRMED'}**; timing claim: "
          f"**{'supported' if (j1 and j2 and j3) else 'not supported'}**", "",
          "| pipeline | mean error | risk | cost (s/query) | cost / always-reference |", "|---|---|---|---|---|"]
    for k, label in (("gen", "generate only"), ("coarse", "coarse_cfl only (no generation)"), ("cert", "certify-or-simulate (4f)"),
                     ("v2", "**V2 (certified prefix + shadow suffix)**"), ("shuf", "V2, shuffled restarts (ablation)"),
                     ("early", "V2, earliest restart k = 3")):
        cost = res.get(f"cost_{k}", np.full(R, cg))
        md.append(f"| {label} | {q(res[f'err_{k}'])} | {q(res[f'risk_{k}'])} | {q(cost)} | {cost.mean() / c_ref:.2f} |")
    md += [f"| always reference | 0 | 0 | {c_ref:.3f} | 1.00 |", "",
           f"Chain decisions on the test halves: accept {q(res['frac_accept'])}, anchor fails {q(res['frac_full'])}, "
           f"suffix {q(res['frac_suffix'])}; lambda {q(res['lam'])}.",
           "Coarse-only mean error caps non-finite rollouts at 10 for the mean (they count as failures in the risk)."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"J1": bool(j1), "J2": bool(j2), "J3": bool(j3),
               **{k: [float(v.mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in res.items()},
               "n": n, "n_blown": n_blown, "n_crash": n_crash, "n_gen_mismatch": n_gen_mismatch}, run.path("summary.json"))
    print(f"[p4j_confirm] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
