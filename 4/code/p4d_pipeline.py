"""Phase 4d: certify-or-simulate as a compute-accounted pipeline. CPU only.

Pre-registered Oct 12, 2026 (PLAN.md "Phase 4d"), before the fallback rollouts existed.
Evaluation set: the 300 Phase 2c queries per model (sample 0 of each generation). Costs are
A100 seconds per query:
  generation  Phase 1 per-sample time (best batch; batch-1 reported too); x8 for a seed-spread
              certificate, which needs 8 samples;
  probes      the certificate's windows, the measured seconds in the Phase 2c scan (A100);
  fallback    one full rollout from the true initial state at the fallback fidelity
              (p4d_fullsim; coarse_cfl), measured on A100;
  reference   PhysCtrl's own setting: the Phase 1 'physctrl' rollout, nosync mode.
Pipelines:
  gen         always generate                        error e_gen
  sim_ref     always simulate at reference            error 0 (it IS the ground truth)
  sim_fb      always simulate at the fallback fidelity  error e_fb
  cert        generate + probe; accept if score < lambda, else fallback
  random      the same fallback rate as cert, chosen at random (expectation; no probes)
  oracle      the same fallback rate, given to the queries with the largest e_gen - e_fb
  ss_cert     the same as cert with seed spread as the score (8 samples, no probes)
lambda: base defect, the Phase 4a object-level value (replicate 0, eps = alpha = 0.10);
seed spread, its Phase 4a value; large seed spread, its object-level value.
Tests (primary model base):
  T1 value of verification: mean error random - cert > 0 (paired bootstrap, 95% lower bound > 0).
  T2 risk: fraction of outputs with err > eps, for cert, gen, random, ss_cert (reported).
  T3 compute: cost(cert) against sim_fb and sim_ref, and the break-even simulator cost
     c* = (c_gen + c_probe) / (1 - reject rate), above which cert is the cheaper option.
  T4 curves: error and risk against cost as lambda sweeps the score quantiles, for cert,
     ss_cert and random (reported in the summary).

Output: results/p4d_pipeline/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bootstrap_ci, load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p2c_summarize import query_inputs  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402


def gen_costs():
    rows = read_jsonl(results_dir("p1_cost_A100") / "timings.jsonl")
    out = {}
    for m in ("base", "large"):
        g = [r for r in rows if r["kind"] == "generation" and r["model"] == m]
        out[m] = {"best": float(min(np.median([r["per_sample_s_median"] for r in g if r["batch_size"] == b])
                                    for b in {r["batch_size"] for r in g})),
                  "b1": float(np.median([r["per_sample_s_median"] for r in g if r["batch_size"] == 1]))}
    sims = [r for r in rows if r["kind"] == "simulation" and r["regime"] == "physctrl" and r["mode"] == "nosync"
            and r.get("stable") and "rollout_s" in r]
    return out, float(np.median([r["rollout_s"] for r in sims]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("run_name", "p4d_pipeline"), cfg, synthetic=False)
    gcost, c_ref = gen_costs()
    fb = {r["qid"]: r for f in sorted(results_dir(cfg["fullsim_run"]).glob("shard_*/fullsim.jsonl"))
          for r in read_jsonl(f) if r["fidelity"] == cfg["fallback_fidelity"]}
    probes = {(r["qid"], r["model"]): r for f in sorted(results_dir(cfg["probe_run"]).glob("shard_*/probes.jsonl"))
              for r in read_jsonl(f)}
    scan = {(r["qid"], r["source"], r["fidelity"], int(r["m"])): r
            for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)}
    ood_index = ood_index_from(["p2_build_ood"])
    data_root, groot, eps, nb = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False), cfg["eps"], cfg["n_boot"]
    spec = cfg["defect_spec"]
    md = ["# Phase 4d: certify-or-simulate (pre-registered Oct 12, 2026)\n",
          f"Costs in A100 seconds per query. Reference rollout (PhysCtrl setting, nosync): {c_ref:.3f} s. "
          f"Generation per sample (best batch / batch 1): " + ", ".join(
              f"{m} {v['best']:.3f} / {v['b1']:.3f}" for m, v in gcost.items()) + f". eps = {eps}.\n"]
    out = {}
    for cert in cfg["certificates"]:
        model, name, kind, lam = cert["model"], cert["name"], cert["score"], float(cert["lambda"])
        rows = []
        for (qid, mdl), pr in sorted(probes.items()):
            if mdl != model or qid not in fb:
                continue
            f = fb[qid]
            e_fb = float(np.mean(f["err_t"])) if f.get("stable") else 0.0          # unstable -> reference rerun
            c_fb = f.get("rollout_s", 0.0) + (0.0 if f.get("stable") else c_ref)   # crashed rows carry no time
            sr = scan.get((qid, model, spec["fidelity"], spec["m"]))
            if kind == "defect":
                if sr is None:
                    continue
                q = query_inputs(qid, pr["split"], data_root, ood_index)
                samples = np.load(groot / model / pr["split"] / f"{qid}.npz")["samples"]
                starts = starts_for(spec["placement"], samples, q, spec["m"])
                win = {s: (np.inf if (b or c or raw is None) else raw)
                       for s, raw, b, c in zip(sr["starts"], sr["raw"], sr["blown"], sr["crashed"])}
                secs = dict(zip(sr["starts"], sr["seconds"]))
                score, c_probe = max(win[s] for s in starts), float(sum(secs[s] or 0.0 for s in starts))
                c_gen = gcost[model]["best"]
            else:
                score, c_probe, c_gen = float(pr["seed_spread"]), 0.0, 8 * gcost[model]["best"]
            rows.append({"qid": qid, "split": pr["split"], "e_gen": float(pr["err_mean"]), "e_fb": e_fb,
                         "c_gen": c_gen, "c_probe": c_probe, "c_fb": c_fb, "score": score})
        n = len(rows)
        e_gen, e_fb = np.array([r["e_gen"] for r in rows]), np.array([r["e_fb"] for r in rows])
        c_gen, c_probe, c_fb = (np.array([r[k] for r in rows]) for k in ("c_gen", "c_probe", "c_fb"))
        score = np.array([r["score"] for r in rows])
        rej = score >= lam
        p = rej.mean()
        e_cert = np.where(rej, e_fb, e_gen)
        e_rand = (1 - p) * e_gen + p * e_fb                       # expectation over the random choice
        k = int(round(p * n))
        orc = np.zeros(n, bool)
        orc[np.argsort(-(e_gen - e_fb))[:k]] = True
        e_orc = np.where(orc, e_fb, e_gen)
        cost = {"gen": c_gen.mean(), "sim_fb": c_fb.mean(), "sim_ref": c_ref,
                "cert": (c_gen + c_probe + rej * c_fb).mean(), "random": (c_gen + p * c_fb).mean(),
                "oracle": (c_gen + orc * c_fb).mean()}
        err = {"gen": e_gen, "sim_fb": e_fb, "sim_ref": np.zeros(n), "cert": e_cert, "random": e_rand, "oracle": e_orc}
        t1 = bootstrap_ci(lambda a, b: float(np.mean(a - b)), [e_rand, e_cert], n_boot=nb)
        risk = {k2: float(np.mean(v > eps)) for k2, v in err.items() if k2 != "random"}
        risk["random"] = float(np.mean((1 - p) * (e_gen > eps) + p * (e_fb > eps)))
        c_star = float((c_gen.mean() + c_probe.mean()) / max(1 - p, 1e-9))
        md += [f"\n## {name} ({model}; score = {kind}; lambda = {lam:g}; n = {n}; fallback rate {p:.2f})\n",
               "| pipeline | mean error | risk (err > eps) | cost (s/query) |", "|---|---|---|---|"]
        for k2 in ("gen", "random", "cert", "oracle", "sim_fb", "sim_ref"):
            md.append(f"| {k2} | {np.mean(err[k2]):.4f} | {risk[k2]:.3f} | {cost[k2]:.3f} |")
        md.append(f"\nT1 random - cert mean error: {t1[0]:+.4f} [{t1[1]:+.4f}, {t1[2]:+.4f}] "
                  f"({'PASS' if t1[1] > 0 else 'FAIL'})")
        md.append(f"T3 break-even simulator cost: cert is cheaper than always-simulate when one rollout costs more than "
                  f"{c_star:.3f} s; here sim_fb costs {c_fb.mean():.3f} s and sim_ref {c_ref:.3f} s "
                  f"(per-query generation {c_gen.mean():.3f} s, probes {c_probe.mean():.3f} s).")
        # T4: sweep lambda over score quantiles
        md += ["\nT4 sweep (lambda at score quantiles): fallback rate / cost / mean error / risk",
               "| quantile | fallback | cost | error cert | error random | risk cert | risk random |",
               "|---|---|---|---|---|---|---|"]
        finite = score[np.isfinite(score)]
        for qt in cfg["sweep_quantiles"]:
            lq = np.quantile(finite, qt) if len(finite) else np.inf
            r2 = score >= lq
            p2 = r2.mean()
            ec, er = np.where(r2, e_fb, e_gen), (1 - p2) * e_gen + p2 * e_fb
            md.append(f"| {qt:.2f} | {p2:.2f} | {(c_gen + c_probe + r2 * c_fb).mean():.3f} | {ec.mean():.4f} | "
                      f"{er.mean():.4f} | {np.mean(ec > eps):.3f} | "
                      f"{np.mean((1 - p2) * (e_gen > eps) + p2 * (e_fb > eps)):.3f} |")
        out[name] = {"fallback_rate": float(p), "cost": {k2: float(v) for k2, v in cost.items()},
                     "error": {k2: float(np.mean(v)) for k2, v in err.items()}, "risk": risk,
                     "T1": [float(x) for x in t1], "T1_pass": bool(t1[1] > 0), "break_even_s": c_star}
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json(out, run.path("summary.json"))
    print(f"[p4d_pipeline] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
