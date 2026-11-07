"""Phase 7d: the missing baselines on the two-object contact benchmark. CPU (after GPU job 78).

Pre-registered Oct 17, 2026 (PLAN.md "Phase 7d"), before any 7d code ran.
Scenes, rows, costs and the SF pipeline come from the 4o verdict (p4n_summarize.build / pipes). New baselines,
with thresholds exactly as used under shift (single-object calibration pool, nothing refit here):
  coarse only   coarse_cfl rollout of the joint scene from the true start (job 78), charged at the 4d rate
  XF            accept the coarse rollout if its agreement with composed sample 0 < lambda_agreement (4m)
  seed spread   8 composed samples; accept sample 0 if the spread < lambda_seed_spread (4m); charged 8 generations
  residual gate momentum residual of composed sample 0 (all particles) < lambda_R (7a)
Tests on contact scenes (cluster bootstrap over the first object, Bonferroni over three, 98.33% CIs):
  D1 risk(coarse) - risk(SF) > 0   D2 risk(seed spread) - risk(SF) > 0   D3 risk(residual) - risk(SF) > 0
  XF dominates SF iff risk(XF) <= risk(SF) + 0.02 and cost(SF) - cost(XF) has lower bound > 0 (reported either way)

Output: results/p7d_contact/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p4m_confirm import agreement  # noqa: E402
from p4n_summarize import build, pipes  # noqa: E402
from p7a_metrics import com_prediction, momentum_inputs, residual  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, B = cfg["eps"], cfg["n_boot"]
    lv = 1 - cfg["family_alpha"] / 3
    Bd = build(cfg)
    cg, c_ref, rate, groot = Bd["cg"], Bd["c_ref"], Bd["rate"], Bd["groot"]
    lam = load_json(results_dir("p4m_confirm") / "summary.json")["lambda"]
    lam_R = float(load_json(results_dir("p7a_metrics") / "summary.json")["lambda_R"])
    paths = {e["qid"]: e["path"] for f in sorted(results_dir(cfg["build_run"]).glob("shard_*/ood_manifest.json"))
             for e in load_json(f)}
    crow = {r["qid"]: r for f in sorted(results_dir(cfg["coarse_run"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    cdir = Path(os.path.expandvars(cfg["coarse_dir"]))

    out_sets = {}
    for kind in ("contact", "control"):
        rows = [r for r in Bd["R"] if r["kind"] == kind]
        P = pipes(rows, cg, c_ref)
        E = np.array([r["E"] for r in rows])
        SS, R0, CE, CC, AG = [], [], [], [], []
        n_crash = 0
        for r in rows:
            s = np.load(groot / "base" / r["split"] / f"{r['qid']}.npz")["samples"].astype(np.float64)
            SS.append(float(np.linalg.norm(s - s.mean(0, keepdims=True), axis=3).mean()))   # p2b_probe's definition
            chat, w = com_prediction(momentum_inputs(paths[r["qid"]]))
            R0.append(residual(s[0], chat, w))
            x = crow.get(r["qid"])
            if x is None:
                raise RuntimeError(f"{r['qid']}: no coarse row; job 78 is incomplete")
            if x.get("crashed") or x.get("coarse_blown"):
                n_crash += 1
                CE.append(np.inf); AG.append(np.inf); CC.append(rate * (x.get("coarse_substeps") or 48 * 52))
                continue
            CE.append(float(np.mean(x["coarse_err_t"]))); CC.append(rate * x["coarse_substeps"])
            AG.append(agreement(s[0], np.load(cdir / r["split"] / f"{r['qid']}.npz")["coarse"].astype(np.float64)))
        SS, R0, CE, CC, AG = map(np.array, (SS, R0, CE, CC, AG))
        acc_x, acc_s, acc_r = AG < lam["agreement"], SS < lam["seed_spread"], R0 < lam_R
        P["coarse only"] = (CE, CC)
        P["xf"] = (np.where(acc_x, CE, 0.0), cg + CC + (~acc_x) * c_ref)
        P["seed spread"] = (np.where(acc_s, E, 0.0), 8 * cg + (~acc_s) * c_ref)
        P["residual gate"] = (np.where(acc_r, E, 0.0), cg + (~acc_r) * c_ref)
        out_sets[kind] = (rows, P, n_crash, {"xf": acc_x.mean(), "seed spread": acc_s.mean(), "residual gate": acc_r.mean()})

    rows, P, n_crash, acc_rates = out_sets["contact"]
    O = np.array([r["obj"] for r in rows])

    def boot(vals, ci_level=0.95):
        vals = np.asarray(vals, float)
        ob = np.array(sorted(set(O)))
        idx = {o: np.nonzero(O == o)[0] for o in ob}
        f = lambda d: np.mean([vals[idx[o]].mean() for o in d])                                       # noqa: E731
        rng = np.random.default_rng(1)
        bs = [f(rng.choice(ob, len(ob), replace=True)) for _ in range(B)]
        a = (1 - ci_level) / 2
        return float(f(ob)), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))
    bad = lambda k: (P[k][0] > eps).astype(float)                                                      # noqa: E731
    d1 = boot(bad("coarse only") - bad("sf"), lv)
    d2 = boot(bad("seed spread") - bad("sf"), lv)
    d3 = boot(bad("residual gate") - bad("sf"), lv)
    xf_saving = boot(P["sf"][1] - P["xf"][1], lv)
    xf_dom = boot(bad("xf"))[0] <= boot(bad("sf"))[0] + 0.02 and xf_saving[1] > 0
    D = {"D1": d1[1] > 0, "D2": d2[1] > 0, "D3": d3[1] > 0}

    f4 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                                        # noqa: E731
    labs = (("generate only", "PhysCtrl (two objects composed)"), ("coarse only", "coarse simulation only"),
            ("seed spread", "seed-spread deferral (8 samples)"), ("residual gate", "momentum-residual gate"),
            ("xf", "accept coarse if it agrees (XF)"), ("cert_ref", "certify or simulate"),
            ("sf", "**ours: certify + repair, shadow-timed switch (SF)**"), ("always reference", "always simulate"))
    md = ["# Phase 7d: the missing baselines on the contact benchmark (pre-registered Oct 17, 2026)\n",
          f"Contact scenes: {len(rows)} (4o); controls: {len(out_sets['control'][0])}. Coarse rollouts crashed or blown "
          f"(counted as failures for coarse only and rejected by XF): {n_crash}. Thresholds from the single-object calibration "
          f"pool: agreement {lam['agreement']:.5g}, seed spread {lam['seed_spread']:.5g}, residual {lam_R:.5g}; window tau as 4o. "
          f"CIs {lv:.4f} two-sided (Bonferroni over 3); cluster bootstrap over the first object.\n",
          f"- **D1** risk(coarse only) - risk(SF): {f4(d1)}: **{'PASS' if D['D1'] else 'FAIL'}**",
          f"- **D2** risk(seed spread) - risk(SF): {f4(d2)}: **{'PASS' if D['D2'] else 'FAIL'}**",
          f"- **D3** risk(residual gate) - risk(SF): {f4(d3)}: **{'PASS' if D['D3'] else 'FAIL'}**",
          f"- **XF rule:** saving SF - XF {f4(xf_saving)} s/scene; XF {'DOMINATES' if xf_dom else 'does not dominate'} SF", "",
          "| pipeline (contact scenes) | mean error | risk | cost / always-simulate | accept rate |", "|---|---|---|---|---|"]
    for k, lab in labs:
        ar = f"{acc_rates[k]:.2f}" if k in acc_rates else "—"
        md.append(f"| {lab} | {boot(np.minimum(P[k][0], 10.0))[0]:.4f} | {boot(bad(k))[0]:.3f} | "
                  f"{boot(P[k][1])[0] / c_ref:.2f} | {ar} |")
    rows_c, P_c, n_crash_c, acc_c = out_sets["control"]
    md += ["", f"Controls ({len(rows_c)} scenes, objects never touch; query-level means): " + "; ".join(
        f"{lab.strip('*')} risk {np.mean(P_c[k][0] > eps):.3f}, cost {np.mean(P_c[k][1]) / c_ref:.2f}" for k, lab in labs)]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({**{k: bool(v) for k, v in D.items()}, "xf_dominates": bool(xf_dom), "d": [d1, d2, d3],
               "xf_saving": xf_saving, "n_crash": n_crash, "accept_rates": {k: float(v) for k, v in acc_rates.items()}},
              run.path("summary.json"))
    print(f"[p7d] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
