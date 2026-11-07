"""Phase 7c: stricter and looser targets with every gate recalibrated. CPU only, existing data.

Pre-registered Oct 17, 2026 (PLAN.md "Phase 7c"), before any 7c code ran.
Grid eps in {0.05, 0.10, 0.20} x alpha in {0.05, 0.10, 0.20}. In each cell every gate is recalibrated on the
4m calibration pool with bad = error > eps: window tau, seed spread and momentum residual by the replicate-0
rule, the agreement (XF) threshold by the pooled rule. Pipelines (4m / 7a definitions): generate, coarse,
certify-or-simulate, gated V2 (4k rule at the new tau), XF, seed-spread deferral, residual gate; risk at eps.
The cell eps = alpha = 0.10 must reproduce the 4m / 7a thresholds (checked; a mismatch raises).
Tests (in distribution, eps = 0.05, alpha = 0.10; Bonferroni over three, 98.33% CIs):
  S1 risk(gated) <= risk(cert) + 0.02 (point)   S2 saving cert - gated > 0   S3 risk(gated) - risk(coarse) < 0

Output: results/p7c_targets/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4b_certificate_objects import object_rate, pooled_threshold  # noqa: E402
from p4k_gated import pipelines  # noqa: E402
from p4m_confirm import assemble, calib_lambda, calibration_pool  # noqa: E402
from p7a_metrics import com_prediction, momentum_inputs, residual  # noqa: E402


def sample0_residual(path, gen_file) -> float:
    chat, w = com_prediction(momentum_inputs(path))
    return residual(np.load(gen_file)["samples"][0].astype(np.float64), chat, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p7c_targets"), cfg, synthetic=False)
    model, B = cfg["model"], cfg["n_boot"]
    X = assemble(cfg)
    A, meta, SS, AG = X["A"], X["meta"], X["SS"], X["AG"]
    c_ref, cg = meta["c_ref"], meta["cg"]
    C = calibration_pool(cfg)
    objs = np.array(sorted(set(C["obj"])))
    groot = gen_dir(cfg, False)

    # residual on the calibration pool, in 7a's row order (sorted qids of the probe rows) so lambda_R reproduces
    oi_cal = ood_index_from(cfg["calib_build_runs"])
    cal = {}
    for rn in cfg["calib_probe_runs"]:
        for f in sorted(results_dir(rn).glob("shard_*/probes.jsonl")):
            for r in read_jsonl(f):
                if r["model"] == model:
                    cal[r["qid"]] = (r["split"], float(r["err_mean"]))
    qs = sorted(cal)
    RC = np.array([sample0_residual(oi_cal[q], groot / model / cal[q][0] / f"{q}.npz") for q in qs])
    EC_r = np.array([cal[q][1] for q in qs])
    OC_r = np.array([q.split("__")[0] for q in qs])
    oi = ood_index_from(cfg["build_runs"])
    R0 = np.array([sample0_residual(oi[q], groot / model / sp / f"{q}.npz") for q, sp in zip(A["qid"], A["split"])])

    O = A["obj"]
    shift = np.array([s.startswith("ood2_") for s in A["split"]])
    sets = {"in distribution": ~shift, "shift": shift}
    names = (("gen", "generate only"), ("coarse", "coarse only"), ("cert", "certify or simulate"),
             ("gated", "**ours (gated V2)**"), ("xf", "accept coarse if it agrees"),
             ("cert_ss", "seed-spread deferral"), ("resid_ref", "momentum-residual gate"))

    def cell(eps, alpha):
        lam = {"window": calib_lambda(C["S"], C["E"] > eps, C["obj"], objs, alpha)[0],
               "seed_spread": calib_lambda(C["SS"], C["E"] > eps, C["obj"], objs, alpha)[0],
               "residual": calib_lambda(RC, EC_r > eps, OC_r, np.array(sorted(set(OC_r))), alpha)[0]}
        b = C["EC"] > eps
        lam["agreement"] = pooled_threshold(C["AG"][b], C["obj"][b], alpha) if b.any() else -np.inf
        out = pipelines(A, lam["window"], c_ref, cg)[0]
        acc_x, acc_s, acc_r = AG < lam["agreement"], SS < lam["seed_spread"], R0 < lam["residual"]
        out["xf"] = (np.where(acc_x, A["CE"], 0.0), cg + A["CC"] + (~acc_x) * c_ref)
        out["cert_ss"] = (np.where(acc_s, A["E"], 0.0), 8 * cg + (~acc_s) * c_ref)
        out["resid_ref"] = (np.where(acc_r, A["E"], 0.0), cg + (~acc_r) * c_ref)
        return lam, out

    # reproduction check: eps = alpha = 0.10 must give the 4m / 7a thresholds
    lam10, _ = cell(0.10, 0.10)
    ref = {"window": X["lam"]["window"][0], "seed_spread": X["lam"]["seed_spread"][0],
           "agreement": X["lam"]["agreement"][0],
           "residual": float(load_json(results_dir("p7a_metrics") / "summary.json")["lambda_R"])}
    for k, v in ref.items():
        if not np.isclose(lam10[k], v, rtol=1e-4):
            raise RuntimeError(f"{k}: threshold {lam10[k]} does not reproduce {v}")

    def boot(stat, sel, lv=0.95, ci=True):
        ob = np.array(sorted(set(O[sel])))
        rows = {o: np.nonzero(sel & (O == o))[0] for o in ob}
        def ev(draw):
            idx = np.concatenate([rows[o] for o in draw])
            return stat(idx, np.concatenate([np.full(len(rows[o]), j) for j, o in enumerate(draw)]))
        pt = float(ev(ob))
        if not ci:
            return pt, np.nan, np.nan
        rng = np.random.default_rng(1)
        bs = np.array([ev(rng.choice(ob, len(ob), replace=True)) for _ in range(B)], float)
        return pt, float(np.nanquantile(bs, (1 - lv) / 2)), float(np.nanquantile(bs, 1 - (1 - lv) / 2))

    grid, table = {}, []
    for eps in cfg["eps_grid"]:
        for alpha in cfg["alpha_grid"]:
            lam, out = cell(eps, alpha)
            risk = lambda k, e=eps, o=out: (lambda i, l: object_rate((o[k][0][i] > e).astype(float), l))  # noqa: E731
            cost = lambda k, o=out: (lambda i, l: object_rate(o[k][1][i], l) / c_ref)                     # noqa: E731
            res = {nm: {k: (boot(risk(k), sel, ci=False)[0], boot(cost(k), sel, ci=False)[0]) for k, _ in names}
                   for nm, sel in sets.items()}
            grid[(eps, alpha)] = (lam, out, res)
            for nm in sets:
                safe = [(c, k) for k, (r, c) in res[nm].items() if r <= alpha]
                best = min(safe)[1] if safe else None
                table.append((eps, alpha, nm, res[nm], best, lam))

    # pre-registered tests at eps = 0.05, alpha = 0.10, in distribution
    lv = 1 - cfg["family_alpha"] / 3
    eps0, a0 = 0.05, 0.10
    _, out0, res0 = grid[(eps0, a0)]
    idm = sets["in distribution"]
    rk = lambda k: (lambda i, l: object_rate((out0[k][0][i] > eps0).astype(float), l))                  # noqa: E731
    s1 = res0["in distribution"]["gated"][0] <= res0["in distribution"]["cert"][0] + 0.02
    s2 = boot(lambda i, l: object_rate(out0["cert"][1][i], l) - object_rate(out0["gated"][1][i], l), idm, lv)
    s3 = boot(lambda i, l: rk("gated")(i, l) - rk("coarse")(i, l), idm, lv)
    S = {"S1": bool(s1), "S2": s2[1] > 0, "S3": s3[2] < 0}

    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                                          # noqa: E731
    lab = dict(names)
    md = ["# Phase 7c: stricter and looser targets, every gate recalibrated (pre-registered Oct 17, 2026)\n",
          f"Fresh pool {len(A['E'])} queries (4m); calibration pool {len(C['E'])}. Thresholds at eps = alpha = 0.10 reproduce "
          "4m / 7a (checked). Risk is measured at the cell's eps; cost is relative to always simulating. "
          f"Tests at eps = 0.05, alpha = 0.10, in distribution; CIs {lv:.4f} two-sided (Bonferroni over 3).\n",
          f"- **S1** risk(gated) {res0['in distribution']['gated'][0]:.3f} vs certify-or-simulate "
          f"{res0['in distribution']['cert'][0]:.3f} (+0.02 allowed): **{'PASS' if S['S1'] else 'FAIL'}**",
          f"- **S2** saving certify-or-simulate - gated (s/query): {f3(s2)}: **{'PASS' if S['S2'] else 'FAIL'}**",
          f"- **S3** risk(gated) - risk(coarse only): {f3(s3)}: **{'PASS' if S['S3'] else 'FAIL'}**", "",
          "## Every cell: risk / cost; bold = cheapest pipeline with risk <= alpha", ""]
    for nm in sets:
        md += [f"### {nm}", "", "| eps | alpha | tau | " + " | ".join(l.strip("*") for _, l in names) + " |",
               "|---|---|---|" + "---|" * len(names)]
        for eps, alpha, s_nm, res, best, lam in table:
            if s_nm != nm:
                continue
            cells = []
            for k, _ in names:
                r, c = res[k]
                t = f"{r:.3f} / {c:.2f}"
                cells.append(f"**{t}**" if k == best else t)
            md.append(f"| {eps} | {alpha} | {lam['window']:.4g} | " + " | ".join(cells) + " |")
        md.append("")
    md.append("A threshold of -inf means too few bad calibration objects for that alpha: the gate rejects everything.")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({**S, "s2": s2, "s3": s3,
               "cells": [{"eps": e, "alpha": a, "set": nm, "lam": lam, "best": best,
                          "res": {k: list(v) for k, v in res.items()}} for e, a, nm, res, best, lam in table]},
              run.path("summary.json"))
    print(f"[p7c] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
