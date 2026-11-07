"""Phase 7a: reviewer metrics and literature baselines on the Phase 4m fresh pool. CPU only.

Pre-registered Oct 17, 2026 (PLAN.md "Phase 7a"), before any 7a code ran.
  B1 momentum residual (ANCHOR-style, no simulator): centre-of-mass drift of generated sample 0 from the
     trajectory momentum balance predicts, R = max_j |c_gen(t_j) - c_hat(t_j)|; gate resid_ref (R < lambda_R
     -> accept, else reference rollout). lambda_R: replicate-0 rule on the 4m calibration pool.
  B2 best-of-8 over the stored samples: medoid, least residual, oracle (upper bound, not a method).
Metrics: AUROC and AURC (risk-coverage) for window / seed spread / residual / agreement scores; risk at
eps in {0.05, 0.10, 0.20}; p90 / p99 / max of the per-query mean error; mean last-frame error.
Tests (Bonferroni over four, 98.75% two-sided object-cluster bootstrap CIs):
  Q1 AURC(window) - AURC(residual) < 0 (pooled)    Q2 AURC(window) - AURC(seed spread) < 0 (pooled)
  Q3 risk(resid_ref) - risk(gated V2) > 0 (shift)  Q4 risk(best-of-8 by residual) - risk(gated V2) > 0 (shift)

Output: results/p7a_metrics/{SUMMARY.md, summary.json, risk_coverage.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402
from p4m_confirm import assemble, calib_lambda  # noqa: E402

RHO, G = 1000.0, np.array([0.0, -9.8, 0.0])
FRAME_DT = 417 * 1e-4                     # simulated length of one stored frame (bridge guard 5)


def momentum_inputs(path) -> dict:
    """Frame 0 state and the external-force inputs; the ground truth only when asked (fresh pool)."""
    with h5py.File(path, "r") as h:
        d = {"x0": np.asarray(h["x"][0], np.float64), "v0": np.asarray(h["v"][0], np.float64),
             "vol": np.asarray(h["vol"], np.float64).reshape(-1),
             "F": np.asarray(h["drag_force"], np.float64).reshape(-1, 3) if "drag_force" in h else np.zeros((0, 3)),
             "mask": np.asarray(h["drag_mask"]).reshape(-1, len(np.asarray(h["vol"]).reshape(-1)))
             if "drag_force" in h else np.zeros((0, 0)),
             "gravity": int(np.asarray(h["gravity"])) if "gravity" in h else 0}
    return d


def com_prediction(d: dict) -> np.ndarray:
    """Centre of mass at the 24 model frames from momentum balance (internal forces cancel; no contacts)."""
    w = d["vol"] / d["vol"].sum()
    c0, u0 = w @ d["x0"], w @ d["v0"]
    a = (d["mask"].sum(axis=1) @ d["F"]) / (RHO * d["vol"].sum()) if len(d["F"]) else np.zeros(3)
    a = a + G * d["gravity"]
    t = (2 * np.arange(24) + 1) * FRAME_DT
    return c0 + u0 * t[:, None] + 0.5 * a * t[:, None] ** 2, w


def residual(traj: np.ndarray, chat: np.ndarray, w: np.ndarray) -> float:
    """traj (24, N, 3): largest distance between its centre of mass and the momentum-balance prediction."""
    return float(np.linalg.norm(np.einsum("n,tnc->tc", w, traj) - chat, axis=1).max())


def aurc(score, bad, perm):
    """Mean selective risk over coverages k = 1..n; ties broken by the fixed permutation; +inf ranks last."""
    s = np.where(np.isfinite(score), score, np.inf)
    order = np.lexsort((perm, s))
    return float(np.mean(np.cumsum(bad[order]) / np.arange(1, len(s) + 1)))


def rc_curve(score, bad, perm):
    s = np.where(np.isfinite(score), score, np.inf)
    order = np.lexsort((perm, s))
    return (np.cumsum(bad[order]) / np.arange(1, len(s) + 1)).tolist()


def auroc(score, bad):
    s = np.where(np.isfinite(score), score, 1e12)
    return float(roc_auc_score(bad, s)) if 0 < bad.sum() < len(bad) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p7a_metrics"), cfg, synthetic=False)
    eps, B, model = cfg["eps"], cfg["n_boot"], cfg["model"]
    level = 1 - cfg["family_alpha"] / cfg["n_tests"]                 # 0.9875
    X = assemble(cfg)
    A, meta, out, lam = X["A"], X["meta"], X["out"], X["lam"]
    c_ref, cg = meta["c_ref"], meta["cg"]
    groot = gen_dir(cfg, False)

    # --- B1 threshold on the calibration pool (replicate-0 rule, bad = sample-0 error > eps) ---
    oi_cal = ood_index_from(cfg["calib_build_runs"])
    cal_err = {}
    for run_name in cfg["calib_probe_runs"]:
        for f in sorted(results_dir(run_name).glob("shard_*/probes.jsonl")):
            for r in read_jsonl(f):
                if r["model"] == model:
                    cal_err[r["qid"]] = (r["split"], float(r["err_mean"]))
    RC, EC, OC, gt_chk = [], [], [], []
    for qid in sorted(cal_err):
        sp, e = cal_err[qid]
        d = momentum_inputs(oi_cal[qid])
        chat, w = com_prediction(d)
        RC.append(residual(np.load(groot / model / sp / f"{qid}.npz")["samples"][0].astype(np.float64), chat, w))
        EC.append(e); OC.append(qid.split("__")[0])
        if len(gt_chk) < 40 and not d["gravity"]:
            with h5py.File(oi_cal[qid], "r") as h:
                gt_chk.append(residual(np.asarray(h["x"][1::2][:24], np.float64), chat, w))
    RC, EC, OC = np.array(RC), np.array(EC), np.array(OC)
    lam_R, n_bad_R = calib_lambda(RC, EC > eps, OC, np.array(sorted(set(OC))), cfg["alpha"])

    # --- fresh pool: residual of every sample, true error of every sample, medoid, last-frame errors ---
    oi = ood_index_from(cfg["build_runs"])
    allk = {r["qid"]: r for f in sorted(results_dir(cfg["run_name"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    R0, Rall, Eall, med, last = [], [], [], [], {k: [] for k in ("gen", "coarse", "ek")}
    for qid, sp in zip(A["qid"], A["split"]):
        d = momentum_inputs(oi[qid])
        with h5py.File(oi[qid], "r") as h:
            gt = np.asarray(h["x"][1::2][:24], np.float64)
        s = np.load(groot / model / sp / f"{qid}.npz")["samples"].astype(np.float64)
        chat, w = com_prediction(d)
        r = np.array([residual(x, chat, w) for x in s])
        e = np.linalg.norm(s - gt, axis=3).mean(axis=(1, 2))
        D = np.array([[np.linalg.norm(a - b, axis=2).mean() for b in s] for a in s])
        R0.append(r[0]); Rall.append(r); Eall.append(e); med.append(int(D.sum(axis=1).argmin()))
        x = allk[qid]
        last["gen"].append(float(x["gen_err_t"][-1]))
        last["coarse"].append(np.inf if (x.get("crashed") or x.get("coarse_blown")) else float(x["coarse_err_t"][-1]))
        last["ek"].append([0.0 if (x.get("crashed") or x["k"][str(k)]["blown"] or x["k"][str(k)]["shadow_blown"])
                           else float(x["k"][str(k)]["err_t"][-1]) for k in cfg["chain_k"]])
    R0, Rall, Eall, med = np.array(R0), np.array(Rall), np.array(Eall), np.array(med)
    if np.max(np.abs(Eall[:, 0] - A["E"])) > 1e-3:
        raise RuntimeError(f"sample-0 error disagrees with the probe rows by {np.max(np.abs(Eall[:, 0] - A['E'])):.2e}")
    n = len(R0)
    i = np.arange(n)
    acc_r = R0 < lam_R
    out["resid_ref"] = (np.where(acc_r, A["E"], 0.0), cg + (~acc_r) * c_ref)
    out["bo8_medoid"] = (Eall[i, med], np.full(n, 8 * cg))
    out["bo8_resid"] = (Eall[i, Rall.argmin(axis=1)], np.full(n, 8 * cg))
    out["bo8_oracle"] = (Eall.min(axis=1), np.full(n, 8 * cg))

    # last-frame error of each pipeline (0 where the output is the reference rollout)
    LG, LC, LK = np.array(last["gen"]), np.array(last["coarse"]), np.array(last["ek"])
    acc_w = A["S"] < X["tau"]
    lastf = {"gen": LG, "coarse": LC, "cert": np.where(acc_w, LG, 0.0),
             "gated": np.where(acc_w, LG, np.where(X["gs"], LK[i, X["kk"]], 0.0)),
             "xf": np.where(X["acc_x"], LC, 0.0), "cert_ss": np.where(X["acc_ss"], LG, 0.0),
             "resid_ref": np.where(acc_r, LG, 0.0)}

    O = A["obj"]
    shift = np.array([s.startswith("ood2_") for s in A["split"]])
    sets = {"pooled": np.ones(n, bool), "in distribution": ~shift, "shift": shift}
    bad = A["E"] > eps
    perm = np.random.default_rng(0).permutation(n)

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

    scores = {"window (ours)": A["S"], "seed spread": X["SS"], "momentum residual": R0,
              "agreement (needs coarse rollout)": X["AG"]}
    sel_rows, curves = [], {}
    for nm, sel in sets.items():
        for sn, sc in scores.items():
            sel_rows.append((nm, sn, auroc(sc[sel], bad[sel]), aurc(sc[sel], bad[sel], perm[sel])))
            curves[f"{nm}|{sn}"] = rc_curve(sc[sel], bad[sel], perm[sel])
    pooled = sets["pooled"]
    d_aurc = lambda a_, b_: (lambda idx, _l: aurc(a_[idx], bad[idx], perm[idx]) - aurc(b_[idx], bad[idx], perm[idx]))  # noqa: E731
    q1 = boot(d_aurc(A["S"], R0), pooled, level)
    q2 = boot(d_aurc(A["S"], X["SS"]), pooled, level)
    risk = lambda k, ep=eps: (lambda idx, l: object_rate((out[k][0][idx] > ep).astype(float), l))  # noqa: E731
    q3 = boot(lambda idx, l: risk("resid_ref")(idx, l) - risk("gated")(idx, l), shift, level)
    q4 = boot(lambda idx, l: risk("bo8_resid")(idx, l) - risk("gated")(idx, l), shift, level)
    Q = {"Q1": q1[2] < 0, "Q2": q2[2] < 0, "Q3": q3[1] > 0, "Q4": q4[1] > 0}

    names = (("gen", "PhysCtrl (generate only)"), ("coarse", "coarse simulation only"),
             ("cert_ss", "seed-spread deferral (8 samples)"), ("resid_ref", "momentum-residual gate (ANCHOR-style)"),
             ("bo8_medoid", "best-of-8, medoid"), ("bo8_resid", "best-of-8, least residual"),
             ("bo8_oracle", "best-of-8, oracle (upper bound)"), ("xf", "accept coarse if it agrees (XF)"),
             ("cert", "certify or simulate"), ("gated", "**ours: certify + repair (gated V2)**"))
    f4 = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"                                           # noqa: E731
    md = ["# Phase 7a: reviewer metrics and literature baselines (pre-registered Oct 17, 2026)\n",
          f"Fresh pool: {n} queries ({int(shift.sum())} shifted, {int((~shift).sum())} in distribution). eps = {eps}. "
          f"CIs: object-cluster bootstrap, {B} resamples, {level:.4f} two-sided (Bonferroni over {cfg['n_tests']}).\n",
          f"Momentum-residual threshold lambda_R = {lam_R:.5g} (calibration pool, {len(RC)} queries, {n_bad_R} bad objects picked). "
          f"Unit check: residual of the ground truth on {len(gt_chk)} gravity-free calibration queries: median {np.median(gt_chk):.2e}, "
          f"max {np.max(gt_chk):.2e} (≈ 0 means the force model matches the simulator).\n",
          f"- **Q1** AURC(window) - AURC(residual), pooled: {f4(q1)}: **{'PASS' if Q['Q1'] else 'FAIL'}**",
          f"- **Q2** AURC(window) - AURC(seed spread), pooled: {f4(q2)}: **{'PASS' if Q['Q2'] else 'FAIL'}**",
          f"- **Q3** risk(residual gate) - risk(gated V2), shift: {f4(q3)}: **{'PASS' if Q['Q3'] else 'FAIL'}**",
          f"- **Q4** risk(best-of-8 by residual) - risk(gated V2), shift: {f4(q4)}: **{'PASS' if Q['Q4'] else 'FAIL'}**", "",
          "## Selective prediction (bad = sample-0 error > 0.10; lower AURC is better)", "",
          "| set | score | AUROC | AURC |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c:.3f} | {d:.4f} |" for a, b, c, d in sel_rows]
    for nm in ("in distribution", "shift"):
        sel = sets[nm]
        md += ["", f"## Pipelines, {nm}", "",
               "| pipeline | risk 0.05 | risk 0.10 | risk 0.20 | mean error | p90 | p99 | max | last-frame error | cost / always-simulate |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for k, lab in names:
            e = out[k][0][sel]
            ef = np.minimum(e, 10.0)
            lf = f"{np.mean(np.minimum(lastf[k][sel], 10.0)):.4f}" if k in lastf else "—"
            md.append(f"| {lab} | " + " | ".join(f"{boot(risk(k, ep), sel, ci=False)[0]:.3f}" for ep in (0.05, 0.10, 0.20))
                      + f" | {boot(lambda idx, l: object_rate(np.minimum(out[k][0][idx], 10.0), l), sel, ci=False)[0]:.4f}"
                      + f" | {np.quantile(ef, 0.9):.4f} | {np.quantile(ef, 0.99):.4f} | {ef.max():.4f} | {lf}"
                      + f" | {boot(lambda idx, l: object_rate(out[k][1][idx], l), sel, ci=False)[0] / c_ref:.2f} |")
    md += ["", "Notes: risk and mean error are object-weighted (as in 4m); percentiles, maximum and AURC are query-level. "
           "Oracle best-of-8 picks the sample with the least true error; it is an upper bound for any selection rule, not a method. "
           "Seed-spread deferral and best-of-8 are charged 8 generations."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({**{k: bool(v) for k, v in Q.items()}, "q": {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}, "lambda_R": lam_R,
               "selective": sel_rows, "gt_residual_check": [float(np.median(gt_chk)), float(np.max(gt_chk))]},
              run.path("summary.json"))
    save_json(curves, run.path("risk_coverage.json"))
    print(f"[p7a] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
