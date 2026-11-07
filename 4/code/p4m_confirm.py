"""Phase 4m verdict: fresh shifted queries, the strongest baselines, placement at matched budget.
CPU only. Pre-registered Oct 14, 2026 (PLAN.md "Phase 4m"), before any 4m data existed.

Thresholds come from the calibration pool only (replicate-0 object split, single-draw
subsampling, eps = alpha = 0.10, the Phase 4a procedure):
  window (spread_event) tau (must reproduce 0.02614); base seed spread; uniform windows;
  cross-fidelity agreement a = max_k mean_i ||gen_k - coarse_k|| (bad = coarse error > eps,
  because XF outputs the coarse trajectory).
Pipelines on the fresh queries: gen, coarse, cert_ref, cert_ref_uniform, gated V2 (Phase 4k),
XF (accept the coarse rollout if a < lambda_a, else a reference rollout).
Tests (object-cluster bootstrap; shift = the 4 ood2_* splits pooled, ID = id_new_20/21):
  M1 FAR(window) among failing queries <= 0.10 and FAR(seed spread) - FAR(window) lower bound > 0
  M2 risk(gated V2) <= 0.10 and risk(coarse) - risk(gated V2) lower bound > 0
  M3 XF dominates gated V2 iff on BOTH sets risk(XF) <= risk(gated) + 0.02 and the cost saving
     gated - XF has lower bound > 0 (decision rule, reported either way)
  M4 cert_ref_uniform - cert_ref in cost and risk, 95% CIs (descriptive)
Amendment (Oct 14, before data): XF's lambda uses the pooled object-weighted rule on the whole pool;
secondary pipeline cert_ss (seed-spread deferral, charged 8 generations).

Output: results/p4m_confirm/{SUMMARY.md, summary.json}
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
from p4b_certificate_objects import object_rate, pooled_threshold  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402
from p4k_gated import load as load_fresh, pipelines  # noqa: E402


def agreement(gen0: np.ndarray, coarse: np.ndarray) -> float:
    """Largest per-frame mean distance between generated sample 0 and the coarse rollout."""
    return float(np.linalg.norm(gen0 - coarse, axis=2).mean(axis=1).max())


def calib_lambda(S, bad, O, objs_all, alpha):
    """The Phase 4a/4f replicate-0 procedure: object split by default_rng(0), one draw per bad object."""
    rng = np.random.default_rng(0)
    cal_o = set(rng.permutation(objs_all)[: len(objs_all) // 2])
    cal = np.array([o in cal_o for o in O])
    bad_c = cal & bad
    picks = [S[rng.choice(np.nonzero(bad_c & (O == o))[0])] for o in np.unique(O[bad_c])]
    return threshold(np.array(picks, float), alpha), len(picks)


def calibration_pool(cfg):
    """Calibration-pool scores in the Phase 4f row order (so the window tau reproduces). Also used by 7c."""
    spec, model = cfg["defect_spec"], cfg["model"]
    oi = ood_index_from(cfg["calib_build_runs"])
    root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    cdir = Path(cfg["paths"]["coarse_dir"])
    coarse_rows = {r["qid"]: r for f in sorted(results_dir(cfg["calib_coarse_run"]).glob("shard_*/allk.jsonl"))
                   for r in read_jsonl(f)}
    R = []
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
                q = query_inputs(r["qid"], r["split"], root, oi)
                samples = np.load(groot / model / r["split"] / f"{r['qid']}.npz")["samples"]
                win = {s: (np.inf if (bl or cr or raw is None) else raw)
                       for s, raw, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
                cr = coarse_rows.get(r["qid"])
                if cr is None or cr.get("crashed") or cr.get("coarse_blown"):
                    a_score, e_coarse = np.inf, np.inf
                else:
                    c = np.load(cdir / r["split"] / f"{r['qid']}.npz")["coarse"].astype(np.float64)
                    a_score, e_coarse = agreement(samples[0], c), float(np.mean(cr["coarse_err_t"]))
                R.append({"obj": r["qid"].split("__")[0], "qid": r["qid"], "split": r["split"],
                          "E": float(pr["err_mean"]), "SS": float(pr["seed_spread"]),
                          "S": max(win[s] for s in starts_for(spec["placement"], samples, q, spec["m"])),
                          "SU": max(win[s] for s in starts_for("uniform", samples, q, spec["m"])),
                          "AG": a_score, "EC": e_coarse})
    C = {k: np.array([r[k] for r in R], dtype=object if k in ("obj", "qid", "split") else float) for k in R[0]}
    for k in ("obj", "qid", "split"):
        C[k] = C[k].astype(str)
    return C


def calibration(cfg, a, b):
    eps = cfg["eps"]
    C = calibration_pool(cfg)
    objs = np.array(sorted(set(C["obj"])))
    lam = {}
    for name, score, bad in (("window", "S", C["E"] > eps), ("seed_spread", "SS", C["E"] > eps),
                             ("uniform", "SU", C["E"] > eps)):
        lam[name] = calib_lambda(C[score], bad, C["obj"], objs, cfg["alpha"])
    # Amendment (Oct 14, before any 4m data): XF uses the pooled object-weighted rule on the WHOLE pool,
    # because coarse failures are too rare in distribution for the replicate-0 subsample rule.
    b = C["EC"] > eps
    lam["agreement"] = (pooled_threshold(C["AG"][b], C["obj"][b], cfg["alpha"]), int(len(set(C["obj"][b]))))
    return lam, len(C["E"])


def assemble(cfg):
    """Fresh-pool scores, thresholds and per-query (error, cost) of every 4m pipeline (also used by Phase 7a)."""
    spec, model = cfg["defect_spec"], cfg["model"]
    a_fit, b_fit, _ = fit_probe_cost(cfg["cost_fit_scan_run"], spec["fidelity"], spec["m"])
    lam, n_cal = calibration(cfg, a_fit, b_fit)

    A, meta = load_fresh(cfg)                                    # 4k loader: costs, chain, V2 outcomes
    c_ref, cg = meta["c_ref"], meta["cg"]
    oi = ood_index_from(cfg["build_runs"])
    root, groot, cdir = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False), Path(cfg["paths"]["coarse_dir"])
    errs = {r["qid"]: r for f in sorted(results_dir(cfg["probe_run"]).glob("shard_*/probes.jsonl"))
            for r in read_jsonl(f) if r["model"] == model}
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["source"] == model and r["fidelity"] == spec["fidelity"] and int(r["m"]) == spec["m"]}
    allk = {r["qid"]: r for f in sorted(results_dir(cfg["run_name"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    SS, SU, CPU, AG = [], [], [], []
    for qid, sp in zip(A["qid"], A["split"]):
        q = query_inputs(qid, sp, root, oi)
        samples = np.load(groot / model / sp / f"{qid}.npz")["samples"]
        sr = scan[qid]
        win = {s: (np.inf if (bl or cr or raw is None) else raw) for s, raw, bl, cr in zip(sr["starts"], sr["raw"], sr["blown"], sr["crashed"])}
        sub = dict(zip(sr["starts"], sr["substeps"]))
        su = starts_for("uniform", samples, q, spec["m"])
        SS.append(float(errs[qid]["seed_spread"])); SU.append(max(win[s] for s in su))
        CPU.append(sum(a_fit + b_fit * (sub.get(s) or 0) for s in su))
        x = allk[qid]
        if x.get("crashed") or x.get("coarse_blown"):
            AG.append(np.inf)
        else:
            AG.append(agreement(samples[0], np.load(cdir / sp / f"{qid}.npz")["coarse"].astype(np.float64)))
    SS, SU, CPU, AG = map(np.array, (SS, SU, CPU, AG))

    tau = lam["window"][0]
    out, gs, kk = pipelines(A, tau, c_ref, cg)
    Au = dict(A); Au["S"], Au["CP"] = SU, CPU
    out["cert_uniform"] = pipelines(Au, lam["uniform"][0], c_ref, cg)[0]["cert"]
    acc_x = AG < lam["agreement"][0]
    out["xf"] = (np.where(acc_x, A["CE"], 0.0), cg + A["CC"] + (~acc_x) * c_ref)
    acc_ss = SS < lam["seed_spread"][0]                          # secondary: conformalised seed-spread deferral
    out["cert_ss"] = (np.where(acc_ss, A["E"], 0.0), 8 * cg + (~acc_ss) * c_ref)
    return {"A": A, "meta": meta, "lam": lam, "n_cal": n_cal, "SS": SS, "SU": SU, "AG": AG, "out": out, "tau": tau,
            "gs": gs, "kk": kk, "acc_x": acc_x, "acc_ss": acc_ss}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p4m_confirm"), cfg, synthetic=False)
    eps, B = cfg["eps"], cfg["n_boot"]
    X = assemble(cfg)
    A, meta, lam, n_cal, SS, out, tau = X["A"], X["meta"], X["lam"], X["n_cal"], X["SS"], X["out"], X["tau"]
    c_ref, cg = meta["c_ref"], meta["cg"]
    names = (("gen", "generate only"), ("coarse", "coarse_cfl only"), ("cert", "certify or simulate"),
             ("cert_uniform", "certify or simulate, uniform windows"), ("gated", "**gated V2**"),
             ("xf", "XF: accept coarse if it agrees with the generation"),
             ("cert_ss", "seed-spread deferral (8 samples)"))
    O = A["obj"]
    shift = np.array([s.startswith("ood2_") for s in A["split"]])
    idm = ~shift
    bad = A["E"] > eps

    def boot(stat, sel, ci=True):
        """stat(idx, labels) on rows idx with object labels (duplicates relabelled); point + 95% CI."""
        ob = np.array(sorted(set(O[sel])))
        rows = {o: np.nonzero(sel & (O == o))[0] for o in ob}
        def ev(draw):
            idx = np.concatenate([rows[o] for o in draw])
            lab = np.concatenate([np.full(len(rows[o]), i) for i, o in enumerate(draw)])
            return stat(idx, lab)
        point = ev(ob)
        if not ci:
            return float(point), np.nan, np.nan
        rng = np.random.default_rng(1)
        bs = np.array([ev(rng.choice(ob, len(ob), replace=True)) for _ in range(B)], float)
        return float(point), float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))

    risk = lambda k: (lambda i, l: object_rate((out[k][0][i] > eps).astype(float), l))          # noqa: E731
    cost = lambda k: (lambda i, l: object_rate(out[k][1][i], l))                                # noqa: E731
    err = lambda k: (lambda i, l: object_rate(np.minimum(out[k][0][i], 10.0), l))               # noqa: E731

    def far(acc):
        def f(i, l):
            m = bad[i]
            return object_rate(acc[i][m].astype(float), l[m]) if m.any() else np.nan
        return f
    acc_w, acc_s = A["S"] < tau, SS < lam["seed_spread"][0]
    m1_w = boot(far(acc_w), shift)
    m1_d = boot(lambda i, l: far(acc_s)(i, l) - far(acc_w)(i, l), shift)
    M1 = m1_w[0] <= 0.10 and m1_d[1] > 0
    m2_g = boot(risk("gated"), shift)
    m2_d = boot(lambda i, l: risk("coarse")(i, l) - risk("gated")(i, l), shift)
    M2 = m2_g[0] <= 0.10 and m2_d[1] > 0
    m3 = {}
    for nm, sel in (("shift", shift), ("in distribution", idm)):
        rk = (boot(risk("xf"), sel), boot(risk("gated"), sel))
        sv = boot(lambda i, l: cost("gated")(i, l) - cost("xf")(i, l), sel)
        m3[nm] = (rk, sv, rk[0][0] <= rk[1][0] + 0.02 and sv[1] > 0)
    M3 = all(v[2] for v in m3.values())
    m4 = {nm: (boot(lambda i, l: cost("cert_uniform")(i, l) - cost("cert")(i, l), sel),
               boot(lambda i, l: risk("cert_uniform")(i, l) - risk("cert")(i, l), sel))
          for nm, sel in (("shift", shift), ("in distribution", idm))}

    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                                    # noqa: E731
    md = ["# Phase 4m: fresh shifted queries, strongest baselines, placement at matched budget (pre-registered Oct 14, 2026)\n",
          f"Fresh queries: {meta['n']} ({int(shift.sum())} shifted in 4 families, {int(idm.sum())} in distribution). "
          f"Calibration pool: {n_cal} queries. Costs in A100 s/query (reference {c_ref:.3f}, generation {cg:.3f}). "
          f"Blown restarts: {meta['n_blown']}; crashed: {meta['n_crash']}.\n",
          "Thresholds (replicate-0 calibration; bad objects picked): " + ", ".join(
              f"{k} {v[0]:.5g} ({v[1]})" for k, v in lam.items()) + f". Window tau reproduces 0.02614: {abs(tau - 0.02614) < 5e-5}.\n",
          f"- **M1 certificate under shift:** FAR(window) {f3(m1_w)} (<= 0.10); FAR(seed spread) - FAR(window) {f3(m1_d)}: "
          f"**{'PASS' if M1 else 'FAIL'}**",
          f"- **M2 pipelines under shift:** risk(gated V2) {f3(m2_g)} (<= 0.10); risk(coarse) - risk(gated V2) {f3(m2_d)}: "
          f"**{'PASS' if M2 else 'FAIL'}**",
          f"- **M3 agreement baseline:** XF {'DOMINATES' if M3 else 'does not dominate'} gated V2. " + "; ".join(
              f"{nm}: risk XF {v[0][0][0]:.3f} vs gated {v[0][1][0]:.3f}, saving gated - XF {f3(v[1])}" for nm, v in m3.items()),
          "- **M4 placement at matched budget (descriptive):** " + "; ".join(
              f"{nm}: cost uniform - spread {f3(v[0])} s, risk {f3(v[1])}" for nm, v in m4.items()), ""]
    for nm, sel in (("Shifted (pooled)", shift), ("In distribution", idm)):
        md += [f"## {nm}", "", "| pipeline | mean error | risk | cost / always-reference |", "|---|---|---|---|"]
        md += [f"| {lab} | {boot(err(k), sel, False)[0]:.4f} | {boot(risk(k), sel, False)[0]:.3f} | {boot(cost(k), sel, False)[0] / c_ref:.2f} |"
               for k, lab in names] + [""]
    md += ["## Per split", "", "| split | n | gen risk | coarse risk | cert risk / cost | gated risk / cost | XF risk / cost | FAR window | FAR seed spread |",
           "|---|---|---|---|---|---|---|---|---|"]
    for sp in sorted(set(A["split"])):
        sel = A["split"] == sp
        r_ = {k: boot(risk(k), sel, False)[0] for k, _ in names}
        c_ = {k: boot(cost(k), sel, False)[0] / c_ref for k, _ in names}
        md.append(f"| {sp} | {sel.sum()} | {r_['gen']:.2f} | {r_['coarse']:.2f} | {r_['cert']:.2f} / {c_['cert']:.2f} | "
                  f"{r_['gated']:.2f} / {c_['gated']:.2f} | {r_['xf']:.2f} / {c_['xf']:.2f} | "
                  f"{far(acc_w)(np.nonzero(sel)[0], O[sel]) if (sel & bad).any() else float('nan'):.2f} | "
                  f"{far(acc_s)(np.nonzero(sel)[0], O[sel]) if (sel & bad).any() else float('nan'):.2f} |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"M1": bool(M1), "M2": bool(M2), "M3_xf_dominates": bool(M3), "lambda": {k: v[0] for k, v in lam.items()},
               "m1": [m1_w, m1_d], "m2": [m2_g, m2_d], "m3": {k: [v[0], v[1]] for k, v in m3.items()},
               "m4": {k: list(v) for k, v in m4.items()}, **{k: v for k, v in meta.items() if k != "splits"}},
              run.path("summary.json"))
    print(f"[p4m_confirm] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
