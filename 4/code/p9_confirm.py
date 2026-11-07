"""Phase 9 verdict, single objects: ours-v2 on fresh in-distribution and shifted queries. CPU.

Pre-registered Oct 19, 2026 (PLAN.md "Phase 9"), before any Phase 9 code ran or data existed.
ours-v2 = the one-window certificate (event window, coarse_cfl m = 4, tau = 0.02614) + the gated chain of 4k with the
repair suffix at M2 from an M1 shadow (job 89, p4j_allk with shadow/restart fidelities). Baselines on the same queries:
generate only, coarse only, always M1 / M2 / M3, always reference, certify-or-simulate (same one-window certificate,
reference fallback), momentum-residual gate (lambda_R from 7a), seed-spread deferral (lambda from 4m).
Tests (object-cluster bootstrap, 2,000 resamples; Bonferroni over five: 99% two-sided CIs):
  C1 ID risk(ours) <= risk(cert) + 0.02 and <= 0.035  C2 ID upper CI of mean cost(ours) <= 0.35 c_ref
  C3 ID saving cert - ours: lower bound > 0            C4 shift risk(ours) <= 0.03 (point) and risk(M1) - risk(ours) > 0
  C5 shift saving cert - ours: lower bound > 0
Also: the contact-scene ladder baselines (always M1/M2/M3, job 93) for the main table; the contact verdict itself is
p4n_summarize.py --config configs/p9c_summary.yaml.

Output: results/p9_confirm/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import spread_event  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402
from p4k_gated import load, pipelines  # noqa: E402
from p5a_amortise import light_query  # noqa: E402
from p7a_metrics import com_prediction, momentum_inputs, residual  # noqa: E402


def ladder(run, qids, c_ref, full, fids):
    rows = {(r["qid"], r["fidelity"]): r for f in sorted(results_dir(run).glob("shard_*/fullsim.jsonl")) for r in read_jsonl(f)}
    out = {}
    for fid in fids:
        e, c = [], []
        for q in qids:
            r = rows.get((q, fid))
            ok = r is not None and r.get("stable") and r.get("err_t") is not None and not r.get("crashed")
            e.append(float(np.mean(r["err_t"])) if ok else np.inf)
            c.append(c_ref * (r or {}).get("substeps", full) / full)
        out[fid] = (np.array(e), np.array(c))
    return out


def one_window(cfg, A):
    """A copy of the loaded rows A with the one-window certificate (as in pilots E2/E1b): score = the event window
    only; chain windows already run are not charged twice. Also used by p7b_summarize.py (Phase 10b)."""
    oi = ood_index_from(cfg["build_runs"]) if cfg.get("build_runs") else {}
    groot = gen_dir(cfg, False)
    a_fit, b_fit, _ = fit_probe_cost(cfg["cost_fit_scan_run"], "coarse_cfl", 4)
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == "coarse_cfl" and int(r["m"]) == 4}
    chain_starts = [0] + [2 * k + 1 for k in cfg["chain_k"]]
    root = Path(cfg["paths"]["data_root"])
    qpath = lambda q, sp: root / "testset" / f"{q}.h5" if sp == "test_id" else Path(oi[q])          # noqa: E731
    S1, CP1, CWg1 = [], [], []
    for q, sp in zip(A["qid"], A["split"]):
        r = scan[q]
        smp = np.load(groot / "base" / sp / f"{q}.npz")["samples"]
        ev = spread_event(smp, {**light_query(qpath(q, sp)), "qid": q}, 4)[0]
        w = {t: (np.inf if (bl or cr or x is None) else x) for t, x, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
        sub = dict(zip(r["starts"], r["substeps"]))
        S1.append(w[ev]); CP1.append(a_fit + b_fit * (sub.get(ev) or 0))
        CWg1.append([0.0 if t == ev else a_fit + b_fit * (sub.get(t) or 0) for t in chain_starts])
    A1 = dict(A); A1["S"], A1["CP"], A1["CWg"] = np.array(S1), np.array(CP1), np.array(CWg1)
    return A1


def assemble(cfg):
    """Per-query (error, cost) of every Phase 9 single-object pipeline P, the loaded rows A and the cost constants
    (also used by p9_table.py)."""
    tau, full = float(cfg["tau"]), float(cfg["full_substeps"])
    A, meta = load(cfg)
    c_ref, cg = meta["c_ref"], meta["cg"]
    oi = ood_index_from(cfg["build_runs"])
    groot = gen_dir(cfg, False)
    root = Path(cfg["paths"]["data_root"])
    qpath = lambda q, sp: root / "testset" / f"{q}.h5" if sp == "test_id" else Path(oi[q])          # noqa: E731
    A1 = one_window(cfg, A)
    R0 = [residual(np.load(groot / "base" / sp / f"{q}.npz")["samples"][0].astype(np.float64),
                   *com_prediction(momentum_inputs(qpath(q, sp)))) for q, sp in zip(A["qid"], A["split"])]
    A["S_one"] = A1["S"]                     # the one-window score, for p9_table.py's false-accept rates
    out, gs1, kk1 = pipelines(A1, tau, c_ref, cg)
    # ours-v2's decision per query, for the paper's qualitative figures (fig_paper_select.py)
    A["acc_one"], A["gs_one"], A["kk_one"] = A1["S"] < tau, gs1, kk1
    P = {"generate only": out["gen"], "certify or simulate (one window)": out["cert"], "ours-v2": out["gated"]}
    lad = ladder(cfg["full_run"], A["qid"], c_ref, full, ("coarse_cfl", "M1", "M2", "M3"))
    P["coarse only"] = lad["coarse_cfl"]
    for f in ("M1", "M2", "M3"):
        P[f"always {f}"] = lad[f]
    n = len(A["E"])
    P["always reference"] = (np.zeros(n), np.full(n, c_ref))
    lam_R = float(load_json(results_dir("p7a_metrics") / "summary.json")["lambda_R"])
    lam_ss = float(load_json(results_dir("p4m_confirm") / "summary.json")["lambda"]["seed_spread"])
    SS = np.array([float(r["seed_spread"]) for r in [{x["qid"]: x for f in sorted(results_dir(cfg["probe_run"]).glob("shard_*/probes.jsonl"))
                                                         for x in read_jsonl(f) if x["model"] == "base"}[q] for q in A["qid"]]])
    acc_r, acc_s = np.array(R0) < lam_R, SS < lam_ss
    A["acc_r"], A["acc_s"] = acc_r, acc_s           # for p10c_decisions.py
    P["momentum-residual gate"] = (np.where(acc_r, A["E"], 0.0), cg + (~acc_r) * c_ref)
    P["seed-spread deferral"] = (np.where(acc_s, A["E"], 0.0), 8 * cg + (~acc_s) * c_ref)
    return P, A, c_ref, cg, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, tau, B, full = cfg["eps"], float(cfg["tau"]), cfg["n_boot"], float(cfg["full_substeps"])
    lv = 1 - cfg["family_alpha"] / 5
    P, A, c_ref, cg, n = assemble(cfg)

    O = A["obj"]
    shift = np.array([s.startswith(cfg.get("shift_prefix", "ood3_")) for s in A["split"]])
    sets = {"in distribution": ~shift, "shift": shift}

    def boot(stat, sel, ci=True):
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

    risk = lambda k: (lambda i, l: object_rate((P[k][0][i] > eps).astype(float), l))                 # noqa: E731
    cost = lambda k: (lambda i, l: object_rate(P[k][1][i], l))                                       # noqa: E731
    err = lambda k: (lambda i, l: object_rate(np.minimum(P[k][0][i], 10.0), l))                      # noqa: E731
    idm = sets["in distribution"]
    r_ours_id = boot(risk("ours-v2"), idm, False)[0]
    # amendment before data (Oct 19): C1 also needs an absolute bound, so a weakened comparator cannot carry it
    c1 = r_ours_id <= boot(risk("certify or simulate (one window)"), idm, False)[0] + 0.02 and r_ours_id <= 0.035
    c2v = boot(cost("ours-v2"), idm)
    c2 = c2v[2] <= 0.35 * c_ref
    c3v = boot(lambda i, l: cost("certify or simulate (one window)")(i, l) - cost("ours-v2")(i, l), idm)
    c3 = c3v[1] > 0
    r_ours_sh = boot(risk("ours-v2"), shift, False)[0]
    c4v = boot(lambda i, l: risk("always M1")(i, l) - risk("ours-v2")(i, l), shift)
    c4 = r_ours_sh <= 0.03 and c4v[1] > 0
    c5v = boot(lambda i, l: cost("certify or simulate (one window)")(i, l) - cost("ours-v2")(i, l), shift)
    c5 = c5v[1] > 0
    C = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"                                         # noqa: E731
    md = ["# Phase 9: ours-v2 confirmation on fresh single-object queries (pre-registered Oct 19, 2026)\n",
          f"Queries: {n} ({int((~shift).sum())} in distribution, {int(shift.sum())} shifted). tau = {tau}; eps = {eps}. "
          f"CIs: object-cluster bootstrap, {B} resamples, {lv:.2f} two-sided (Bonferroni over 5). Costs in A100 s "
          f"(reference {c_ref:.3f} s, generation {cg:.3f} s).\n",
          f"- **C1** ID risk: ours-v2 {boot(risk('ours-v2'), idm, False)[0]:.3f} vs certify-or-simulate "
          f"{boot(risk('certify or simulate (one window)'), idm, False)[0]:.3f} (+0.02) and <= 0.035: **{'PASS' if c1 else 'FAIL'}**",
          f"- **C2** ID cost: ours-v2 {c2v[0]:.3f} s [{c2v[1]:.3f}, {c2v[2]:.3f}] = {c2v[0] / c_ref:.2f}x reference; upper bound "
          f"<= 0.35 x c_ref ({0.35 * c_ref:.3f} s): **{'PASS' if c2 else 'FAIL'}**",
          f"- **C3** ID saving vs certify-or-simulate: {f3(c3v)} s: **{'PASS' if c3 else 'FAIL'}**",
          f"- **C4** shift: ours-v2 risk {r_ours_sh:.3f} (<= 0.03); risk(always M1) - risk(ours-v2) {f3(c4v)}: **{'PASS' if c4 else 'FAIL'}**",
          f"- **C5** shift saving vs certify-or-simulate: {f3(c5v)} s: **{'PASS' if c5 else 'FAIL'}**",
          f"- **Single-object part: {'CONFIRMED' if all(C.values()) else 'NOT CONFIRMED'}**", ""]
    for nm, sel in sets.items():
        md += [f"## {nm}", "", "| pipeline | mean error | risk | cost / always reference |", "|---|---|---|---|"]
        for k in P:
            md.append(f"| {'**' + k + '**' if k == 'ours-v2' else k} | {boot(err(k), sel, False)[0]:.4f} | "
                      f"{boot(risk(k), sel, False)[0]:.3f} | {boot(cost(k), sel, False)[0] / c_ref:.2f} |")
        md.append("")
    # contact-scene ladder baselines for the main table (descriptive)
    kinds = {e["qid"]: e.get("kind", "contact") for f in sorted(results_dir(cfg["contact_build_run"]).glob("shard_*/ood_manifest.json"))
             for e in load_json(f)}
    cq = sorted(q for q, k in kinds.items() if k == "contact")
    if cq:
        lc = ladder(cfg["contact_full_run"], cq, c_ref, full, ("M1", "M2", "M3"))
        md += ["## Contact scenes: tuned simulators (descriptive; query-level)", ""] + [
            f"- always {f}: risk {np.mean(lc[f][0] > eps):.3f}, cost {np.mean(lc[f][1]) / c_ref:.2f}x" for f in ("M1", "M2", "M3")]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({**{k: bool(v) for k, v in C.items()}, "c2": c2v, "c3": c3v, "c4": c4v, "c5": c5v,
               "confirmed_single": bool(all(C.values()))}, run.path("summary.json"))
    print(f"[p9] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
