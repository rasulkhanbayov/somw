"""Pilot E1 (+ E3 latency) analysis: the stiff-material regime (EXPLORATORY). CPU, after jobs 81-83 and probe errors.

Defined Oct 18, 2026 in PLAN.md "Pilots E1-E3" before any E-pilot code ran.
Queries stiff_0/1 (E = 10^U(7.5, 9.0)); ground truth = the CFL-safe reference (reference_cfl 0.3, 125 grid).
Pipelines (A100 seconds, the 4k cost model):
  always simulate at M1 / M2 / M3 / the stiff reference; generate only;
  ours = the gated pipeline at tau = 0.02614 (p4k_gated.pipelines) with the 8a-B2 repair (M1 shadow, M2 suffix).
S* = the cheapest of M1/M2/M3 with risk <= 0.02 on the pooled E >= 1e8 queries, else the reference.
E1 promising if (pooled E >= 1e8): cost(S*) >= 0.5 x cost(stiff reference), and ours has risk <= 0.02 at
<= 0.5 x cost(S*).
E3 (idealised): latency = generation + one certificate window (windows concurrent) + for rejected queries one
chain window + the serial repair (shadow up to k, then the suffix); promising if median <= 0.5 x S*'s serial time.

Output: results/p8e_stiff/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import spread_event  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402
from p5a_amortise import light_query  # noqa: E402
from p4k_gated import load, pipelines  # noqa: E402
from verifier import resolve_fidelity  # noqa: E402

FRAME_DT = 0.0417


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, tau, full = cfg["eps"], float(cfg["tau"]), float(cfg["full_substeps"])
    A, meta = load(cfg)
    c_ref, cg = meta["c_ref"], meta["cg"]
    out, gs, kk = pipelines(A, tau, c_ref, cg)
    acc = A["S"] < tau
    oi = ood_index_from(cfg["build_runs"])
    E, ref_cost = [], []
    for q in A["qid"]:
        with h5py.File(oi[q], "r") as h:
            e, nu = float(np.asarray(h["E"])), float(np.asarray(h["nu"]))
        dt = resolve_fidelity({"substep_dt": 1e-4, "n_grid": 125, "cfl_target": float(cfg["reference_cfl"])},
                              {"E": e, "nu": nu})["substep_dt"]
        E.append(np.log10(e)); ref_cost.append(c_ref * 48 * int(FRAME_DT / dt) / full)
    E, ref_cost = np.array(E), np.array(ref_cost)
    lad = {r["qid"] + "|" + r["fidelity"]: r for f in sorted(results_dir(cfg["full_run"]).glob("shard_*/fullsim.jsonl"))
           for r in read_jsonl(f)}
    L3, L1 = cfg.get("label_three", "ours (gated, M2 repair)"), cfg.get("label_one", "ours, one-window certificate (secondary)")
    P = {"generate only": out["gen"], L3: out["gated"], "stiff reference": (np.zeros(len(E)), ref_cost)}
    for fid in ("M1", "M2", "M3"):
        e_, c_ = [], []
        for q in A["qid"]:
            r = lad.get(f"{q}|{fid}")
            ok = r is not None and r.get("stable") and r.get("err_t") is not None and not r.get("crashed")
            e_.append(float(np.mean(r["err_t"])) if ok else np.inf)
            c_.append(c_ref * (r or {}).get("substeps", full) / full)
        P[f"always {fid}"] = (np.array(e_), np.array(c_))
    hi = E >= 8.0
    risk = lambda k, sel: float(np.mean(P[k][0][sel] > eps))                                        # noqa: E731
    cost = lambda k, sel: float(np.mean(P[k][1][sel]))                                              # noqa: E731
    cands = [k for k in ("always M1", "always M2", "always M3") if risk(k, hi) <= 0.02]
    s_star = min(cands, key=lambda k: cost(k, hi)) if cands else "stiff reference"
    # Secondary (added after pilot E2, before any E1 data): the one-window certificate (event window only)
    a_fit, b_fit, _ = fit_probe_cost(cfg["cost_fit_scan_run"], "coarse_cfl", 4)
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == "coarse_cfl" and int(r["m"]) == 4}
    groot = gen_dir(cfg, False)
    chain_starts = [0] + [2 * k + 1 for k in cfg["chain_k"]]
    S1, CP1, CWg1 = [], [], []
    for q, sp in zip(A["qid"], A["split"]):
        r = scan[q]
        smp = np.load(groot / "base" / sp / f"{q}.npz")["samples"]
        ev = spread_event(smp, {**light_query(oi[q]), "qid": q}, 4)[0]
        w = {t: (np.inf if (bl or cr or x is None) else x) for t, x, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
        sub = dict(zip(r["starts"], r["substeps"]))
        S1.append(w[ev]); CP1.append(a_fit + b_fit * (sub.get(ev) or 0))
        CWg1.append([0.0 if t == ev else a_fit + b_fit * (sub.get(t) or 0) for t in chain_starts])
    A1 = dict(A); A1["S"], A1["CP"], A1["CWg"] = np.array(S1), np.array(CP1), np.array(CWg1)
    out1, gs1, kk1 = pipelines(A1, tau, c_ref, cg)
    P[L1] = out1["gated"]
    # Pilot E1b (Oct 19, 2026): the primary certificate is configurable; the default reproduces E1 (three windows)
    one = cfg.get("primary_windows", "three") == "one"
    LP = L1 if one else L3
    Ap, accp, gsp, kkp, ncert = (A1, A1["S"] < tau, gs1, kk1, 1) if one else (A, acc, gs, kk, 3)
    r1 = cost(s_star, hi) >= 0.5 * cost("stiff reference", hi)
    r2 = risk(LP, hi) <= 0.02 and cost(LP, hi) <= 0.5 * cost(s_star, hi)
    # E3 idealised latency: generation + one certificate window (concurrent) + for rejected queries one chain window
    # and the serial repair
    i = np.arange(len(E))
    lat = cg + Ap["CP"] / ncert + np.where(~accp, np.max(Ap["CW"], axis=1), 0.0) \
        + np.where(accp, 0.0, np.where(gsp, Ap["CK"][i, kkp], c_ref))
    lat_ok = float(np.median(lat[hi])) <= 0.5 * cost(s_star, hi)
    bands = [(7.5, 8.0), (8.0, 8.5), (8.5, 9.01)]
    md = ["# " + cfg.get("title", "Pilot E1: the stiff-material regime (EXPLORATORY, development pilot set; defined Oct 18, 2026)") + "\n",
          f"Queries: {len(E)} (log10 E {E.min():.2f}-{E.max():.2f}); E >= 1e8: {int(hi.sum())}. tau = {tau}. "
          f"Accepted by the certificate: {accp.mean():.2f}. Costs in A100 s (reference at PhysCtrl's setting = {c_ref:.3f} s).\n",
          "| pipeline | " + " | ".join(f"log10 E [{a}, {min(b, 9.0)}) risk / cost" for a, b in bands) + " | E >= 1e8: risk / cost |",
          "|---|" + "---|" * (len(bands) + 1)]
    for k in P:
        cells = []
        for a, b in bands:
            sel = (E >= a) & (E < b)
            cells.append(f"{risk(k, sel):.3f} / {cost(k, sel):.2f} s" if sel.any() else "—")
        md.append(f"| {k} | " + " | ".join(cells) + f" | {risk(k, hi):.3f} / {cost(k, hi):.2f} s |")
    md += ["", f"S* (cheapest of M1/M2/M3 with risk <= 0.02 on E >= 1e8) = **{s_star}**.", "",
           "## Pre-stated rules (E >= 1e8)", "",
           f"- (i) cost(S*) {cost(s_star, hi):.2f} s >= 0.5 x stiff reference {cost('stiff reference', hi):.2f} s: **{'yes' if r1 else 'no'}**",
           f"- (ii) ours risk {risk(LP, hi):.3f} <= 0.02 and cost {cost(LP, hi):.2f} s "
           f"<= 0.5 x S* ({0.5 * cost(s_star, hi):.2f} s): **{'yes' if r2 else 'no'}**",
           f"- **{cfg.get('pilot_name', 'E1')} promising: {'YES' if (r1 and r2) else 'NO'}**",
           f"- E3 (idealised latency): median {np.median(lat[hi]):.2f} s vs 0.5 x S* serial {0.5 * cost(s_star, hi):.2f} s: "
           f"**{'promising' if lat_ok else 'not promising'}**",
           "", "Exploratory pilot; the stiff split is a pilot set, not a confirmation; latency assumes concurrent windows."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"s_star": s_star, "rule_i": bool(r1), "rule_ii": bool(r2), "promising": bool(r1 and r2), "latency_ok": bool(lat_ok),
               "pooled_hi": {k: [risk(k, hi), cost(k, hi)] for k in P}}, run.path("summary.json"))
    print(f"[p8e] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
