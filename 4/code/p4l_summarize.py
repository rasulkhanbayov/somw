"""Phase 4l verdict: the multi-object contact benchmark (H0-H3). CPU only.

Pre-registered Oct 15, 2026 (PLAN.md "Phase 4l", with its amendment), before any 4l data.
Contact scenes unless stated; cluster bootstrap over the first object of each scene; eps = 0.10;
fixed tau = 0.02614 on the max raw defect of the chosen windows.
  H0 median lift floor (gt source) at the contact window <= 0.25 x median base defect there
  H1 capture(contact_spread) - capture(uniform) and - capture(spread_event): 95% lower bounds > 0 (base)
  H2 FAR(contact_spread) among bad scenes <= 0.15 and FAR(uniform) - FAR(contact_spread) lower bound > 0
  H3 event-gated V2 (accept by contact_spread; else restart at k_e) vs cert_ref: risk <= +0.02 and
     saving lower bound > 0; timing: error(chain-timed) - error(event) and error(shuffled event)
     - error(event) lower bounds > 0
Costs as Phase 4k; generation charged twice (two objects).

Output: results/p4l_summary/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from backends import query_from_h5  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import contact_window, starts_for  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402
from p4d_pipeline import gen_costs  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402


def win_of(r):
    return {s: (np.inf if (b or c or raw is None) else raw) for s, raw, b, c in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}


def capture(win, chosen):
    m_all, m = max(win.values()), max(win[s] for s in chosen)
    if not np.isfinite(m_all):
        return 1.0 if not np.isfinite(m) else 0.0
    return m / m_all if m_all > 0 else 1.0


def err(e):
    m = float(np.mean(e)) if e is not None else np.inf
    return m if np.isfinite(m) else np.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p4l_summary"), cfg, synthetic=False)
    eps, tau, B, spec = cfg["eps"], float(cfg["tau"]), cfg["n_boot"], cfg["defect_spec"]
    m = spec["m"]
    gcost, c_ref = gen_costs()
    cg = 2 * gcost["base"]["best"]                                   # two objects generated
    share = lambda subs: c_ref * subs / float(cfg["full_substeps"])  # noqa: E731
    a_fit, b_fit, _ = fit_probe_cost(cfg["cost_fit_scan_run"], spec["fidelity"], m)
    rates = [r["rollout_s"] / r["substeps"] for f in sorted(results_dir(cfg["shadow_rate_run"]).glob("shard_*/fullsim.jsonl"))
             for r in read_jsonl(f) if r.get("fidelity") == "coarse_cfl" and r.get("stable") and r.get("substeps")
             and r.get("rollout_s") and r["split"].startswith("id_new_")]
    rate = float(np.median(rates))
    groot = gen_dir(cfg, False)

    man = [e for f in sorted(results_dir(cfg["build_run"]).glob("shard_*/ood_manifest.json")) for e in load_json(f)]
    design = [r for f in sorted(results_dir(cfg["build_run"]).glob("shard_*/design.jsonl")) for r in read_jsonl(f)]
    errs = {(r["qid"], r["model"]): r for f in sorted(results_dir(cfg["probe_run"]).glob("shard_*/probes.jsonl")) for r in read_jsonl(f)}
    scan = {(r["qid"], r["source"]): r for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["fidelity"] == spec["fidelity"] and int(r["m"]) == m}
    allk = {r["qid"]: r for f in sorted(results_dir(cfg["allk_run"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    chain_starts = [0] + [2 * k + 1 for k in cfg["chain_k"]]

    R = []
    for e in man:
        q = query_from_h5(e["path"], e["split"], e["qid"])
        samples = np.load(groot / "base" / e["split"] / f"{e['qid']}.npz")["samples"]
        wb, wg, wt = win_of(scan[(e["qid"], "base")]), win_of(scan[(e["qid"], "gt_super")]), win_of(scan[(e["qid"], "gt")])
        sub = dict(zip(scan[(e["qid"], "base")]["starts"], scan[(e["qid"], "base")]["substeps"]))
        wc = lambda s: a_fit + b_fit * (sub.get(s) or 0)                               # noqa: E731
        pl = {p: starts_for(p, samples, q, m) for p in ("contact_spread", "uniform", "spread_event")}
        cw = contact_window(samples, q, m)
        x = allk[e["qid"]]
        rk = {}
        for k, v in x.get("k", {}).items():
            c = rate * v["shadow_substeps"] + share(v["substeps"])
            rk[int(k)] = (0.0, c + c_ref) if (v["shadow_blown"] or v["blown"]) else (err(v["err_t"]), c)
        R.append({"qid": e["qid"], "obj": e["qid"].split("__")[0], "kind": e["kind"], "k_c": e["k_c"],
                  "E": float(errs[(e["qid"], "base")]["err_mean"]), "E_sup": float(errs[(e["qid"], "gt_super")]["err_mean"]),
                  "cap": {p: capture(wb, s) for p, s in pl.items()}, "cap_sup": {p: capture(wg, s) for p, s in pl.items()},
                  "score": {p: max(wb[s] for s in st) for p, st in pl.items()},
                  "probe": {p: sum(wc(s) for s in st) for p, st in pl.items()},
                  "cw": cw, "floor_cw": wt[cw] if cw is not None else np.nan, "def_cw": wb[cw] if cw is not None else np.nan,
                  "W": [wb[s] for s in chain_starts], "CWg": [0.0 if s in pl["contact_spread"] else wc(s) for s in chain_starts],
                  "k_e": x.get("k_e"), "k_g": x.get("k_g"), "rk": rk})
    con = [r for r in R if r["kind"] == "contact"]
    O = np.array([r["obj"] for r in con])

    def boot(vals):
        vals = np.asarray(vals, float)
        ob = np.array(sorted(set(O)))
        idx = {o: np.nonzero(O == o)[0] for o in ob}
        f = lambda draw: np.mean([vals[idx[o]].mean() for o in draw])                  # noqa: E731
        rng = np.random.default_rng(1)
        bs = [f(rng.choice(ob, len(ob), replace=True)) for _ in range(B)]
        return float(f(ob)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))

    def boot_far(acc, bad):
        """Object-weighted false-accept rate among bad scenes, with cluster bootstrap."""
        ob = np.array(sorted(set(O[bad])))
        idx = {o: np.nonzero((O == o) & bad)[0] for o in ob}
        f = lambda draw: np.mean([acc[idx[o]].mean() for o in draw])                   # noqa: E731
        rng = np.random.default_rng(1)
        return float(f(ob)), [f(rng.choice(ob, len(ob), replace=True)) for _ in range(B)] if len(ob) else []

    # H0
    ok0 = [r for r in con if r["cw"] is not None]
    fl, de = np.nanmedian([r["floor_cw"] for r in ok0]), np.nanmedian([r["def_cw"] for r in ok0])
    H0 = fl <= 0.25 * de
    # H1
    d_u = boot([r["cap"]["contact_spread"] - r["cap"]["uniform"] for r in con])
    d_s = boot([r["cap"]["contact_spread"] - r["cap"]["spread_event"] for r in con])
    H1 = d_u[1] > 0 and d_s[1] > 0
    d_u_sup = boot([r["cap_sup"]["contact_spread"] - r["cap_sup"]["uniform"] for r in con])
    # H2
    bad = np.array([r["E"] > eps for r in con])
    acc = {p: np.array([r["score"][p] < tau for r in con]) for p in ("contact_spread", "uniform", "spread_event")}
    far_c, bs_c = boot_far(acc["contact_spread"].astype(float), bad)
    far_u, bs_u = boot_far(acc["uniform"].astype(float), bad)
    dfar = np.array(bs_u) - np.array(bs_c)
    H2 = far_c <= 0.15 and len(dfar) and np.quantile(dfar, .025) > 0
    ctrl = [r for r in R if r["kind"] == "control"]
    ctrl_fr = float(np.mean([r["score"]["contact_spread"] >= tau for r in ctrl if r["E"] <= eps])) if ctrl else np.nan
    # H3
    ac = acc["contact_spread"]
    E = np.array([r["E"] for r in con])
    P = np.array([r["probe"]["contact_spread"] for r in con])

    def at(r, k):
        return r["rk"].get(k, (0.0, c_ref))                     # no rollout at k -> reference fallback
    ev = [at(r, r["k_e"]) if r["k_e"] is not None else (0.0, c_ref) for r in con]
    e_ev = np.where(ac, E, [v[0] for v in ev]); c_ev = cg + P + np.where(ac, 0.0, [v[1] for v in ev])
    e_cr = np.where(ac, E, 0.0); c_cr = cg + P + (~ac) * c_ref
    ch_e, ch_c = [], []
    for r in con:                                                # chain timing (4k): first failing chain window
        W = np.array(r["W"], float)
        fail = W >= tau
        j = int(np.argmax(fail)) if fail.any() else -1
        last = len(W) - 1 if j < 0 else j
        pg = float(np.cumsum(r["CWg"])[last])
        v = (0.0, c_ref) if j <= 0 else at(r, cfg["chain_k"][j - 1])
        ch_e.append(v[0]); ch_c.append(pg + v[1])
    e_ch = np.where(ac, E, ch_e); c_ch = cg + P + np.where(ac, 0.0, ch_c)
    rej_ke = np.array([(not a) and (r["k_e"] is not None) for a, r in zip(ac, con)])
    ke_pool = np.array([r["k_e"] for r, z in zip(con, rej_ke) if z])
    # exact expectation over permutations of k_e among the rejected scenes that have one
    e_sh = e_ev.copy()
    for i in np.nonzero(rej_ke)[0]:
        e_sh[i] = np.mean([min(at(con[i], int(k))[0], 10.0) for k in ke_pool])
    risk = lambda e: boot((np.asarray(e) > eps).astype(float))                          # noqa: E731
    cap10 = lambda e: np.minimum(np.asarray(e, float), 10.0)                            # noqa: E731
    r_ev, r_cr = risk(e_ev), risk(e_cr)
    sv = boot(c_cr - c_ev)
    H3a = r_ev[0] <= r_cr[0] + 0.02 and sv[1] > 0
    t_ch, t_sh = boot(cap10(e_ch) - cap10(e_ev)), boot(cap10(e_sh) - cap10(e_ev))
    H3b = t_ch[1] > 0 and t_sh[1] > 0

    f3 = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"                             # noqa: E731
    reasons = {}
    for d in design:
        if not d.get("valid"):
            reasons[d.get("reason", "?")] = reasons.get(d.get("reason", "?"), 0) + 1
    md = ["# Phase 4l: multi-object contact benchmark (pre-registered Oct 15, 2026)\n",
          f"Valid scenes: {len(R)} ({len(con)} contact, {len(ctrl)} control); designed {len(design)}; excluded: "
          + (", ".join(f"{k}: {v}" for k, v in reasons.items()) or "none") + f". tau = {tau}; eps = {eps}. "
          f"Costs in A100 s (reference {c_ref:.3f}; generation {cg:.3f} for two objects).\n",
          f"- **H0 lift floor:** median floor {fl:.4f} vs 0.25 x median base defect {0.25 * de:.4f} at the contact window "
          f"({len(ok0)} scenes with an approach): **{'PASS' if H0 else 'FAIL'}**",
          f"- **H1 placement (capture, base):** contact_spread - uniform {f3(d_u)}; contact_spread - spread_event {f3(d_s)}: "
          f"**{'PASS' if H1 else 'FAIL'}** (gt_super, secondary: vs uniform {f3(d_u_sup)})",
          f"- **H2 certificate:** FAR(contact_spread) {far_c:.3f} (<= 0.15) over {int(bad.sum())} bad scenes; "
          f"FAR(uniform) - FAR(contact_spread) {far_u - far_c:+.3f} [{np.quantile(dfar, .025) if len(dfar) else np.nan:+.3f}, "
          f"{np.quantile(dfar, .975) if len(dfar) else np.nan:+.3f}]: **{'PASS' if H2 else 'FAIL'}**; controls falsely rejected: {ctrl_fr:.3f}",
          f"- **H3a event-gated V2 vs cert_ref:** risk {r_ev[0]:.3f} vs {r_cr[0]:.3f} (limit +0.02); saving {f3(sv)} s: "
          f"**{'PASS' if H3a else 'FAIL'}**",
          f"- **H3b timing:** chain - event {f3(t_ch)}; shuffled - event {f3(t_sh)}: **{'PASS' if H3b else 'FAIL'}**", "",
          "| pipeline (contact scenes) | mean error | risk | cost / always-reference |", "|---|---|---|---|"]
    for lab, e_, c_ in (("generate only (two objects)", E, np.full(len(E), cg)), ("certify or simulate", e_cr, c_cr),
                        ("event-gated V2", e_ev, c_ev), ("chain-gated V2 (4k timing)", e_ch, c_ch),
                        ("event-gated V2, shuffled timing", e_sh, c_ev)):
        md.append(f"| {lab} | {boot(cap10(e_))[0]:.4f} | {risk(e_)[0]:.3f} | {boot(c_)[0] / c_ref:.2f} |")
    md += ["", f"Mean capture (base): " + ", ".join(f"{p} {np.mean([r['cap'][p] for r in con]):.3f}" for p in ("contact_spread", "uniform", "spread_event")),
           f"Mean error: base {np.mean([r['E'] for r in con]):.4f}, gt_super {np.mean([r['E_sup'] for r in con]):.4f} (the ignored interaction alone).",
           "", "## Restart curve (exploratory): mean error of V2 by restart frame relative to the generated approach k_g", "",
           "| k - k_g | n | mean error |", "|---|---|---|"]
    curve = {}
    for r in con:
        if r["k_g"] is None:
            continue
        for k, (e_, _c) in r["rk"].items():
            curve.setdefault(k - r["k_g"], []).append(min(e_, 10.0))
    for off in sorted(curve):
        if -10 <= off <= 6:
            md.append(f"| {off:+d} | {len(curve[off])} | {np.mean(curve[off]):.4f} |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"H0": bool(H0), "H1": bool(H1), "H2": bool(H2), "H3a": bool(H3a), "H3b": bool(H3b),
               "n_contact": len(con), "n_control": len(ctrl), "excluded": reasons,
               "h1": [d_u, d_s], "h2": [far_c, far_u], "h3": {"risk_ev": r_ev, "risk_cr": r_cr, "saving": sv, "chain": t_ch, "shuffled": t_sh}},
              run.path("summary.json"))
    print(f"[p4l_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
