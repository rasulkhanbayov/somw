"""Phase 4k verdict: cert-gated V2 on fresh queries. CPU only.

Pre-registered Oct 14, 2026 (PLAN.md "Phase 4k"), before any 4k data existed. Fresh queries
id_new_12..19 (100 test objects x 8 new in-distribution physics draws), fixed tau.
  cert_ref   accept if the 3 spread_event windows pass tau, else a reference rollout (Phase 4f)
  gated V2   cert_ref's decision first; a rejected query runs the chain (anchor + k in chain_k,
             windows the certificate already ran are not charged twice): anchor fails or no chain
             window fails -> reference rollout; else V2 at the first failure k_j (Phase 4i)
  secondary  ungated V2 (Phase 4j), gated V2 with shuffled restarts, coarse-only, generate-only
Tests (object-cluster bootstrap, object-weighted rates):
  K1 risk(gated) <= risk(cert_ref) + 0.02 and <= alpha + 0.01
  K2 saving cert_ref - gated: 95% lower bound > 0 at tau and at tau_hi
  K3 error(shuffled) - error(gated): 95% lower bound > 0 at tau (exact permutation expectation)
Confirmed if K1 and K2 pass; the timing claim needs K3 too.

Output: results/p4k_gated/{SUMMARY.md, summary.json}
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
from p4a_certificate import ood_index_from  # noqa: E402
from p4d_pipeline import gen_costs  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402

SUBSTEPS_PER_FRAME = 417


def err(e, blown=False):
    if blown or e is None:
        return np.inf
    m = float(np.mean(e))
    return m if np.isfinite(m) else np.inf


def obj_mean(v, O, objs):
    """Object-weighted mean: the mean over objects of each object's mean."""
    return float(np.mean([v[O == o].mean() for o in objs]))


def load(cfg):
    spec, model = cfg["defect_spec"], cfg["model"]
    chain_k = [int(k) for k in cfg["chain_k"]]
    chain_starts = [0] + [2 * k + 1 for k in chain_k]
    gcost, c_ref = gen_costs()
    share = lambda subs: c_ref * subs / float(cfg["full_substeps"])     # noqa: E731
    a, b, n_fit = fit_probe_cost(cfg["cost_fit_scan_run"], spec["fidelity"], spec["m"])
    rates = [r["rollout_s"] / r["substeps"] for f in sorted(results_dir(cfg["shadow_rate_run"]).glob("shard_*/fullsim.jsonl"))
             for r in read_jsonl(f) if r.get("fidelity") == "coarse_cfl" and r.get("stable") and r.get("substeps")
             and r.get("rollout_s") and r["split"].startswith(cfg["shadow_rate_splits_prefix"])]
    rate = float(np.median(rates))
    ood_index = ood_index_from(cfg["build_runs"])
    data_root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    errs = {r["qid"]: r for f in sorted(results_dir(cfg["probe_run"]).glob("shard_*/probes.jsonl"))
            for r in read_jsonl(f) if r["model"] == model}
    allk = {r["qid"]: r for f in sorted(results_dir(cfg.get("run_name", "p4k_allk")).glob("shard_*/allk.jsonl"))
            for r in read_jsonl(f)}
    rows, n_blown, n_crash = [], 0, 0
    for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")):
        for r in read_jsonl(f):
            if r["source"] != model or r["fidelity"] != spec["fidelity"] or int(r["m"]) != spec["m"]:
                continue
            if cfg.get("only_splits") and r["split"] not in cfg["only_splits"]:
                continue                                         # Phase 7b: test_id rows of the shared 2c scan
            pr, x = errs.get(r["qid"]), allk.get(r["qid"])
            if pr is None or x is None:
                raise RuntimeError(f"{r['qid']}: missing error or all-k row; the 4k chain is incomplete")
            q = query_inputs(r["qid"], r["split"], data_root, ood_index)
            samples = np.load(groot / model / r["split"] / f"{r['qid']}.npz")["samples"]
            st = starts_for(spec["placement"], samples, q, spec["m"])
            win = {s: (np.inf if (bl or cr or raw is None) else raw)
                   for s, raw, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
            sub = dict(zip(r["starts"], r["substeps"]))
            wc = lambda s: a + b * (sub.get(s) or 0)                    # noqa: E731
            ek, ck = [], []
            for k in chain_k:
                s = 2 * k + 1
                if x.get("crashed"):
                    ek.append(0.0); ck.append(rate * s * 52 + share((47 - s) * SUBSTEPS_PER_FRAME) + c_ref)
                    continue
                v = x["k"][str(k)]
                c = rate * v["shadow_substeps"] + share(v["substeps"])
                if v["shadow_blown"] or v["blown"]:
                    n_blown += 1
                    ek.append(0.0); ck.append(c + c_ref)
                else:
                    ek.append(err(v["err_t"])); ck.append(c)
            n_crash += bool(x.get("crashed"))
            rows.append({
                "obj": r["qid"].split("__")[0], "qid": r["qid"], "split": r["split"], "E": float(pr["err_mean"]),
                "S": max(win[s] for s in st), "CP": sum(wc(s) for s in st),
                "W": [win[s] for s in chain_starts],
                # chain window costs: full (ungated) and net of windows the certificate already ran (gated)
                "CW": [wc(s) for s in chain_starts], "CWg": [0.0 if s in st else wc(s) for s in chain_starts],
                "EK": ek, "CK": ck,
                "CE": np.inf if x.get("crashed") else err(x.get("coarse_err_t"), x.get("coarse_blown", False)),
                "CC": rate * (x.get("coarse_substeps") or 48 * 52)})
    A = {k: np.array([r[k] for r in rows], dtype=object if k in ("obj", "qid", "split") else float) for k in rows[0]}
    A["obj"] = A["obj"].astype(str)
    splits = sorted(set(A["split"]))
    meta = {"c_ref": c_ref, "cg": gcost[model]["best"], "probe_fit": [a, b, n_fit], "shadow_rate": rate,
            "n_rate_rows": len(rates), "n_blown": n_blown, "n_crash": n_crash, "n": len(rows), "splits": splits}
    return A, meta


def pipelines(A, tau, c_ref, cg):
    """Per-query (error, cost) of every pipeline at threshold tau, plus the gated suffix decision."""
    n = len(A["E"])
    i = np.arange(n)
    acc = A["S"] < tau
    out = {"gen": (A["E"], np.full(n, cg)),
           "coarse": (A["CE"], A["CC"]),
           "cert": (np.where(acc, A["E"], 0.0), cg + A["CP"] + (~acc) * c_ref)}
    fail = A["W"] >= tau
    j = np.where(fail.any(axis=1), fail.argmax(axis=1), -1)
    last = np.where(j < 0, A["W"].shape[1] - 1, j)
    probe_u = np.cumsum(A["CW"], axis=1)[i, last]
    probe_g = np.cumsum(A["CWg"], axis=1)[i, last]
    kk = np.clip(j - 1, 0, A["EK"].shape[1] - 1)
    sfx = j >= 1
    e_sfx, c_sfx = A["EK"][i, kk], A["CK"][i, kk]
    # ungated V2 (4j): accept only if every chain window passes
    out["v2"] = (np.where(j < 0, A["E"], np.where(sfx, e_sfx, 0.0)),
                 cg + probe_u + np.where(j < 0, 0.0, np.where(sfx, c_sfx, c_ref)))
    # gated V2: cert decides acceptance; rejected -> chain; no chain failure or anchor failure -> reference
    gs = ~acc & sfx
    out["gated"] = (np.where(acc, A["E"], np.where(gs, e_sfx, 0.0)),
                    cg + A["CP"] + np.where(acc, 0.0, probe_g + np.where(gs, c_sfx, c_ref)))
    return out, gs, kk


def shuffled_error(A, gs, kk, e_gated, sel):
    """Exact expectation over permutations of the restart points among the gated suffix queries in sel."""
    s = sel & gs
    e = e_gated.copy()
    if s.sum() == 0:
        return e
    p = np.bincount(kk[s], minlength=A["EK"].shape[1]) / s.sum()
    EKf = np.where(np.isfinite(A["EK"]), A["EK"], 1e3)          # +inf would swamp the expectation
    e[s] = EKf[s] @ p
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p4k_gated"), cfg, synthetic=False)
    A, meta = load(cfg)
    c_ref, cg, eps, alpha, B = meta["c_ref"], meta["cg"], cfg["eps"], cfg["alpha"], cfg["n_boot"]
    O = A["obj"]
    objs = np.array(sorted(set(O)))
    rng0 = np.random.default_rng(0)                             # replicate-0 split of the tau calibration
    cal0 = set(rng0.permutation(objs)[: len(objs) // 2])
    test0 = np.array([o not in cal0 for o in O])

    def boot(stat_per_query, sel=None, extra=None):
        """Object-cluster bootstrap of an object-weighted mean; returns point, 2.5%, 97.5%."""
        sel = np.ones(len(O), bool) if sel is None else sel
        ob = np.array(sorted(set(O[sel])))
        idx = {o: np.nonzero(sel & (O == o))[0] for o in ob}
        point = float(np.mean([stat_per_query(idx[o]).mean() for o in ob])) if extra is None else extra(ob, idx)
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(B):
            draw = rng.choice(ob, len(ob), replace=True)
            bs.append(np.mean([stat_per_query(idx[o]).mean() for o in draw]) if extra is None else extra(draw, idx))
        return point, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    tau, tau_hi = float(cfg["tau"]), float(cfg["tau_hi"])
    res = {}
    for t in sorted(set([tau, tau_hi] + [float(x) for x in cfg["tau_secondary"]])):
        out, gs, kk = pipelines(A, t, c_ref, cg)
        e_sh = shuffled_error(A, gs, kk, out["gated"][0], np.ones(len(O), bool))
        out["shuffled"] = (e_sh, out["gated"][1])
        res[t] = {"out": out, "gs": gs, "kk": kk,
                  "rate": {k: obj_mean((v[0] > eps).astype(float), O, objs) for k, v in out.items()},
                  "err": {k: obj_mean(np.minimum(v[0], 10.0), O, objs) for k, v in out.items()},
                  "cost": {k: obj_mean(v[1], O, objs) for k, v in out.items()},
                  "frac_rej": obj_mean((A["S"] >= t).astype(float), O, objs),
                  "frac_gsfx": obj_mean(gs.astype(float), O, objs)}
    r0 = res[tau]
    k1 = r0["rate"]["gated"] <= r0["rate"]["cert"] + 0.02 and r0["rate"]["gated"] <= alpha + 0.01
    save = {t: boot(lambda ix, t=t: res[t]["out"]["cert"][1][ix] - res[t]["out"]["gated"][1][ix]) for t in (tau, tau_hi)}
    k2 = all(save[t][1] > 0 for t in (tau, tau_hi))

    def k3_stat(draw, idx):
        ix = np.concatenate([idx[o] for o in draw])
        sel = np.zeros(len(O), bool); sel[np.unique(ix)] = True
        # the permutation distribution comes from the resampled set (duplicates counted)
        gs, kk, eg = r0["gs"], r0["kk"], r0["out"]["gated"][0]
        s = gs[ix]
        if s.sum() == 0:
            return 0.0
        p = np.bincount(kk[ix][s], minlength=A["EK"].shape[1]) / s.sum()
        EKf = np.where(np.isfinite(A["EK"]), A["EK"], 1e3)
        esh = np.where(gs, EKf @ p, eg)
        d = np.minimum(esh, 10.0) - np.minimum(eg, 10.0)
        return float(np.mean([d[idx[o]].mean() for o in draw]))
    d3 = boot(None, extra=k3_stat)
    k3 = d3[1] > 0

    f = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"     # noqa: E731
    md = ["# Phase 4k: cert-gated V2 on fresh queries (pre-registered Oct 14, 2026)\n",
          f"Fresh queries: {meta['n']} ({len(objs)} objects; splits {', '.join(meta['splits'])}). Fixed tau = {tau}; tau_hi = {tau_hi}. "
          f"Costs in A100 s/query: reference {c_ref:.3f}, generation {cg:.3f}; probe fit {meta['probe_fit'][0]:.3f} s/window + "
          f"{meta['probe_fit'][1] * 1e3:.4f} ms/substep; shadow/coarse {meta['shadow_rate'] * 1e3:.4f} ms/substep "
          f"(pooled median of {meta['n_rate_rows']} A100 calibration rollouts). Object-cluster bootstrap, {B} resamples.\n",
          f"Blown restarts (fallback to reference): {meta['n_blown']}; crashed queries: {meta['n_crash']}.\n",
          f"- **K1 risk** at tau: gated {r0['rate']['gated']:.3f} vs cert_ref {r0['rate']['cert']:.3f} (limit +0.02, and <= "
          f"{alpha + 0.01:.2f}): **{'PASS' if k1 else 'FAIL'}**",
          f"- **K2 compute** saving cert_ref - gated (s/query): tau {f(save[tau])}; tau_hi {f(save[tau_hi])}: "
          f"**{'PASS' if k2 else 'FAIL'}**",
          f"- **K3 timing** error(shuffled) - error(gated) at tau: {f(d3)}: **{'PASS' if k3 else 'FAIL'}**",
          f"- **Confirmation: {'CONFIRMED' if (k1 and k2) else 'NOT CONFIRMED'}**; timing claim: "
          f"**{'supported' if (k1 and k2 and k3) else 'not supported'}**", ""]
    names = (("gen", "generate only"), ("coarse", "coarse_cfl only"), ("cert", "certify-or-simulate"),
             ("v2", "V2 ungated (4j)"), ("gated", "**gated V2**"), ("shuffled", "gated V2, shuffled restarts"))
    for t in sorted(res):
        r = res[t]
        md += [f"## tau = {t}{' (primary)' if t == tau else ' (K2)' if t == tau_hi else ''}: rejected {r['frac_rej']:.3f}, "
               f"gated suffix {r['frac_gsfx']:.3f}", "",
               "| pipeline | mean error | risk | cost (s/query) | cost / always-reference |", "|---|---|---|---|---|"]
        md += [f"| {lab} | {r['err'][k]:.4f} | {r['rate'][k]:.3f} | {r['cost'][k]:.3f} | {r['cost'][k] / c_ref:.2f} |"
               for k, lab in names] + [""]
    ob0 = np.array(sorted(set(O[test0])))
    md += [f"## Secondary: the {len(ob0)} replicate-0 test-half objects (unseen geometry for the tau calibration), tau = {tau}", "",
           "| pipeline | mean error | risk | cost (s/query) |", "|---|---|---|---|"]
    for k, lab in names:
        e, c = r0["out"][k]
        md.append(f"| {lab} | {obj_mean(np.minimum(e[test0], 10.0), O[test0], ob0):.4f} | "
                  f"{obj_mean((e[test0] > eps).astype(float), O[test0], ob0):.3f} | {obj_mean(c[test0], O[test0], ob0):.3f} |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"K1": bool(k1), "K2": bool(k2), "K3": bool(k3), "save": {str(t): v for t, v in save.items()}, "k3": d3,
               "by_tau": {str(t): {"rate": r["rate"], "err": r["err"], "cost": r["cost"]} for t, r in res.items()}, **meta},
              run.path("summary.json"))
    print(f"[p4k_gated] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
