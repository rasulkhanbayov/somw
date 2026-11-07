"""Phase 4i verdict: full-state suffix restarts (V1 stable F-lift, V2 coarse shadow) vs
certify-or-simulate. CPU only.

Pre-registered Oct 13, 2026 (PLAN.md "Phase 4i"). Same 300 queries, chain, tau, comparator and
costs as Phase 4h; only the 4h 'suffix' queries change. Per variant, PASS if
  (1) the cost saving cert - variant has a paired-bootstrap lower bound > 0 at alpha = 0.025
      (Bonferroni over the two variants), and
  (2) risk P(err > eps) of the variant <= risk of cert + 0.02.
Blow-ups (shadow or suffix) fall back to a full reference rollout: error 0, cost = spent + c_ref.
A crashed unit (CUDA context lost) is charged the full planned shadow and suffix plus c_ref.

Output: results/p4i_summary/SUMMARY.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bootstrap_ci, load_config, read_jsonl, results_dir, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p2c_summarize import query_inputs  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4d_pipeline import gen_costs  # noqa: E402

SUBSTEPS_PER_FRAME = 417


def err(e):
    """Mean error over frames; non-finite trajectories score +inf (project rule)."""
    m = float(np.mean(e)) if e is not None else np.inf
    return m if np.isfinite(m) else np.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p4i_summary"), cfg, synthetic=False)
    gcost, c_ref = gen_costs()
    full = float(cfg["full_substeps"])
    share = lambda subs: c_ref * subs / full                           # noqa: E731
    cg, tau, eps, spec = gcost["base"]["best"], float(cfg["tau"]), cfg["eps"], cfg["cert_spec"]

    h4 = [r for f in sorted(results_dir(cfg["h4_run"]).glob("shard_*/suffix.jsonl")) for r in read_jsonl(f)
          if not r.get("crashed")]
    v = {r["qid"]: r for f in sorted(results_dir(cfg.get("run_name", "p4i_state")).glob("shard_*/state.jsonl"))
         for r in read_jsonl(f)}
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == spec["fidelity"] and int(r["m"]) == spec["m"]}
    # V2 shadow: A100 seconds per coarse_cfl substep, measured per query in Phase 4d
    d4 = [r for f in sorted(results_dir(cfg["shadow_cost_run"]).glob("shard_*/fullsim.jsonl")) for r in read_jsonl(f)
          if r.get("stable") and r.get("rollout_s") and r.get("substeps")]
    rate_q = {r["qid"]: r["rollout_s"] / r["substeps"] for r in d4}
    by_split = defaultdict(list)
    for r in d4:
        by_split[r["split"]].append(r["rollout_s"] / r["substeps"])
    rate_split = {k: float(np.median(x)) for k, x in by_split.items()}
    oi = ood_index_from(["p2_build_ood"])
    root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)

    names = ("cert", "h4", "v1", "v2")
    C = {n: [] for n in names}
    E = {n: [] for n in names}
    meta = defaultdict(list)
    n_fb = {"v1": 0, "v2": 0}
    n_crash = 0
    for r in h4:
        sr = scan[r["qid"]]
        q = query_inputs(r["qid"], r["split"], root, oi)
        samples = np.load(groot / "base" / r["split"] / f"{r['qid']}.npz")["samples"]
        st = starts_for(spec["placement"], samples, q, spec["m"])
        win = {s: (np.inf if (b or c or raw is None) else raw) for s, raw, b, c in zip(sr["starts"], sr["raw"], sr["blown"], sr["crashed"])}
        secs = dict(zip(sr["starts"], sr["seconds"]))
        rej = max(win[s] for s in st) >= tau
        C["cert"].append(cg + sum(secs[s] or 0.0 for s in st) + (c_ref if rej else 0.0))
        E["cert"].append(0.0 if rej else err(r["gen_err_t"]))
        base = cg + r["probe_seconds"]
        meta["split"].append(r["split"]); meta["mode"].append(r["mode"]); meta["j"].append(r["j"])
        if r["mode"] != "suffix":
            extra = 0.0 if r["mode"] == "accept" else c_ref
            for n in ("h4", "v1", "v2"):
                C[n].append(base + extra); E[n].append(err(r["err_t"]))
            continue
        C["h4"].append(base + share(r["suffix_substeps"])); E["h4"].append(err(r["err_t"]))
        s = 2 * r["k_j"] + 1
        planned_suffix = (47 - s) * SUBSTEPS_PER_FRAME
        rate = rate_q.get(r["qid"], rate_split.get(r["split"]))
        x = v.get(r["qid"])
        if x is None:
            raise RuntimeError(f"no 4i row for suffix query {r['qid']}; the job is incomplete")
        if x.get("crashed"):
            n_crash += 1
            C["v1"].append(base + share(planned_suffix) + c_ref); E["v1"].append(0.0)
            # planned shadow: coarse substeps for s frames, scaled from the 4d full rollout of 48 frames
            sh = next((d["substeps"] for d in d4 if d["qid"] == r["qid"]), None)
            sh_cost = rate * (sh * s / 48.0) if sh else 0.0
            C["v2"].append(base + sh_cost + share(planned_suffix) + c_ref); E["v2"].append(0.0)
            continue
        if x["v1_blown"]:
            n_fb["v1"] += 1
            C["v1"].append(base + share(x["v1_substeps"]) + c_ref); E["v1"].append(0.0)
        else:
            C["v1"].append(base + share(x["v1_substeps"])); E["v1"].append(err(x["v1_err_t"]))
        sh_cost = rate * x["v2_shadow_substeps"]
        if x["v2_shadow_blown"] or x["v2_blown"]:
            n_fb["v2"] += 1
            C["v2"].append(base + sh_cost + share(x["v2_substeps"]) + c_ref); E["v2"].append(0.0)
        else:
            C["v2"].append(base + sh_cost + share(x["v2_substeps"])); E["v2"].append(err(x["v2_err_t"]))
        meta["qid_suffix"].append(r["qid"])

    C = {n: np.array(a, float) for n, a in C.items()}
    E = {n: np.array(a, float) for n, a in E.items()}
    split, mode, J = np.array(meta["split"]), np.array(meta["mode"]), np.array(meta["j"], dtype=object)
    risk = {n: float(np.mean(E[n] > eps)) for n in names}
    alpha = float(cfg["alpha"])
    md = ["# Phase 4i: full-state suffix restarts (pre-registered Oct 13, 2026)\n",
          f"{len(h4)} base queries (Phase 2c); {int((mode == 'suffix').sum())} suffix queries change. tau = {tau}. "
          f"Costs in A100 s/query: reference {c_ref:.3f}, generation {cg:.3f}; suffix = c_ref x substeps / {int(full)}; "
          f"V2 shadow = measured 4d coarse_cfl A100 s/substep. Bonferroni alpha = {alpha}.\n",
          f"Blow-up fallbacks: V1 {n_fb['v1']}, V2 {n_fb['v2']}; crashed units: {n_crash}.\n"]
    verdict = {}
    for n, label in (("v1", "V1 stable F-lift"), ("v2", "V2 coarse shadow")):
        save = bootstrap_ci(lambda a, b: float(np.mean(a - b)), [C["cert"], C[n]], n_boot=cfg["n_boot"], alpha=alpha)
        p1, p2 = save[1] > 0, risk[n] <= risk["cert"] + 0.02
        verdict[n] = p1 and p2
        md += [f"## {label}: **{'PASS' if verdict[n] else 'FAIL'}**",
               f"- (1) cost saving cert - {n}: {save[0]:+.3f} [{save[1]:+.3f}, {save[2]:+.3f}] s/query "
               f"({100 * (1 - alpha):.1f}% CI): **{'PASS' if p1 else 'FAIL'}**",
               f"- (2) risk {risk[n]:.3f} vs cert {risk['cert']:.3f} (limit +0.02): **{'PASS' if p2 else 'FAIL'}**", ""]
    md += ["| pipeline | mean error | risk | cost (s/query) | cost / always-reference |", "|---|---|---|---|---|"]
    for n, label in (("cert", "certify-or-simulate"), ("h4", "4h suffix (F = I, generated positions)"),
                     ("v1", "**V1 stable F-lift**"), ("v2", "**V2 coarse shadow**")):
        md.append(f"| {label} | {E[n].mean():.4f} | {risk[n]:.3f} | {C[n].mean():.3f} | {C[n].mean() / c_ref:.2f} |")
    md.append(f"| always reference | 0 | 0 | {c_ref:.3f} | 1.00 |")

    k = mode == "suffix"
    md += ["", f"## Secondary (on the {int(k.sum())} suffix queries)", "",
           "| comparison | mean error difference (variant - 4h) | 95% CI |", "|---|---|---|"]
    for n in ("v1", "v2"):
        d = bootstrap_ci(lambda a, b: float(np.mean(a - b)), [E[n][k], E["h4"][k]], n_boot=cfg["n_boot"])
        md.append(f"| {n} vs 4h | {d[0]:+.4f} | [{d[1]:+.4f}, {d[2]:+.4f}] |")
    rows4 = [v[q] for q in meta["qid_suffix"] if not v[q].get("crashed")]
    h4s = {r["qid"]: r for r in h4}
    fl1 = np.mean([err(x["v1_floor_err_t"]) for x in rows4])
    fl4 = np.mean([err(h4s[x["qid"]]["floor_err_t"]) for x in rows4])
    md += ["", f"- Suffix floor (ground-truth positions): V1 stable F {fl1:.4f} vs 4h F = I {fl4:.4f}; "
               f"V1 floor blow-ups {sum(x['v1_floor_blown'] for x in rows4)}.",
           f"- Position error at k_j: V2 shadow {np.mean([x['v2_shadow_err_kj'] for x in rows4]):.4f} vs generated "
           f"{np.mean([x['gen_err_kj'] for x in rows4]):.4f}.",
           f"- Share of singular values clamped by V1: mean {np.mean([x['v1_clamped_frac'] for x in rows4]):.3f} "
           f"(bounds {rows4[0]['bounds'] if rows4 else '-'}).", "",
           "| split | n suffix | 4h error | V1 error | V2 error | 4h risk | V1 risk | V2 risk |", "|---|---|---|---|---|---|---|---|"]
    for sp in sorted(set(split[k])):
        m = k & (split == sp)
        md.append(f"| {sp} | {m.sum()} | {E['h4'][m].mean():.4f} | {E['v1'][m].mean():.4f} | {E['v2'][m].mean():.4f} | "
                  f"{np.mean(E['h4'][m] > eps):.2f} | {np.mean(E['v1'][m] > eps):.2f} | {np.mean(E['v2'][m] > eps):.2f} |")
    md += ["", "| first failure j (k_j) | n | 4h error | V1 error | V2 error |", "|---|---|---|---|---|"]
    ks = [0] + list(cfg["chain_k"])
    for jj in sorted({x for x in J[k]}):
        m = k & (J == jj)
        md.append(f"| {jj} (k = {ks[jj]}) | {m.sum()} | {E['h4'][m].mean():.4f} | {E['v1'][m].mean():.4f} | {E['v2'][m].mean():.4f} |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    print(f"[p4i_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
