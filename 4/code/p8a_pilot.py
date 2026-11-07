"""Pilot 8a analysis: the fidelity ladder (EXPLORATORY, development data). CPU, after GPU job 79.

Defined Oct 17, 2026 in PLAN.md "Pilots 8a and 8b" before any pilot code ran.
Part A  full rollouts from the true start at M1 (dt<=8e-4, grid 96), M2 (dt<=4e-4, grid 96), M3 (dt<=2e-4, grid 125),
        plus the existing coarse_cfl rows (p4d_fullsim), on the 2c pool and the 4l scenes: risk, error, stability, cost.
Part B  gated V2 (the 4k rule, tau = 0.02614) with the suffix at M2, from the coarse shadow (B1) or an M1 shadow (B2),
        against always-simulate at each medium setting and certify-then-simulate-at-medium.
Cost = substeps / reference substeps per rollout (Phase 1: the grid size barely changes the per-substep time);
probes, shadow and generation as in 4k. An unstable rollout counts as a failure.
Promising if (a) some medium setting has risk(>0.10) <= 0.02 in every single-object 2c split at <= 0.30x, and
(b) gated V2 with a medium suffix has risk <= that baseline + 0.02 and costs >= 20% less in distribution (test_id)
and >= 10% less on the pooled 2c shift splits; contact: a medium full rollout has risk <= 0.02 on 4l contact scenes.

Output: results/p8a_pilot/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p4k_gated import load, pipelines  # noqa: E402

FIDS = ("coarse_cfl", "M1", "M2", "M3")


def full_rows(cfg):
    rows = {}
    for run_name in (cfg["full_run"], cfg["coarse_full_run"]):
        for f in sorted(results_dir(run_name).glob("shard_*/fullsim.jsonl")):
            for r in read_jsonl(f):
                if r.get("fidelity") in FIDS:
                    rows[(r["qid"], r["fidelity"])] = r
    return rows


def err_cost(r, full):
    """Mean error (inf when unstable / crashed / missing) and cost relative to one reference rollout."""
    if r is None or r.get("crashed") or not r.get("stable") or r.get("err_t") is None:
        e = np.inf
    else:
        e = float(np.mean(r["err_t"]))
    subs = (r or {}).get("substeps") or full
    return e, subs / full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, full, tau = cfg["eps"], float(cfg["full_substeps"]), float(cfg["tau"])
    FR = full_rows(cfg)
    kinds = {e["qid"]: e.get("kind", "contact") for f in sorted(results_dir("p4l_scene_build").glob("shard_*/ood_manifest.json"))
             for e in load_json(f)}
    md = ["# Pilot 8a: the fidelity ladder (EXPLORATORY, development data; defined Oct 17, 2026)\n",
          "Cost relative to one reference rollout (dt 1e-4, grid 125). Unstable rollouts count as failures.\n",
          "## Part A: full rollouts from the true initial state", "",
          "| set | n | fidelity | risk > 0.10 | risk > 0.05 | median error | stable | cost |", "|---|---|---|---|---|---|---|---|"]
    qids = sorted({q for q, _ in FR})
    groups = {}
    for q in qids:
        sp = next((FR[(q, f)]["split"] for f in FIDS if (q, f) in FR), None)
        g = f"4l {kinds[q]}" if q in kinds else sp
        groups.setdefault(g, []).append(q)
    A_res = {}
    for g in sorted(groups):
        for fid in FIDS:
            ec = [err_cost(FR.get((q, fid)), full) for q in groups[g] if (q, fid) in FR]
            if not ec:
                continue
            e = np.array([x[0] for x in ec]); c = np.array([x[1] for x in ec])
            A_res[(g, fid)] = {"risk": float(np.mean(e > eps)), "risk05": float(np.mean(e > 0.05)), "cost": float(c.mean()),
                               "n": len(ec)}
            md.append(f"| {g} | {len(ec)} | {fid} | {np.mean(e > eps):.3f} | {np.mean(e > 0.05):.3f} | "
                      f"{np.median(e):.4f} | {np.mean(np.isfinite(e)):.2f} | {c.mean():.2f} |")
    single = [g for g in groups if not g.startswith("4l")]
    ok_a = {fid: all(A_res.get((g, fid), {}).get("risk", 1) <= 0.02 for g in single) and
            max(A_res.get((g, fid), {}).get("cost", 9) for g in single) <= 0.30 for fid in ("M1", "M2", "M3")}
    contact_ok = {fid: A_res.get(("4l contact", fid), {}).get("risk", 1) <= 0.02 for fid in ("M1", "M2", "M3")}

    # Part B: gated V2 with a medium suffix, on the 2c pool
    md += ["", "## Part B: pipelines on the 2c pool (gated V2 = the 4k rule at tau = 0.02614)", "",
           "| set | pipeline | risk > 0.10 | mean error | cost |", "|---|---|---|---|---|"]
    B = {}
    for variant in ("b1", "b2"):
        A, meta = load({**cfg, "run_name": f"p8a_restart_{variant}"})
        out, gs, kk = pipelines(A, tau, meta["c_ref"], meta["cg"])
        B[variant] = (A, meta, out)
    A, meta, _ = B["b1"]
    c_ref, cg = meta["c_ref"], meta["cg"]
    acc = A["S"] < tau
    sets = {"in distribution (test_id)": A["split"] == "test_id", "shift (4 OOD splits)": A["split"] != "test_id"}
    rows_b = {}
    for nm, sel in sets.items():
        cand = {"gated V2, reference suffix (4k)": None,
                "gated V2, M2 suffix, coarse shadow (B1)": B["b1"][2]["gated"],
                "gated V2, M2 suffix, M1 shadow (B2)": B["b2"][2]["gated"],
                "certify or simulate (reference)": B["b1"][2]["cert"]}
        for fid in ("M1", "M2", "M3"):
            ec = np.array([err_cost(FR.get((q, fid)), full) for q in A["qid"]])
            cand[f"always simulate {fid}"] = (ec[:, 0], ec[:, 1] * c_ref)
            cand[f"certify, else simulate {fid}"] = (np.where(acc, A["E"], ec[:, 0]),
                                                      B["b1"][2]["cert"][1] - (~acc) * c_ref + (~acc) * ec[:, 1] * c_ref)
        cand.pop("gated V2, reference suffix (4k)")
        for k, v in cand.items():
            e, c = v[0][sel], v[1][sel] / c_ref
            rows_b[(nm, k)] = (float(np.mean(e > eps)), float(np.mean(np.minimum(e, 10))), float(np.mean(c)))
            md.append(f"| {nm} | {k} | {np.mean(e > eps):.3f} | {np.mean(np.minimum(e, 10)):.4f} | {np.mean(c):.2f} |")
    # rule (b): best medium always-simulate that passed (a), vs the better gated variant
    verdict_b = {}
    for fid, ok in ok_a.items():
        if not ok:
            continue
        for var in ("B1", "B2"):
            key = [k for k in ("gated V2, M2 suffix, coarse shadow (B1)", "gated V2, M2 suffix, M1 shadow (B2)") if var in k][0]
            good = True
            for nm, need in (("in distribution (test_id)", 0.20), ("shift (4 OOD splits)", 0.10)):
                r_g, _, c_g = rows_b[(nm, key)]
                r_b, _, c_b = rows_b[(nm, f"always simulate {fid}")]
                good &= r_g <= r_b + 0.02 and c_g <= (1 - need) * c_b
            verdict_b[f"{var} vs always {fid}"] = bool(good)
    md += ["", "## Pre-stated rules", "",
           f"- (a) medium full rollout risk <= 0.02 in every single-object split at <= 0.30x: " +
           ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in ok_a.items()),
           f"- (b) gated V2 with a medium suffix beats always-simulate-medium (risk +0.02, cost -20% ID / -10% shift): " +
           (", ".join(f"{k} {'yes' if v else 'no'}" for k, v in verdict_b.items()) if verdict_b else "not applicable ((a) failed)"),
           f"- contact: medium full rollout risk <= 0.02 on 4l contact scenes: " +
           ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in contact_ok.items()),
           "", "Exploratory pilot on development data; nothing here is a confirmation."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"a": ok_a, "b": verdict_b, "contact": contact_ok,
               "part_a": {f"{g}|{f}": v for (g, f), v in A_res.items()},
               "part_b": {f"{a}|{b}": v for (a, b), v in rows_b.items()}}, run.path("summary.json"))
    print(f"[p8a] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
