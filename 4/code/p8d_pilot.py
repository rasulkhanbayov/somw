"""Pilot 8d analysis: certified M1 (CM1) on development data (EXPLORATORY). CPU, after GPU job 80.

Defined Oct 18, 2026 in PLAN.md "Pilot 8d" before any 8d code ran.
Score = max window defect over k in {3, 11, 19} (+inf if a window or M1 blew up). Novelty threshold tau_M1 =
the ceil(0.95 (n + 1))-th smallest score over the windows-only id_new_0..3 queries. CM1: accept M1 if score < tau;
else keep M1 frames up to the first k with d_k >= tau and use the reference suffix from M1's state there (a blown
M1 or suffix falls back to a full reference rollout). Cost = (M1 + windows + suffix substeps) / reference substeps.
Promising if: pooled 2c shift risk <= 0.02 at <= 0.50x; test_id risk <= 0.01 at <= 0.45x (amended before data from 0.30x, infeasible); 4l contact risk <= 0.02
at <= 0.60x. Compared against the 8a pipelines on the same 2c queries (always-M1/M2, B2, certify-or-simulate).

Output: results/p8d_pilot/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402


def score(r, K):
    if r.get("crashed") or r.get("m1_blown"):
        return np.inf
    ds = [r["k"][str(k)]["d"] for k in K]
    return np.inf if any(d is None for d in ds) else max(ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, full, K = cfg["eps"], float(cfg["full_substeps"]), [int(k) for k in cfg["check_k"]]
    rows = [r for f in sorted(results_dir(cfg["certsim_run"]).glob("shard_*/certsim.jsonl")) for r in read_jsonl(f)]
    kinds = {e["qid"]: e.get("kind", "contact") for f in sorted(results_dir("p4l_scene_build").glob("shard_*/ood_manifest.json"))
             for e in load_json(f)}
    cal = np.array([score(r, K) for r in rows if r.get("windows_only")])
    n = len(cal)
    j = math.ceil(0.95 * (n + 1))
    tau = float(np.sort(cal)[j - 1]) if j <= n else np.inf
    dev = [r for r in rows if not r.get("windows_only") and not r.get("crashed")]
    out = []
    for r in dev:
        s = score(r, K)
        m1e = np.inf if r["m1_blown"] else float(np.mean(r["m1_err_t"]))
        win_c = sum(r["k"][str(k)]["win_substeps"] for k in K) / full
        c = r["m1_substeps"] / full + win_c
        if s < tau:
            e, acc = m1e, True
        else:
            acc = False
            kf = next((k for k in K if (r["k"][str(k)]["d"] is None or r["k"][str(k)]["d"] >= tau)), None)
            v = r["k"][str(kf)] if (kf is not None and not r["m1_blown"]) else None
            if v is None or v.get("blown") or v.get("err_t") is None:
                e, c = 0.0, c + 1.0                                          # full reference rollout
            else:
                e, c = float(np.mean(v["err_t"])), c + v["suffix_substeps"] / full
        grp = f"4l {kinds[r['qid']]}" if r["qid"] in kinds else ("test_id" if r["split"] == "test_id" else "shift")
        out.append({"qid": r["qid"], "split": r["split"], "grp": grp, "score": s, "m1_err": m1e, "acc": acc, "err": e,
                    "cost": c, "m1_cost": r["m1_substeps"] / full})
    md = ["# Pilot 8d: certified M1 (EXPLORATORY, development data; defined Oct 18, 2026)\n",
          f"Novelty threshold tau_M1 = {tau:.5g} (95th percentile rule over {n} in-distribution id_new_0..3 queries). "
          "Cost relative to one reference rollout.\n",
          "| set | n | M1 alone: risk / cost | CM1: risk / cost | CM1 accept rate | AUROC (score vs M1 error > 0.10) |",
          "|---|---|---|---|---|---|"]
    res = {}
    for g in ("test_id", "shift", "4l contact", "4l control"):
        sel = [o for o in out if o["grp"] == g]
        if not sel:
            continue
        m1e = np.array([o["m1_err"] for o in sel]); e = np.array([o["err"] for o in sel])
        sc = np.array([o["score"] for o in sel]); bad = m1e > eps
        au = float(roc_auc_score(bad, np.where(np.isfinite(sc), sc, 1e9))) if 0 < bad.sum() < len(bad) else float("nan")
        res[g] = {"n": len(sel), "m1_risk": float(np.mean(m1e > eps)), "m1_cost": float(np.mean([o["m1_cost"] for o in sel])),
                  "risk": float(np.mean(e > eps)), "cost": float(np.mean([o["cost"] for o in sel])),
                  "accept": float(np.mean([o["acc"] for o in sel])), "auroc": au}
        v = res[g]
        md.append(f"| {g} | {v['n']} | {v['m1_risk']:.3f} / {v['m1_cost']:.2f} | {v['risk']:.3f} / {v['cost']:.2f} | "
                  f"{v['accept']:.2f} | {v['auroc']:.3f} |")
    rule = {"shift": res.get("shift", {}).get("risk", 1) <= 0.02 and res.get("shift", {}).get("cost", 9) <= 0.50,
            "test_id": res.get("test_id", {}).get("risk", 1) <= 0.01 and res.get("test_id", {}).get("cost", 9) <= 0.45,   # amended before data
            "contact": res.get("4l contact", {}).get("risk", 1) <= 0.02 and res.get("4l contact", {}).get("cost", 9) <= 0.60}
    p8a = load_json(results_dir("p8a_pilot") / "summary.json")["part_b"]
    md += ["", "## Same 2c queries, 8a pipelines (risk / mean error / cost)", ""]
    for key in ("in distribution (test_id)|gated V2, M2 suffix, M1 shadow (B2)", "in distribution (test_id)|always simulate M1",
                "in distribution (test_id)|certify or simulate (reference)",
                "shift (4 OOD splits)|gated V2, M2 suffix, M1 shadow (B2)", "shift (4 OOD splits)|always simulate M1",
                "shift (4 OOD splits)|always simulate M2", "shift (4 OOD splits)|certify or simulate (reference)"):
        if key in p8a:
            r_, e_, c_ = p8a[key]
            md.append(f"- {key.replace('|', ': ')}: {r_:.3f} / {e_:.4f} / {c_:.2f}")
    md += ["", "## Pre-stated rules", "",
           f"- shift: CM1 risk <= 0.02 at <= 0.50x: **{'yes' if rule['shift'] else 'no'}**",
           f"- in distribution: CM1 risk <= 0.01 at <= 0.45x (amended before data): **{'yes' if rule['test_id'] else 'no'}**",
           f"- contact: CM1 risk <= 0.02 at <= 0.60x: **{'yes' if rule['contact'] else 'no'}**",
           f"- **Promising: {'YES' if all(rule.values()) else 'NO'}**",
           "", "Exploratory pilot on development data; the novelty threshold bounds false rejections in distribution, "
           "not false accepts. Nothing here is a confirmation."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"tau": tau, "n_cal": n, "res": res, "rules": rule, "promising": bool(all(rule.values()))}, run.path("summary.json"))
    print(f"[p8d] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
