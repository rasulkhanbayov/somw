"""Phase 7b: PhysCtrl's own metrics on PhysCtrl's released test set (descriptive, no test). CPU.

Pre-registered Oct 17, 2026 (PLAN.md "Phase 7b"). test_id was Phase 2c development data, so nothing here
is a confirmation. Pipelines by the exact 4k rule (p4k_gated.pipelines) with tau fixed at 0.02614:
generate only (sample 0), certify-or-simulate, gated V2, always simulate. Each pipeline's output is
assembled per query: accepted -> sample 0; gated suffix -> the saved splice at the chosen chain point;
otherwise (or a blown restart) -> the reference rollout, i.e. the ground truth.
Metrics: PhysCtrl eval.py's position MSE in model coordinates ((x - 5) / 2) and its DeformLoss (computed on
GPU by p4j_allk, row key "deform"); our mean point error, risk (eps) and cost.

Phase 10b (pre-registered Oct 20, 2026): configs/p10b_testset.yaml re-runs this with the final pipeline
(cert_windows: one; M1 shadow and M2 suffix in the GPU run) -> results/p10b_testset/.

Output: results/p7b_testset/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from backends import query_from_h5  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p4k_gated import load, pipelines  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg["summary_run_name"], cfg, synthetic=False)
    eps, tau, chain_k = cfg["eps"], float(cfg["tau"]), [int(k) for k in cfg["chain_k"]]
    A, meta = load(cfg)
    c_ref, cg = meta["c_ref"], meta["cg"]
    if cfg.get("cert_windows") == "one":          # Phase 10b: the final (Phase 9) single-object certificate
        from p9_confirm import one_window
        A = one_window(cfg, A)
    out, gs, kk = pipelines(A, tau, c_ref, cg)
    acc = A["S"] < tau
    allk = {r["qid"]: r for f in sorted(results_dir(cfg["run_name"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    tdir = Path(os.path.expandvars(cfg["save_traj_dir"]))
    groot = gen_dir(cfg, False)
    root = Path(cfg["paths"]["data_root"])
    M = {k: {"mse": [], "deform": [], "err": []} for k in ("gen", "cert", "gated", "always")}
    vol = None
    if cfg.get("physctrl_volume_metrics"):        # Phase 10b addition: PhysCtrl's own vIoU / Chamfer (its Table 2)
        import importlib.util
        import torch
        spec = importlib.util.spec_from_file_location(
            "physctrl_volume_iou", Path(os.path.expandvars(cfg["physctrl_root"])) / "src" / "volume_iou.py")
        vol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vol)              # by path, unchanged PhysCtrl code
        for k in M:
            M[k]["viou"], M[k]["cd"], M[k]["cmse"] = [], [], []
    n_missing = 0
    for i, (qid, sp) in enumerate(zip(A["qid"], A["split"])):
        q = query_from_h5(root / "testset" / f"{qid}.h5", sp, qid)
        gt = q["x"][1::2][:24].astype(np.float64)
        g0 = np.load(groot / cfg["model"] / sp / f"{qid}.npz")["samples"][0].astype(np.float64)
        row = allk[qid]
        dl = row["deform"]
        saved = np.load(tdir / sp / f"{qid}.npz")
        if np.abs(saved["gen"] - g0).max() > 1e-4:
            raise RuntimeError(f"{qid}: saved sample 0 differs from the generation file")
        choice = {"gen": "gen", "cert": "gen" if acc[i] else "gt", "always": "gt"}
        if acc[i]:
            choice["gated"] = "gen"
        elif gs[i]:
            key = f"k{chain_k[kk[i]]}"
            choice["gated"] = key if key in saved.files else "gt"
            n_missing += key not in saved.files
        else:
            choice["gated"] = "gt"
        for pk, c in choice.items():
            traj = gt if c == "gt" else (g0 if c == "gen" else saved[c].astype(np.float64))
            M[pk]["mse"].append(float(np.mean(((traj - gt) / 2.0) ** 2)))          # eval.py's F.mse_loss scale
            M[pk]["deform"].append(float(dl[c]))
            M[pk]["err"].append(float(np.linalg.norm(traj - gt, axis=2).mean()))
            if vol is not None:                   # model coordinates, as PhysCtrl's volume_iou.py main
                iou, cd, cmse = vol.evaluate_4d(torch.from_numpy(((traj - 5.0) / 2.0).astype(np.float32)),
                                                torch.from_numpy(((gt - 5.0) / 2.0).astype(np.float32)))
                M[pk]["viou"].append(float(iou)); M[pk]["cd"].append(float(cd)); M[pk]["cmse"].append(float(cmse))
    for pk, o in (("gen", "gen"), ("cert", "cert"), ("gated", "gated")):        # cross-check against 4k's rule
        dev = np.abs(np.array(M[pk]["err"]) - out[o][0]).max()
        if dev > 2e-3:
            raise RuntimeError(f"{pk}: assembled error differs from the 4k pipeline by {dev:.4f}")
    cost = {"gen": out["gen"][1], "cert": out["cert"][1], "gated": out["gated"][1], "always": np.full(len(acc), c_ref)}
    labs = (("gen", "PhysCtrl (generate only)"), ("cert", "certify or simulate"), ("gated", "**ours: certify + repair**"),
            ("always", "always simulate (exact)"))
    md = ["# " + cfg.get("title", "Phase 7b: PhysCtrl's own metrics on PhysCtrl's test set (descriptive; pre-registered Oct 17, 2026)") + "\n",
          f"Queries: {len(acc)} (test_id; Phase 2c development data, so descriptive only). tau = {tau}. "
          f"Accepted {acc.mean():.2f}; gated suffix {gs.mean():.2f}; missing restarts charged as reference: {n_missing}. "
          f"DeformLoss of the ground truth itself (PhysCtrl's reference value): {np.mean(M['always']['deform']):.4g}.\n",
          "| pipeline | position MSE (PhysCtrl eval.py) | DeformLoss (PhysCtrl) | mean point error | risk (> 0.10) | cost / always-simulate |",
          "|---|---|---|---|---|---|"]
    for k, lab in labs:
        e = np.array(M[k]["err"])
        md.append(f"| {lab} | {np.mean(M[k]['mse']):.3e} | {np.mean(M[k]['deform']):.4g} | {e.mean():.4f} | "
                  f"{np.mean(e > eps):.3f} | {np.mean(cost[k]) / c_ref:.2f} |")
    if vol is not None:
        md += ["", "PhysCtrl's own Table-2 metrics (its src/volume_iou.py, per frame, averaged; PhysCtrl reports 77.59% / "
               "0.0028 / 0.0015 at 25 sampling steps):", "", "| pipeline | vIoU | Chamfer | correspondence MSE |", "|---|---|---|---|"]
        md += [f"| {lab} | {100 * np.mean(M[k]['viou']):.2f}% | {np.mean(M[k]['cd']):.4f} | {np.mean(M[k]['cmse']):.2e} |" for k, lab in labs]
    md += ["", "Means over queries (eval.py sums per batch; the per-query mean is reported). Position MSE uses PhysCtrl's "
           "normalised coordinates, as eval.py's F.mse_loss(output, points_tgt). Lower is better for every column."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({k: {m: float(np.mean(v)) for m, v in M[k].items()} for k in M}, run.path("summary.json"))
    print(f"[p7b] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
