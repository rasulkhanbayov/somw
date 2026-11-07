"""Phase 2c verdicts from the exhaustive window scan (results/p2c_scan). CPU only.

Pre-registered on Oct 11, 2026, before any scan result existed. Each question maps to one
paper contribution. All windows that blew up score raw = +inf (the Phase 2 rule). "defect"
means the max raw defect over the chosen windows; "uniform-3" is Phase 2-v2's placement.

A. Simulator fidelity (contribution 3: the probe can run at the cheap matched-accuracy
   setting, dt 8e-4 / grid 64, about 1/8 of the reference cost). Using m = 1, uniform-3:
     PASS-A if at 'coarse': lift floor / generated defect (in-distribution) <= 0.25 for both
     models, base pooled Spearman with true error >= 0.5, and that Spearman is no more than
     0.05 below the same signal at 'ref'.
A'. (Added Oct 11, 2026, after the fixed coarse setting violated the CFL condition on the
   stiff ood_E_high split and crashed; pre-registered before any 'coarse_cfl' result.)
   The same criteria as A for fidelity 'coarse_cfl': grid 64 and, per query, substep
   min(8e-4, 0.5 x dx / wave speed). A itself is unchanged and still reported.
B. Window length (contribution 1 for the large model, whose errors drift smoothly): at
   'ref', uniform-3, primary m = 2 (m = 1 and 4 secondary). The fusion signal is the mean of
   the ranks of seed spread and window defect within the pool (no weights, no fitting).
     PASS-B if, for large, fusion beats seed spread in pooled Spearman and on OOD by paired
     bootstrap (95% lower bound > 0).
C. Placement (contribution 2: windows belong where errors happen). Split ood_gravity_drop,
   the one split with contact events; 'ref', m = 1, k = 3 windows.
   capture = strategy defect / exhaustive defect (max over all windows).
     PASS-C if event_union's mean capture exceeds uniform's by paired bootstrap (95% lower
     bound > 0) for each model separately. Drag-only splits are reported, not judged:
     their error is global (results/diag_p2), so no placement can matter there.
D. Certificate (contribution 1: one threshold that works across conditions). Pooled over
   all splits; 'ref', m = 1, uniform-3. AUROC for detecting err_mean > eps, eps in
   {0.05, 0.10, 0.20}; primary eps = 0.10.
     PASS-D if AUROC(defect) - AUROC(seed spread) has 95% lower bound > 0 for base.
     Large and fusion are reported.

Output: results/p2c_summary/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (auroc, bootstrap_ci, load_config, paired_bootstrap_diff, rankdata,  # noqa: E402
                    read_jsonl, results_dir, save_json, spearman, start_run)
from p2b_generate import gen_dir  # noqa: E402
from verifier import candidate_starts, select_windows  # noqa: E402

INF = float("inf")
STRATS = ["uniform", "random", "event", "event_union", "disagree", "oracle_v1", "exhaustive"]


GATED = defaultdict(int)


def load_scan(run: str, allow_syn: bool) -> dict:
    """{(qid, source, fidelity, m): {start: raw (+inf if blown/crashed)}} plus split per qid."""
    scan, split = {}, {}
    for f in sorted(results_dir(run, allow_syn).glob("shard_*/scan.jsonl")):
        for r in read_jsonl(f):
            d = {}
            for s, raw, b, c in zip(r["starts"], r["raw"], r["blown"], r["crashed"]):
                d[int(s)] = INF if (b or c or raw is None) else float(raw)
            scan[(r["qid"], r["source"], r["fidelity"], int(r["m"]))] = d
            split[r["qid"]] = r["split"]
            if r.get("cfl_gated"):
                GATED[(r["fidelity"], r["split"])] += len(r["starts"])
    return scan, split


def query_inputs(qid: str, sp: str, data_root: Path, ood_index: dict) -> dict:
    path = data_root / "testset" / f"{qid}.h5" if sp == "test_id" else Path(ood_index[qid])
    with h5py.File(path, "r") as h:
        x = np.asarray(h["x"], dtype=np.float64)
        return {"qid": qid, "x0": x[0], "gt": x[1::2][:24],
                "floor_height": float(np.asarray(h["floor_height"])),
                "gravity": int(np.asarray(h["gravity"])) if "gravity" in h else 0}


def strategy_starts(strat: str, samples, q: dict, m: int, seed: int) -> list[int]:
    if strat == "exhaustive":
        return [2 * k + 1 for k in candidate_starts(m)]
    if strat == "oracle_v1":
        w = select_windows("oracle", samples, q, 3, m, None, gt=q["gt"])
    else:
        rng = np.random.default_rng(seed + zlib.crc32(q["qid"].encode()))
        w = select_windows(strat, samples, q, 3, m, rng)
    return [s for s, _L, a in w if not a]


def fusion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (rankdata(a) + rankdata(b)) / 2.0


def rho_ci(sig, err, n_boot):
    return bootstrap_ci(spearman, [np.asarray(sig, float), np.asarray(err, float)], n_boot=n_boot)


def fmt_ci(t):
    return "n/a" if not np.isfinite(t[0]) else f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.allow_synthetic)
    syn = args.allow_synthetic
    run = start_run(cfg.get("run_name", "p2c_summary"), cfg, synthetic=syn)
    nb = cfg["n_boot"]
    scan, split = load_scan(cfg["scan_run"], syn)
    probes = {(r["qid"], r["model"]): r
              for f in sorted(results_dir(cfg["probe_run"], syn).glob("shard_*/probes.jsonl"))
              for r in read_jsonl(f)}
    data_root = Path(cfg["paths"]["data_root"])
    ood_index = {}
    if not syn:
        import json
        for f in sorted(results_dir("p2_build_ood").glob("shard_*/ood_manifest.json")):
            ood_index.update({e["qid"]: e["path"] for e in json.load(open(f))})
    groot = gen_dir(cfg, syn)
    models = cfg["models"]
    qids = sorted({k[0] for k in scan})

    # Per (qid, model): strategy defects at every (fidelity, m) available.
    rows = []
    for qid in qids:
        sp = split[qid]
        q = query_inputs(qid, sp, data_root, ood_index) if not syn else None
        for model in models:
            pr = probes.get((qid, model))
            if pr is None:
                continue
            samples = np.load(groot / model / sp / f"{qid}.npz")["samples"]
            if syn:
                q = {"qid": qid, "x0": samples[0][0], "gt": samples[0], "floor_height": -1e9, "gravity": 0}
            row = {"qid": qid, "split": sp, "model": model, "err": pr["err_mean"],
                   "seed_spread": pr["seed_spread"]}
            for fid in cfg["fidelities"]:
                for m in cfg["ms"]:
                    win = scan.get((qid, model, fid, m))
                    floor = scan.get((qid, "gt", fid, m))
                    if win is None:
                        continue
                    for st in STRATS:
                        starts = strategy_starts(st, samples, q, m, cfg["seed"])
                        row[f"{fid}|{m}|{st}"] = max(win[s] for s in starts)
                    if floor is not None:
                        uni = strategy_starts("uniform", samples, q, m, cfg["seed"])
                        row[f"{fid}|{m}|floor"] = max(floor[s] for s in uni)
                    vals = [win[2 * k + 1] for k in candidate_starts(m)]
                    fin = [v for v in vals if np.isfinite(v)]
                    row[f"{fid}|{m}|loc"] = (max(fin) / sum(fin)) if len(fin) == len(vals) and sum(fin) > 0 else np.nan
            rows.append(row)

    md, out = [], {}
    sel = lambda model, sp=None: [r for r in rows if r["model"] == model and (sp is None or (r["split"] == sp))]
    ood = lambda model: [r for r in rows if r["model"] == model and r["split"] != "test_id"]
    col = lambda rs, k: np.array([r.get(k, np.nan) for r in rs], dtype=float)
    med = lambda x: float(np.median(x[~np.isnan(x)])) if np.any(~np.isnan(x)) else np.nan

    # ---- A. fidelity ----------------------------------------------------------------------
    md.append("## A. Simulator fidelity (m = 1, uniform-3)\n")
    md.append("| fidelity | model | floor / defect (ID) | rho all [95% CI] | rho OOD |")
    md.append("|---|---|---|---|---|")
    A = {}
    for fid in cfg["fidelities"]:
        for model in models:
            k = f"{fid}|1|uniform"
            rid = sel(model, "test_id")
            ratio = med(col(rid, f"{fid}|1|floor")) / med(col(rid, k))
            rall = rho_ci(col(sel(model), k), col(sel(model), "err"), nb)
            rood = spearman(col(ood(model), k), col(ood(model), "err"))
            A[(fid, model)] = (ratio, rall[0])
            md.append(f"| {fid} | {model} | {ratio:.3f} | {fmt_ci(rall)} | {rood:+.2f} |")
    def pass_fid(f):
        return bool(all(A.get((f, m), (np.inf,))[0] <= 0.25 for m in models)
                    and A.get((f, "base"), (0, -1))[1] >= 0.5
                    and A.get((f, "base"), (0, -1))[1] >= A.get(("ref", "base"), (0, 9))[1] - 0.05)
    passA = pass_fid("coarse")
    passA2 = pass_fid("coarse_cfl") if "coarse_cfl" in cfg["fidelities"] else None
    out["A"], out["A_prime"] = passA, passA2
    if GATED:
        md.append("\nWindows skipped by the CFL gate and scored as blown (+inf): "
                  + ", ".join(f"{f}/{s} {n}" for (f, s), n in sorted(GATED.items())))

    # ---- B. window length + fusion ---------------------------------------------------------
    md.append("\n## B. Window length and fusion with seed spread ('ref', uniform-3)\n")
    md.append("| model | m | rho defect | rho fusion | rho seed spread | fusion - seed spread, OOD [95% CI] |")
    md.append("|---|---|---|---|---|---|")
    B = {}
    for model in models:
        rs, ro = sel(model), ood(model)
        for m in cfg["ms"]:
            k = f"ref|{m}|uniform"
            if np.all(np.isnan(col(rs, k))):
                continue
            fu, fuo = fusion(col(rs, "seed_spread"), col(rs, k)), fusion(col(ro, "seed_spread"), col(ro, k))
            ss, sso = col(rs, "seed_spread"), col(ro, "seed_spread")
            e, eo = col(rs, "err"), col(ro, "err")
            adv = paired_bootstrap_diff(spearman, [fuo, eo], [sso, eo], n_boot=nb)
            B[(model, m)] = (spearman(fu, e), spearman(ss, e), adv)
            md.append(f"| {model} | {m} | {spearman(col(rs, k), e):.2f} | {spearman(fu, e):.2f} | "
                      f"{spearman(ss, e):.2f} | {fmt_ci(adv)} |")
    b = B.get(("large", cfg["primary_m"]))
    passB = bool(b and b[0] > b[1] and b[2][1] > 0)
    out["B"] = passB

    # ---- C. placement ----------------------------------------------------------------------
    md.append("\n## C. Placement: capture of the exhaustive defect ('ref', m = 1, k = 3)\n")
    md.append("| model | split group | " + " | ".join(STRATS[:-1]) + " | event_union - uniform [95% CI] |")
    md.append("|---|---|" + "---|" * (len(STRATS) - 1) + "---|")

    def capture(rs, st):
        a, ex = col(rs, f"ref|1|{st}"), col(rs, "ref|1|exhaustive")
        c = np.where(np.isinf(ex), np.isinf(a).astype(float), a / np.where(ex > 0, ex, np.nan))
        return c

    C = {}
    for model in models:
        for gname, rs in (("ood_gravity_drop", sel(model, "ood_gravity_drop")),
                          ("drag-only", [r for r in sel(model) if r["split"] != "ood_gravity_drop"])):
            if not rs:
                continue
            caps = {st: capture(rs, st) for st in STRATS[:-1]}
            d = paired_bootstrap_diff(lambda x: float(np.nanmean(x)), [caps["event_union"]], [caps["uniform"]],
                                      n_boot=nb)
            C[(model, gname)] = d
            md.append(f"| {model} | {gname} | " + " | ".join(f"{np.nanmean(caps[s]):.2f}" for s in STRATS[:-1])
                      + f" | {fmt_ci(d)} |")
    passC = bool(all(C.get((m, "ood_gravity_drop"), (0, -1, 0))[1] > 0 for m in models))
    out["C"] = passC
    md.append("\nDefect localisation (max window / sum over windows, 'ref', m = 1; uniform spread = 1/23 = 0.043): "
              + ", ".join(f"{m}/{s} {med(col(sel(m, s), 'ref|1|loc')):.3f}"
                          for m in models for s in sorted({r['split'] for r in rows})))

    # ---- D. certificate --------------------------------------------------------------------
    md.append("\n## D. One threshold across all splits: AUROC for err_mean > eps ('ref', m = 1, uniform-3)\n")
    md.append("| model | eps | positives | AUROC defect | AUROC seed spread | AUROC fusion | defect - seed spread [95% CI] |")
    md.append("|---|---|---|---|---|---|---|")
    D = {}
    for model in models:
        rs = sel(model)
        d, ss, e = col(rs, "ref|1|uniform"), col(rs, "seed_spread"), col(rs, "err")
        fu = fusion(ss, d)
        for eps in cfg["eps"]:
            y = (e > eps).astype(float)
            diff = paired_bootstrap_diff(auroc, [d, y], [ss, y], n_boot=nb)
            D[(model, eps)] = diff
            md.append(f"| {model} | {eps} | {int(y.sum())}/{len(y)} | {auroc(d, y):.3f} | {auroc(ss, y):.3f} | "
                      f"{auroc(fu, y):.3f} | {fmt_ci(diff)} |")
    passD = bool(D.get(("base", cfg["primary_eps"]), (0, -1, 0))[1] > 0)
    out["D"] = passD

    head = ["# Phase 2c summary (pre-registered Oct 11, 2026)\n",
            f"- A fidelity, fixed coarse dt 8e-4 (contribution 3): {'PASS' if passA else 'FAIL'}",
            f"- A' fidelity, CFL-adaptive coarse (contribution 3): "
            f"{'n/a' if passA2 is None else ('PASS' if passA2 else 'FAIL')}",
            f"- B window length + fusion, large (contribution 1): {'PASS' if passB else 'FAIL'}",
            f"- C placement on contact events (contribution 2): {'PASS' if passC else 'FAIL'}",
            f"- D single-threshold certificate, base (contribution 1): {'PASS' if passD else 'FAIL'}", ""]
    if syn:
        head.insert(0, "> SYNTHETIC INPUT. These numbers test the pipeline only and mean nothing.\n")
    run.path("SUMMARY.md").write_text("\n".join(head + md) + "\n")
    save_json(out, run.path("summary.json"))
    print(f"[p2c_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
