"""Signal selection on the Phase 2c scan (the selection set). CPU only, exploratory.

Phase 2c answered its pre-registered questions one at a time: fidelity at m = 1, window
length at 'ref', placement at 'ref'. The certificate needs ONE signal per model. This script
tabulates the candidate combinations on the 2c data so the choice can be fixed in PLAN.md
BEFORE the held-out calibration set (Phase 4a) is analysed. Nothing here is a confirmation.

Candidates: fidelity in {ref, coarse_cfl} x m in {1, 2, 4} x placement in {uniform,
event, spread_event}, alone and fused with seed spread.
  spread_event (new, defined here before any calibration result): one window at the
  generation's strongest event frame (the 'event' score) plus two windows at the 1/3 and 2/3
  points of the remaining candidates. It keeps uniform's temporal coverage, which wins on
  drag-only splits, and event's targeting, which wins on contacts.
Metrics: pooled Spearman with true error, OOD Spearman, AUROC for err_mean > 0.10, and
capture on ood_gravity_drop. Cost is simulator substeps as a fraction of a reference rollout.

Output: results/p2c_select/SELECT.md
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import auroc, load_config, read_jsonl, results_dir, spearman  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_summarize import fusion, load_scan, query_inputs  # noqa: E402
from verifier import candidate_starts, select_windows  # noqa: E402

REF_SUBSTEPS = 417 * 48


def spread_event(samples, q, m):
    """One window at the strongest generation event, two spread over the rest."""
    ev = select_windows("event", samples, q, 1, m, np.random.default_rng(0))[0][0]
    rest = [2 * k + 1 for k in candidate_starts(m) if 2 * k + 1 != ev]
    picks = [rest[int(round(f * (len(rest) - 1)))] for f in (1 / 3, 2 / 3)]
    return [ev] + picks


def contact_window(samples, q, m):
    """Phase 4l: start of the window just before the generated objects first approach (k_g - 1),
    or None (single object, or the objects never come within D_C)."""
    if q.get("object_id") is None:
        return None
    from mo_scenes import gen_contact_frame
    kg = gen_contact_frame(samples[0], q["object_id"])
    if kg is None:
        return None
    cands = candidate_starts(m)
    k = min(max(kg - 1, cands[0]), cands[-1])
    return 2 * k + 1


def starts_for(placement, samples, q, m):
    if placement == "spread_event":
        return spread_event(samples, q, m)
    if placement in ("contact", "contact_spread"):
        # Phase 4l (pre-registered Oct 15, 2026); falls back to spread_event without an approach
        c = contact_window(samples, q, m)
        if c is None:
            return spread_event(samples, q, m)
        if placement == "contact":
            return [c]
        rest = [2 * k + 1 for k in candidate_starts(m) if 2 * k + 1 != c]
        return [c] + [rest[int(round(f * (len(rest) - 1)))] for f in (1 / 3, 2 / 3)]
    rng = np.random.default_rng(zlib.crc32(q["qid"].encode()))
    return [s for s, _L, a in select_windows(placement, samples, q, 3, m, rng) if not a]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scan, split = load_scan(cfg["scan_run"], False)
    raw_rows = {(r["qid"], r["source"], r["fidelity"], int(r["m"])): r
                for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)}
    probes = {(r["qid"], r["model"]): r for f in sorted(results_dir(cfg["probe_run"]).glob("shard_*/probes.jsonl"))
              for r in read_jsonl(f)}
    ood_index = {}
    for f in sorted(results_dir("p2_build_ood").glob("shard_*/ood_manifest.json")):
        ood_index.update({e["qid"]: e["path"] for e in json.load(open(f))})
    data_root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    fids, ms, places = ["ref", "coarse_cfl"], [1, 2, 4], ["uniform", "event", "spread_event"]

    rows = []
    for qid in sorted({k[0] for k in scan}):
        sp = split[qid]
        q = query_inputs(qid, sp, data_root, ood_index)
        for model in cfg["models"]:
            pr = probes.get((qid, model))
            if pr is None:
                continue
            samples = np.load(groot / model / sp / f"{qid}.npz")["samples"]
            row = {"model": model, "split": sp, "err": pr["err_mean"], "ss": pr["seed_spread"]}
            for m in ms:
                cache = {p: starts_for(p, samples, q, m) for p in places}
                cache["exhaustive"] = [2 * k + 1 for k in candidate_starts(m)]
                for fid in fids:
                    win = scan[(qid, model, fid, m)]
                    rr = raw_rows[(qid, model, fid, m)]
                    sub = dict(zip(rr["starts"], rr["substeps"]))
                    for p, st in cache.items():
                        row[(fid, m, p)] = max(win[s] for s in st)
                        if p != "exhaustive":
                            row[("cost", fid, m, p)] = sum((sub.get(s) or 0) for s in st) / REF_SUBSTEPS
            rows.append(row)

    col = lambda rs, k: np.array([r[k] for r in rs], dtype=float)
    md = ["# Phase 2c signal selection (selection set; exploratory, confirms nothing)\n",
          "Spearman with true error (pooled / OOD), AUROC for err_mean > 0.10 (pooled), capture of the "
          "exhaustive defect on ood_gravity_drop, and median cost (fraction of a reference rollout).\n"]
    for model in cfg["models"]:
        rs = [r for r in rows if r["model"] == model]
        ro = [r for r in rs if r["split"] != "test_id"]
        gd = [r for r in rs if r["split"] == "ood_gravity_drop"]
        e, eo, y = col(rs, "err"), col(ro, "err"), (col(rs, "err") > 0.10).astype(float)
        md += [f"\n## {model}  (seed spread: rho {spearman(col(rs, 'ss'), e):.2f}, "
               f"OOD {spearman(col(ro, 'ss'), eo):.2f}, AUROC {auroc(col(rs, 'ss'), y):.3f})\n",
               "| fidelity | m | placement | rho | rho OOD | AUROC | fused rho | fused AUROC | capture (grav. drop) | cost |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for fid in fids:
            for m in ms:
                for p in places:
                    k = (fid, m, p)
                    d, do = col(rs, k), col(ro, k)
                    fu = fusion(col(rs, "ss"), d)
                    a, ex = col(gd, k), col(gd, (fid, m, "exhaustive"))
                    cap = np.where(np.isinf(ex), np.isinf(a).astype(float), a / np.where(ex > 0, ex, np.nan))
                    cost = np.median(col(rs, ("cost", fid, m, p)))
                    md.append(f"| {fid} | {m} | {p} | {spearman(d, e):.2f} | {spearman(do, eo):.2f} | "
                              f"{auroc(d, y):.3f} | {spearman(fu, e):.2f} | {auroc(fu, y):.3f} | "
                              f"{np.nanmean(cap):.2f} | {cost:.3f} |")
    out = results_dir("p2c_scan").parent / "p2c_select"
    out.mkdir(exist_ok=True)
    (out / "SELECT.md").write_text("\n".join(md) + "\n")
    print(f"[p2c_select] wrote {out / 'SELECT.md'}")


if __name__ == "__main__":
    main()
