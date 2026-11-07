"""Phase 14a diagnosis (CPU, EXPLORATORY; Oct 24, 2026, after the KILL verdict; PLAN.md "Phase 14", "14a result").

Freezes as a script the two analyses that were first run inline on Oct 24 (their numbers are in PLAN):
  wall survey    share of reference trajectories that come within 0.5 of the MPM domain boundary (p10e_wall rule) and
                 the median first such model frame: the 14a pool, its ACT arm, the 4g replay re-simulations (first
                 all 3,000; the inline run of Oct 24 used an unsorted "first 600" and reported 1.3%, a subset artefact),
                 the Phase 9 test families and the Phase 4m development families.
  motion split   on the 14a evaluation generations (configs/p14a_eval.yaml): per model, centre-of-mass travel ratio and
                 deformation ratio (generation / reference, medians), centre-of-mass error, shape error (positions
                 about the centre of mass), frame-1 error and the acceleration magnitude ratio; first 120 queries of
                 ID-dev (id_new_0/1), all of ood2_E_high and ood2_E_low.
Output: results/p14a_diagnose/{SUMMARY.md, diagnose.json}
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, load_config, save_json, start_run  # noqa: E402
from p10e_wall import wall_frame  # noqa: E402
from p14_eval_generate import entries  # noqa: E402


def manifest_paths(run, split=None):
    return [e["path"] for f in sorted(glob.glob(str(Path(RESULTS_ROOT) / run / "shard_*" / "ood_manifest.json")))
            for e in json.load(open(f)) if split is None or e["split"] == split]


def survey(paths):
    import h5py
    ks = []
    for p in paths:
        with h5py.File(p, "r") as h:
            ks.append(wall_frame(np.asarray(h["x"][1:48:2])))
    hits = [k for k in ks if k is not None]
    return {"n": len(paths), "share": len(hits) / len(paths), "median_first_frame": float(np.median(hits)) if hits else None}


def motion(cfg, splits, limit):
    import h5py
    out = Path(os.path.expandvars(cfg["out_dir"]))
    sel = [e for e in entries(cfg) if e["split"] in splits][:limit]
    res = {}
    for m in cfg["models"]:
        r = {k: [] for k in ("err", "travel", "deform", "com", "shape", "frame1", "accel")}
        for e in sel:
            with h5py.File(e["path"], "r") as h:
                gt = np.asarray(h["x"][1:48:2])
            g = np.load(out / m / e["split"] / f"{e['qid']}.npz")["samples"][0]
            r["err"].append(np.linalg.norm(g - gt, axis=2).mean())
            dg, dr = np.linalg.norm(g.mean(1)[-1] - g.mean(1)[0]), np.linalg.norm(gt.mean(1)[-1] - gt.mean(1)[0])
            r["travel"].append(dg / max(dr, 1e-6))
            cg, cr = g - g.mean(1, keepdims=True), gt - gt.mean(1, keepdims=True)
            r["deform"].append(np.linalg.norm(cg - cg[0], axis=2).mean() / max(np.linalg.norm(cr - cr[0], axis=2).mean(), 1e-6))
            r["com"].append(np.linalg.norm(g.mean(1) - gt.mean(1), axis=1).mean())
            r["shape"].append(np.linalg.norm(cg - cr, axis=2).mean())
            r["frame1"].append(np.linalg.norm(g[0] - gt[0], axis=1).mean())
            acc, acr = g[2:] - 2 * g[1:-1] + g[:-2], gt[2:] - 2 * gt[1:-1] + gt[:-2]
            r["accel"].append(np.linalg.norm(acc, axis=2).mean() / max(np.linalg.norm(acr, axis=2).mean(), 1e-9))
        res[m] = {"n": len(sel), "err": float(np.mean(r["err"])), "travel": float(np.median(r["travel"])),
                  "deform": float(np.median(r["deform"])), "com": float(np.mean(r["com"])), "shape": float(np.mean(r["shape"])),
                  "frame1": float(np.mean(r["frame1"])), "accel": float(np.median(r["accel"]))}
    return res


def main():
    cfg = load_config("configs/p14a_eval.yaml")
    pool = json.load(open(Path(RESULTS_ROOT) / "p14a_acquire" / "pool.json"))
    act = sorted(set(json.load(open(Path(RESULTS_ROOT) / "p14a_acquire" / "arm_ACT.json"))["files"]))
    sets = {"14a pool (all)": [p["path"] for p in pool], "14a ACT arm": act,
            "replay (4g ID re-sims, all 3,000)": manifest_paths("p4g_train_build")}
    for fam in ("ood3_E_low", "ood3_E_high", "ood3_force", "ood3_drop", "id_new_22", "id_new_23"):
        sets["Phase 9 " + fam] = manifest_paths("p9_build", fam)
    for fam in ("ood2_E_low", "ood2_E_high", "ood2_force"):
        sets["dev 4m " + fam] = manifest_paths("p4m_build", fam)
    res = {"wall": {k: survey(v) for k, v in sets.items()}}
    md = ["# Phase 14a diagnosis (Oct 24, 2026; EXPLORATORY, after the KILL verdict)\n", "## Wall survey (reference within 0.5 of the domain boundary)", "",
          "| set | n | reach a wall | median first model frame |", "|---|---|---|---|"]
    for k, v in res["wall"].items():
        md.append(f"| {k} | {v['n']} | {100 * v['share']:.1f}% | {v['median_first_frame'] if v['median_first_frame'] is not None else '—'} |")
    for name, splits, limit in (("ID-dev (first 120)", ("id_new_0", "id_new_1"), 120), ("ood2_E_high", ("ood2_E_high",), None),
                                ("ood2_E_low", ("ood2_E_low",), None)):
        r = motion(cfg, splits, limit)
        res[name] = r
        md += ["", f"## Motion split: {name}", "", "| model | error | travel gen/ref | deformation gen/ref | centre-of-mass error | "
               "shape error | frame-1 error | acceleration gen/ref |", "|---|---|---|---|---|---|---|---|"]
        for m, v in r.items():
            md.append(f"| {m} | {v['err']:.4f} | {v['travel']:.2f} | {v['deform']:.2f} | {v['com']:.4f} | {v['shape']:.4f} | "
                      f"{v['frame1']:.4f} | {v['accel']:.2f} |")
    run = start_run("p14a_diagnose", cfg, synthetic=False)
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json(res, run.path("diagnose.json"))
    print("\n".join(md))
    run.finish("ok")


if __name__ == "__main__":
    main()
