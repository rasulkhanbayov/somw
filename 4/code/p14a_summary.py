"""Phase 14a verdict (CPU; pre-registered Oct 22, 2026, PLAN.md "Phase 14"): the pilot's kill rule and variant choice.

Errors of sample 0 (configs/p14a_eval.yaml) against the stored reference x[1::2][:24]; risk = P(error > 0.10),
object-weighted (object = qid prefix). shift-dev = ood2_E_low/E_high/force; ID-dev = id_new_0/1.
KILL A unless some ACT variant has shift-dev risk <= 0.7 x base's AND ID-dev mean error <= base's + 0.005.
Variant for 14b: among ACT variants passing the ID rule, the lower shift-dev risk (tie -> L). ACT vs RND: descriptive.
Config key `summary` (pilot 14a-v2, PLAN "Phase 14"): run_name, title, variants, tie, and truncate_at_wall: errors over
model frames before the reference first comes within 0.5 of the MPM domain wall (p10e_wall rule; queries with fewer
than 4 such frames dropped and counted). 14a itself runs with the defaults below.
Output: results/<run_name>/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, save_json, start_run  # noqa: E402
from p14_eval_generate import entries  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402
from p10e_wall import MIN_FRAMES, wall_frame  # noqa: E402

EPS, KILL_RATIO, ID_MARGIN = 0.10, 0.7, 0.005
SHIFT, ID = ("ood2_E_low", "ood2_E_high", "ood2_force"), ("id_new_0", "id_new_1")


def main():
    import h5py
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p14a_eval.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc = {"run_name": "p14a_summary", "title": "Phase 14a pilot: simulator distillation (pre-registered Oct 22, 2026; "
          "development data, exploratory)", "variants": ["F", "L"], "tie": "L", "truncate_at_wall": False,
          **cfg.get("summary", {})}
    out = Path(os.path.expandvars(cfg["out_dir"]))
    E = entries(cfg)
    gt, n_drop = {}, 0
    for e in E:
        with h5py.File(e["path"], "r") as h:
            gt[e["qid"]] = np.asarray(h["x"][1:49:2][:24], np.float64)
        if sc["truncate_at_wall"]:
            kw = wall_frame(gt[e["qid"]])
            if kw is not None and kw < MIN_FRAMES:
                del gt[e["qid"]]
                n_drop += 1
            elif kw is not None:
                gt[e["qid"]] = gt[e["qid"]][:kw]
    E = [e for e in E if e["qid"] in gt]
    res, err = {}, {}
    for name in cfg["models"]:
        err[name] = {e["qid"]: float(np.linalg.norm(np.load(out / name / e["split"] / f"{e['qid']}.npz")["samples"][0]
                                                    [:len(gt[e["qid"]])].astype(np.float64) - gt[e["qid"]], axis=2).mean())
                     for e in E}
    def stats(name, splits):
        sel = [e for e in E if e["split"] in splits]
        ev = np.array([err[name][e["qid"]] for e in sel])
        objs = np.array([e["qid"].split("__")[0] for e in sel])   # an array: object_rate compares with ==
        return {"risk": float(object_rate((ev > EPS).astype(float), objs)), "mean": float(object_rate(ev, objs)), "n": len(sel)}
    md = [f"# {sc['title']}\n"] + ([f"Errors before the first wall frame; {n_drop} queries with < {MIN_FRAMES} wall-free "
                                     "frames dropped. Query counts per set in summary.json.\n"] if sc["truncate_at_wall"] else [])
    md += [
          "| model | shift-dev risk | shift-dev mean error | ID-dev risk | ID-dev mean error | " +
          " | ".join(f"{s} risk" for s in SHIFT) + " |", "|---|---|---|---|---|" + "---|" * len(SHIFT)]
    for name in cfg["models"]:
        res[name] = {"shift": stats(name, SHIFT), "id": stats(name, ID), **{s: stats(name, (s,)) for s in SHIFT}}
        r = res[name]
        md.append(f"| {name} | {r['shift']['risk']:.3f} | {r['shift']['mean']:.4f} | {r['id']['risk']:.3f} | "
                  f"{r['id']['mean']:.4f} | " + " | ".join(f"{r[s]['risk']:.3f}" for s in SHIFT) + " |")
    b = res["base"]
    V = sc["variants"]
    ok = {v: (res[f"ACT_{v}"]["shift"]["risk"] <= KILL_RATIO * b["shift"]["risk"]
              and res[f"ACT_{v}"]["id"]["mean"] <= b["id"]["mean"] + ID_MARGIN) for v in V}
    id_ok = [v for v in V if res[f"ACT_{v}"]["id"]["mean"] <= b["id"]["mean"] + ID_MARGIN]
    chosen = min(id_ok, key=lambda v: (res[f"ACT_{v}"]["shift"]["risk"], v != sc["tie"])) if id_ok else None
    verdict = "CONTINUE" if any(ok.values()) else "KILL"
    md += ["", "- Kill rule: " + ", ".join(f"ACT_{v} {'passes' if ok[v] else 'fails'}" for v in V) + " "
               f"(shift-dev risk <= {KILL_RATIO} x base {b['shift']['risk']:.3f} = {KILL_RATIO * b['shift']['risk']:.3f}; "
               f"ID-dev mean <= base {b['id']['mean']:.4f} + {ID_MARGIN}).",
           f"- **Verdict: {verdict}**" + (f"; variant for 14b: **{chosen}**" if verdict == "CONTINUE" else ""),
           "- ACT vs RND (descriptive): shift-dev risk " + "; ".join(
               f"{v} {res[f'ACT_{v}']['shift']['risk']:.3f} vs {res[f'RND_{v}']['shift']['risk']:.3f}" for v in V) + "."]
    run = start_run(sc["run_name"], cfg, synthetic=False)
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"verdict": verdict, "variant": chosen, "res": res}, run.path("summary.json"))
    print("\n".join(md))
    run.finish("ok")


if __name__ == "__main__":
    main()
