"""Phase 15a verdict (CPU; pre-registered Oct 22, 2026, PLAN.md "Phase 15"): the kill rule for the contact-aware
two-object generator.

Error = mean point distance over the 2,048 model points (both objects) and 24 model frames vs the joint reference on
the same points (model file x[1::2][:24]); risk = P(error > 0.10). Means and risks are object-weighted by A's object.
  KILL B if the fine-tuned model's contact-scene mean error is not <= 0.7 x the composed baseline's, or its
  control-scene mean error exceeds the composed baseline's by more than 0.01.
Descriptive: base_joint (base on the joint input without fine-tuning), risk, and errors after the first approach k_c.
Output: results/p15a_summary/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, save_json, start_run  # noqa: E402
from p15a_eval import scenes  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402

EPS, RATIO, CONTROL_MARGIN = 0.10, 0.7, 0.01


def main():
    import h5py
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p15a_eval.yaml")
    ap.add_argument("--sandbox", default=None, help="code test only: generations from this folder, output there too")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.sandbox) if args.sandbox else Path(os.path.expandvars(cfg["out_dir"]))
    E = scenes(cfg, dry_run=bool(args.sandbox))     # the sandbox reads the dry-run build
    systems = ["composed", "base_joint", *cfg["models"]]
    rows = []
    for e in E:
        with h5py.File(e["path"], "r") as h:
            gt = np.asarray(h["x"][1:49:2][:24], np.float64)
        r = {"qid": e["qid"], "obj": e["qid"].split("__")[0], "kind": e["kind"], "k_c": e["k_c"], "err": {}, "post": {}}
        for s in systems:
            g = np.load(out / s / e["split"] / f"{e['qid']}.npz")["samples"][0].astype(np.float64)
            d = np.linalg.norm(g - gt, axis=2).mean(axis=1)
            r["err"][s] = float(d.mean())
            r["post"][s] = float(d[e["k_c"]:].mean()) if e["k_c"] is not None else None
        rows.append(r)

    def stat(kind, s, key="err"):
        sel = [r for r in rows if r["kind"] == kind and r[key][s] is not None]
        v = np.array([r[key][s] for r in sel])
        o = np.array([r["obj"] for r in sel])
        return {"mean": float(object_rate(v, o)), "risk": float(object_rate((v > EPS).astype(float), o)), "n": len(sel)}
    res = {k: {s: stat(k, s) for s in systems} for k in ("contact", "control")}
    post = {s: stat("contact", s, "post") for s in systems}
    c, ctl = res["contact"]["composed"]["mean"], res["control"]["composed"]["mean"]
    ok = {s: (res["contact"][s]["mean"] <= RATIO * c and res["control"][s]["mean"] <= ctl + CONTROL_MARGIN)
          for s in cfg["models"]}
    verdict = "CONTINUE" if any(ok.values()) else "KILL"
    md = ["# Phase 15a pilot: contact-aware two-object generator (pre-registered Oct 22, 2026; development scenes, "
          "exploratory)\n",
          f"Contact scenes {res['contact']['composed']['n']}, controls {res['control']['composed']['n']}. "
          f"Error over both objects' 2,048 points and 24 frames; risk = P(error > {EPS}); object-weighted.\n",
          "| system | contact mean error | contact risk | contact error after k_c | control mean error | control risk |",
          "|---|---|---|---|---|---|"]
    for s in systems:
        a, b = res["contact"][s], res["control"][s]
        md.append(f"| {s} | {a['mean']:.4f} | {a['risk']:.3f} | {post[s]['mean']:.4f} | {b['mean']:.4f} | {b['risk']:.3f} |")
    md += ["", f"- Kill rule: contact mean <= {RATIO} x composed {c:.4f} = {RATIO * c:.4f}; control mean <= composed "
               f"{ctl:.4f} + {CONTROL_MARGIN}. " + ", ".join(f"{s} {'passes' if v else 'fails'}" for s, v in ok.items()) + ".",
           f"- **Verdict: {verdict}**"]
    if args.sandbox:
        (Path(args.sandbox) / "SUMMARY.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
        return
    run = start_run("p15a_summary", cfg, synthetic=False)
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"verdict": verdict, "ok": ok, "res": res, "post": post, "rows": rows}, run.path("summary.json"))
    print("\n".join(md))
    run.finish("ok")


if __name__ == "__main__":
    main()
