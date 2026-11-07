"""Phase 11 verdict: video-level evaluation (pre-registered Oct 20, 2026; PLAN.md "Phase 11"). CPU.

Cases: case_v2.json under the render roots of configs/p11_judge.yaml. A case is *shifted* if its force is outside
PhysCtrl's training range (Phase 11 coefficient 0.40, or PhysCtrl's own examples at 3x their force) and *rejected* if
ours-v2 did not accept the generation (only there do ours and PhysCtrl differ).
Tests on rejected shifted cases (95% bootstrap over cases, 2,000 resamples, seed 1):
  V0 judge validity: PC(reference) - PC(PhysCtrl) > 0 (lower bound); if not, V1 is reported as uninformative
  V1: PC(ours) - PC(PhysCtrl) > 0 (lower bound)
  V2 (human 2AFC): computed by p11_study.py when the study has >= 10 raters (not here)
Also: SA / VQ, in-range cases, trajectory errors, exclusions, the figure case (median PhysCtrl error among rejected
shifted Phase 11 cases). Output: results/p11_summary/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402

NAMES = {"generation": "PhysCtrl", "certified_v2": "ours", "ground_truth": "reference"}


def boot(x, B=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(1)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run("p11_summary", cfg, synthetic=False)
    S = {}
    for r in read_jsonl(results_dir(cfg["run_name"]) / "scores.jsonl"):
        if not r.get("missing"):
            S[(r["case"], r["variant"])] = r
    cases, excluded = [], []
    for root, _ in cfg["render_roots"]:
        root = Path(os.path.expandvars(root))
        for c in sorted(root.glob("*/case.json")) + sorted(root.glob("*/case_v2.json")):
            rec = json.loads(c.read_text())
            if rec.get("invalid"):
                excluded.append(rec["case"])
        for c in sorted(root.glob("*/case_v2.json")):
            rec = json.loads(c.read_text())
            name = rec["case"]
            shifted = name.endswith("__coeff_0.4") or name.endswith("__force_x3")
            cases.append({"case": name, "phase11": "__coeff_" in name, "shifted": shifted,
                          "rejected": not rec["decision"]["accepted"], "err_gen": rec["errors"]["generation"],
                          "err_ours": rec["errors"]["certified_v2"]})
    def scores(sel, key):
        out = {}
        for v in NAMES:
            vals = [S[(c["case"], v)][key] for c in sel if (c["case"], v) in S and S[(c["case"], v)][key] is not None]
            out[v] = vals
        return out
    md = ["# Phase 11: video-level evaluation (pre-registered Oct 20, 2026)\n",
          f"Cases: {len(cases)} ({sum(c['phase11'] for c in cases)} Phase 11, {sum(not c['phase11'] for c in cases)} PhysCtrl "
          f"examples); excluded (preprocessing, empty drag box or unstable reference): {len(excluded)} {excluded}. Judge: {cfg['judge_model']}.\n"]
    res = {}
    for nm, sel in (("rejected shifted", [c for c in cases if c["shifted"] and c["rejected"]]),
                    ("accepted shifted", [c for c in cases if c["shifted"] and not c["rejected"]]),
                    ("in range", [c for c in cases if not c["shifted"]])):
        md += [f"## {nm} ({len(sel)} cases)", "", "| variant | SA | PC | VQ | trajectory error |", "|---|---|---|---|---|"]
        for v in NAMES:
            cells = []
            for key in ("sa", "pc", "vq"):
                x = scores(sel, key)[v]
                cells.append(f"{np.mean(x):.2f} (n {len(x)})" if x else "—")
            e = [c["err_gen"] if v == "generation" else c["err_ours"] for c in sel] if v != "ground_truth" else [0.0] * len(sel)
            md.append(f"| {NAMES[v]} | " + " | ".join(cells) + f" | {np.mean(e):.4f} |" if sel else f"| {NAMES[v]} | — | — | — | — |")
        md.append("")
        res[nm] = sel
    rs = res["rejected shifted"]
    paired = lambda a, b: [S[(c["case"], a)]["pc"] - S[(c["case"], b)]["pc"] for c in rs                        # noqa: E731
                           if all((c["case"], v) in S and S[(c["case"], v)]["pc"] is not None for v in (a, b))]
    d0, d1 = paired("ground_truth", "generation"), paired("certified_v2", "generation")
    v0, v1 = (boot(d0) if len(d0) > 1 else (np.nan,) * 3), (boot(d1) if len(d1) > 1 else (np.nan,) * 3)
    V0, V1 = bool(v0[1] > 0), bool(v1[1] > 0)
    md += ["## Tests (rejected shifted cases, PC = physical commonsense)", "",
           f"- **V0 judge validity:** PC(reference) - PC(PhysCtrl) {v0[0]:+.2f} [{v0[1]:+.2f}, {v0[2]:+.2f}] (n {len(d0)}): **{'PASS' if V0 else 'FAIL'}**",
           f"- **V1:** PC(ours) - PC(PhysCtrl) {v1[0]:+.2f} [{v1[1]:+.2f}, {v1[2]:+.2f}] (n {len(d1)}): **{'PASS' if V1 else 'FAIL'}**"
           + ("" if V0 else " (uninformative: the judge does not separate the reference from PhysCtrl)"),
           "- V2 (human preference): see p11_study.py.", ""]
    fig = sorted([c for c in rs if c["phase11"]], key=lambda c: (c["err_gen"], c["case"]))
    figc = fig[(len(fig) - 1) // 2]["case"] if fig else None
    md.append(f"Figure case (median PhysCtrl error among rejected shifted Phase 11 cases): {figc}")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"V0": V0, "V1": V1, "v0": v0, "v1": v1, "figure_case": figc, "n_cases": len(cases), "excluded": excluded},
              run.path("summary.json"))
    print(f"[p11] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
