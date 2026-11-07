"""Paper figure: rendered video, PhysCtrl vs ours (Phase 9 pipeline) vs exact simulation (Oct 19, 2026). CPU, after job 95.

Case: the Phase 6a rule, fixed Oct 15 before any render ("among the cases the certificate rejects, the one with the
median generation error"), applied to ours-v2's decisions in $BIG_ROOT/renders/<case>/case_v2.json (lower median).
Frames: video frames 1, 11, 21, 31 (model frames 0, 5, 10, 15), evenly spaced over the first two thirds of the clip:
under 3x force the object leaves the camera view in the last third (Phase 6a note). Each frame is labelled with its per-frame error; last column = error over
the clip, with the repair point when ours-v2 repaired mid-clip.
Outputs: paper/fig/qual_video.{pdf,png}; results/fig_qualitative/paper_video_selection.json

--stand-in DIR draws the Phase 6a 'certified' variant instead (layout check only; output to DIR).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_deck_video import frame  # noqa: E402

BLUE, INK, ORANGE, GREY = "#2a78d6", "#14213D", "#D9762B", "#6A7280"
MODEL_K = [0, 5, 10, 15]          # evenly spaced over the first two thirds (see docstring)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stand-in", default=None)
    args = ap.parse_args()
    root = Path(os.path.expandvars("${BIG_ROOT}/renders"))
    var, rec, tkey = ("certified", "case.json", "certified") if args.stand_in else ("certified_v2", "case_v2.json", "certified_v2")
    cases = sorted(p.parent.name for p in root.glob(f"*/{rec}"))
    recs = {c: json.loads((root / c / rec).read_text()) for c in cases}
    rej = sorted(((c, r["errors"]["generation"]) for c, r in recs.items() if not r["decision"]["accepted"]), key=lambda t: (t[1], t[0]))
    if not rej:
        raise RuntimeError("no rejected case: the figure rule selects nothing")
    case = rej[(len(rej) - 1) // 2][0]
    dec = recs[case]["decision"]
    tr = np.load(root / case / "trajectories.npz")
    ours = np.load(root / case / "trajectories_v2.npz")[tkey] if not args.stand_in else tr["certified"]
    gt = tr["ground_truth"]
    err = {"PhysCtrl": np.linalg.norm(tr["generation"] - gt, axis=2).mean(axis=1), "ours": np.linalg.norm(ours - gt, axis=2).mean(axis=1)}
    rows = [("generation", "PhysCtrl"), (var, "ours"), ("ground_truth", "reference simulation")]
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 0.32, 1.25], wspace=0.04, hspace=0.06)
    for i, (v, lab) in enumerate(rows):
        for j, k in enumerate(MODEL_K):
            a = fig.add_subplot(gs[i, j])
            a.imshow(frame(root / case / "variants" / v / "result.mp4", 2 * k + 1)[60:420, 120:600])
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(i == 1); sp.set_color(BLUE); sp.set_linewidth(1.4)
            if i == 0:
                a.set_title(f"t = {(2 * k + 1) / 24:.2f} s", fontsize=7, color=INK, pad=2)
            if j == 0:
                a.set_ylabel(lab, fontsize=7, color=BLUE if i == 1 else INK)
            if i < 2:
                e = err["PhysCtrl" if i == 0 else "ours"][k]
                a.text(6, 352, f"{e:.3f}", fontsize=6, color="white", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.2", fc=BLUE if i == 1 else INK, ec="none", alpha=.85))
    a = fig.add_subplot(gs[:, 5])
    t = (2 * np.arange(24) + 1) / 24
    a.plot(t, err["PhysCtrl"], color=ORANGE, lw=1.2, label="PhysCtrl")
    a.plot(t, err["ours"], color=BLUE, lw=1.6, label="ours")
    a.axhline(0.1, color=GREY, ls=":", lw=0.8)
    if dec.get("restart_k") is not None:
        a.axvline((2 * dec["restart_k"] + 1) / 24, color=BLUE, lw=0.8, alpha=.6)
    a.set_xlabel("time (s)", fontsize=6.5, labelpad=1); a.set_title("error vs reference", fontsize=7, color=INK, pad=2)
    a.set_ylim(0, None); a.tick_params(labelsize=6, length=2, pad=1); a.spines[["top", "right"]].set_visible(False)
    a.legend(fontsize=6, frameon=False, loc="upper left")
    out = Path(args.stand_in) if args.stand_in else Path("paper/fig")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"qual_video.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    sel = {"rule": "Phase 6a (fixed Oct 15): median generation error among the cases the certificate rejects (lower median)",
           "pipeline": "stand-in (Phase 6a certified)" if args.stand_in else "ours-v2 (Phase 9)", "selected": case,
           "rejected": [c for c, _ in rej], "decision": dec["action"], "frames_video": [2 * k + 1 for k in MODEL_K],
           "mean_error": {k: float(v.mean()) for k, v in err.items()}}
    (out / "paper_video_selection.json" if args.stand_in else Path("results/fig_qualitative/paper_video_selection.json")).write_text(
        json.dumps(sel, indent=1))
    print("selected", case, dec["action"], {k: round(float(v.mean()), 4) for k, v in err.items()})


if __name__ == "__main__":
    main()
