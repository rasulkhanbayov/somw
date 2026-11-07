"""Deck figure: rendered videos of all five 3x-force PhysCtrl examples (the cases the certificate rejects).

Rows: PhysCtrl generation | ours (certify + repair) | exact simulation. Columns: the 5 rejected cases of Phase 6a.
Frame rule (fixed before viewing): frame 24 of 49, the clip midpoint, for every case. Errors are the Phase 6a
numbers recorded in PLAN.md (passed in, not recomputed). CPU.
Output: results/fig_qualitative/deck_videos.png

Earphone strip (added Oct 16, after a reader found frame 24 misleading: PhysCtrl's earphone looks cleaner but
moves wrongly). The earphone is the largest-error case of the five, chosen for that reason, not by a rule.
Columns = video frames 1, 13, 25, 37 (= model frames 0, 6, 12, 18; by frame 48 the object has left the view),
each labelled with its per-frame error from trajectories.npz; last column = error per frame.
Output: results/fig_qualitative/deck_video_earphone.png
"""
from __future__ import annotations

import os
from pathlib import Path

import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FRAME = 24
CASES = [("blanket", 0.089, 0.029), ("chair", 0.061, 0.038), ("earphone", 0.222, 0.010),
         ("giraffeX", 0.086, 0.027), ("penguin", 0.073, 0.017)]          # PLAN.md, Phase 6a result
ROWS = [("generation", "PhysCtrl"), ("certified", "Ours"), ("ground_truth", "Exact simulation")]


def frame(path, k, w=720, h=480):
    """Frame k as an RGB array, through the system ffmpeg (the env has no PyAV)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{k})", "-vframes", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def main():
    root = Path(os.path.expandvars("${BIG_ROOT}/renders"))
    fig, ax = plt.subplots(3, 5, figsize=(17.5, 7.6), gridspec_kw=dict(wspace=0.03, hspace=0.06))
    for j, (c, eg, eo) in enumerate(CASES):
        for i, (v, lab) in enumerate(ROWS):
            im = frame(root / f"{c}__force_x3" / "variants" / v / "result.mp4", FRAME)
            a = ax[i, j]
            a.imshow(im)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(i == 1); s.set_color("#2F6FB0"); s.set_linewidth(3)
            if j == 0:
                a.set_ylabel(lab, fontsize=15, color="#2F6FB0" if i == 1 else "#14213D")
            err = {0: f"error {eg:.3f}", 1: f"error {eo:.3f}", 2: "reference"}[i]
            a.text(8, 470, err, fontsize=12, color="white", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.25", fc="#14213D" if i != 1 else "#2F6FB0", ec="none", alpha=0.85))
        ax[0, j].set_title(f"{c} · 3× force", fontsize=15, color="#14213D")
    out = Path("results/fig_qualitative/deck_videos.png")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def earphone_strip():
    root = Path(os.path.expandvars("${BIG_ROOT}/renders/earphone__force_x3"))
    tr = np.load(root / "trajectories.npz")
    err = {v: np.linalg.norm(tr[v] - tr["ground_truth"], axis=2).mean(axis=1) for v in ("generation", "certified")}
    model_k = [0, 6, 12, 18]
    fig = plt.figure(figsize=(21, 8.2))
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 0.22, 1.25], wspace=0.04, hspace=0.06)
    for i, (v, lab) in enumerate(ROWS):
        for j, k in enumerate(model_k):
            a = fig.add_subplot(gs[i, j])
            a.imshow(frame(root / "variants" / v / "result.mp4", 2 * k + 1)[60:420, 120:600])
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(i == 1); sp.set_color("#2F6FB0"); sp.set_linewidth(3)
            if i == 0:
                a.set_title(f"t = {(2 * k + 1) / 24:.2f} s", fontsize=15, color="#14213D")
            if j == 0:
                a.set_ylabel(lab, fontsize=16, color="#2F6FB0" if i == 1 else "#14213D")
            if v != "ground_truth":
                a.text(8, 352, f"error {err[v][k]:.3f}", fontsize=13, color="white", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.25", fc="#2F6FB0" if i == 1 else "#14213D", ec="none", alpha=.85))
    a = fig.add_subplot(gs[:, 5])
    t = (2 * np.arange(24) + 1) / 24
    a.plot(t, err["generation"], color="#D9762B", lw=2.4, label="PhysCtrl")
    a.plot(t, err["certified"], color="#2F6FB0", lw=2.8, label="Ours")
    a.axhline(0.1, color="#6A7280", ls=":", lw=1.4)
    a.text(1.95, 0.088, "failure line 0.10", fontsize=12, color="#6A7280", ha="right", va="top")
    k_rep = 3                                   # Phase 6a: rejected, repaired from model frame 3 (PLAN.md)
    a.axvline((2 * k_rep + 1) / 24, color="#2F6FB0", lw=1, alpha=.6)
    a.text((2 * k_rep + 1) / 24 + 0.04, 0.33, "check fails here:\nrest re-simulated", fontsize=12, color="#2F6FB0")
    a.set_xlabel("time (s)", fontsize=14); a.set_ylabel("error vs exact simulation", fontsize=14)
    a.set_ylim(0, None); a.tick_params(labelsize=12); a.spines[["top", "right"]].set_visible(False)
    a.legend(fontsize=14, frameon=False, loc="upper left")
    a.set_title("error over the clip", fontsize=15, color="#14213D")
    out = Path("results/fig_qualitative/deck_video_earphone.png")
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    print("wrote", out, "| last-frame error: PhysCtrl", round(float(err["generation"][-1]), 3),
          "ours", round(float(err["certified"][-1]), 3))


if __name__ == "__main__":
    main()
    earphone_strip()
