"""Paper teaser (Fig. 1; Oct 20, 2026, after the paper review). CPU, existing data only.

(a) The rule-selected contact scene of the paper's contact figure (results/fig_qualitative/paper_selection.json,
    'contact_switch': the median rejected scene with a mid-clip switch, fixed before any picture), side view at model
    frames 0, 8, 16, 23 (evenly spaced; first draft used 0, 12, 23, widened for layout only): PhysCtrl / ours / reference. Points are
    coloured by their distance to the reference at that frame; the reference shape is drawn in light grey behind.
(b) Every pipeline of the main table summarised over the three Phase 9 test sets: largest gap to simulating every query
    (over the sets and the three reference steps of Phase 10c; changed Oct 21 from worst risk, which held against dt
    only) against largest cost (results/p9_table/table.json; contact cells of seed spread, residual gate and coarse from
    Phase 7d, as in the table).
Output: paper/fig/teaser.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_deck_points import axis_h, proj  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, INK, GREY, MUTED = "#2a78d6", "#14213D", "#8a8984", "#52514e"
VMAX, SAFE = 0.4, 0.03
FRAMES = [0, 8, 16, 23]
SHORT = {"generate only": "PhysCtrl", "coarse only": "coarse", "always M1": "M1", "always M2": "M2", "always M3": "M3",
         "always reference": "reference", "seed-spread deferral": "seed spread", "momentum-residual gate": "residual gate",
         "certify or simulate": "certify-or-simulate", "ours": "ours"}


def p7d_contact():
    out = {}
    names = {"coarse simulation only": "coarse only", "seed-spread deferral (8 samples)": "seed-spread deferral",
             "momentum-residual gate": "momentum-residual gate"}
    for line in (ROOT / "results/p7d_contact/SUMMARY.md").read_text().splitlines():
        c = [x.strip(" *") for x in line.strip().strip("|").split("|")]
        if len(c) >= 4 and c[0] in names:
            out[names[c[0]]] = (float(c[2]), float(c[3]))
    return out


def main():
    d = dict(np.load(ROOT / "results/fig_qualitative/paper_contact_switch.npz"))
    k_sw = int(json.loads((ROOT / "results/fig_qualitative/paper_selection.json").read_text())["cases"][-1]["k"])
    gt = d["gt"]
    h = axis_h(gt)
    fig = plt.figure(figsize=(7.0, 2.45))
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 0.7, 3.2], wspace=0.04, hspace=0.06)
    rows = [("gen", "PhysCtrl"), ("ours", "ours"), ("gt", "reference")]
    px = np.concatenate([proj(d[k][f], h)[0] for k, _ in rows for f in FRAMES])
    py = np.concatenate([proj(d[k][f], h)[1] for k, _ in rows for f in FRAMES])
    bx, by = np.percentile(px, [0.5, 99.5]), np.percentile(py, [0.5, 99.5])
    c = (bx.mean(), by.mean())
    r = max(bx[1] - bx[0], by[1] - by[0]) / 2 * 1.08 + 0.05
    for i, (k, lab) in enumerate(rows):
        for j, f in enumerate(FRAMES):
            a = fig.add_subplot(gs[i, j])
            gx, gy = proj(gt[f], h)
            a.scatter(gx, gy, s=0.15, c="#D5D9E0", lw=0, rasterized=True)
            e = np.linalg.norm(d[k][f] - gt[f], axis=1)
            qx, qy = proj(d[k][f], h)
            a.scatter(qx, qy, s=0.15, c=np.minimum(e, VMAX), cmap="plasma", vmin=0, vmax=VMAX, lw=0, rasterized=True)
            a.set_xlim(c[0] - r, c[0] + r); a.set_ylim(c[1] - r, c[1] + r); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_color(BLUE if k == "ours" else "#C9CED6"); s.set_linewidth(1.2 if k == "ours" else 0.5)
            if i == 0:
                a.set_title(f"frame {f + 1}", fontsize=6.5, color=INK, pad=2)
            if j == 0:
                a.set_ylabel(lab, fontsize=7, color=BLUE if k == "ours" else INK)
            if k == "ours" and j == 0:
                a.text(0.03, 0.04, f"switch: frame {k_sw + 1}", transform=a.transAxes, fontsize=5.3, color=BLUE, va="bottom")
            if k != "gt" and f == FRAMES[-1]:
                m = float(np.linalg.norm(d[k] - gt, axis=2).mean())
                a.text(0.97, 0.04, f"error {m:.3f}", transform=a.transAxes, ha="right", va="bottom", fontsize=5.5,
                       color="white", bbox=dict(boxstyle="round,pad=0.2", fc=BLUE if k == "ours" else INK, ec="none"))
    fig.text(0.02, 0.98, "(a) two objects collide", fontsize=7, fontweight="bold", color=INK, va="top")

    TJ = json.loads((ROOT / "results/p9_table/table.json").read_text())
    T, GAP = TJ["table"], TJ["max_gap"]
    extra = p7d_contact()
    ax = fig.add_subplot(gs[:, 5])
    ax.set_yscale("function", functions=(lambda y: np.sqrt(np.clip(y, 0, None)), np.square))
    ax.axhspan(0, SAFE, color=BLUE, alpha=0.08, lw=0)
    off = {"generate only": (4, -2, "left"), "coarse only": (4, -2, "left"), "always M1": (-4, 2, "right"),
           "always M2": (4, 2, "left"), "always M3": (4, -2, "left"), "always reference": (4, 3, "left"),
           "seed-spread deferral": (-4, 2, "right"), "momentum-residual gate": (-4, 3, "right"),
           "certify or simulate": (4, 4, "left"), "ours": (-5, 4, "right")}
    for k, lab in SHORT.items():
        vals = [(T[b][k]["risk"], T[b][k]["cost"]) if k in T[b] else extra[k] for b in T]
        w, mc = max(GAP[k], 0.0), max(v[1] for v in vals)
        o = k == "ours"
        ax.scatter(mc, w, s=40 if o else 16, c=BLUE if o else GREY, zorder=3, edgecolors="white", linewidths=0.8, clip_on=False)
        dx, dy, ha = off[k]
        ax.annotate(lab, (mc, w), xytext=(dx, dy), textcoords="offset points", fontsize=6, ha=ha,
                    color=INK if o else MUTED, fontweight="bold" if o else "normal")
    ticks = [0, 0.03, 0.1, 0.2, 0.4, 0.7]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.set_ylim(0, 0.75)
    ax.set_xlim(0, 1.8); ax.axvline(1.0, color="#C9CED6", lw=0.6, ls="--")
    ax.set_xlabel("largest cost over the three test sets\n(x simulating every query)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("largest risk gap to simulating every query\n(over the test sets and three reference steps)", fontsize=6.0, labelpad=1)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1.78, SAFE * 1.15, "within 0.03 of simulating all", ha="right", va="bottom", fontsize=5.5, color=BLUE)
    ax.set_title("(b) in distribution, under shift and on contact", fontsize=7, fontweight="bold", color=INK, pad=3, loc="left")
    out = ROOT / "paper" / "fig"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"teaser.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", out / "teaser.pdf", "| switch frame", k_sw)


if __name__ == "__main__":
    main()
