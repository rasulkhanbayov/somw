"""Paper figures: point-cloud comparison on the Phase 9 cases (Oct 19, 2026). CPU, after job 94.

Cases: results/fig_qualitative/paper_selection.json (median rule fixed before any picture). Positions:
results/fig_qualitative/paper_<name>.npz from fig_ours_traj.py (ours = the Phase 9 pipeline's output; m1 = always
simulating at the tuned setting M1). Side view of the last model frame as in fig_deck_points.py; points coloured by
their distance to the exact simulation, the exact final shape in light grey behind; last column = error per frame.
  shift    rows softer / stronger drag / drop; columns exact | PhysCtrl | always M1 | ours
  contact  rows median scene / mid-clip switch; columns exact | PhysCtrl (composed) | simulated separately | ours
Outputs: paper/fig/qual_shift.pdf, paper/fig/qual_contact.pdf (+ .png)

--stand-in DIR draws the deck's npz files from DIR (m1 replaced by the coarse rollout) to check the layout only;
it writes to DIR, never to paper/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_deck_points import axis_h, proj  # noqa: E402

EPS, VMAX = 0.1, 0.4
INK, BLUE, ORANGE, GREY = "#14213D", "#2a78d6", "#D9762B", "#6A7280"
ROWLAB = {"shift_soft": "softer", "shift_force": "stronger drag", "shift_drop": "drop onto floor",
          "contact_median": "median scene", "contact_switch": "mid-clip switch"}


def draw(rows, cols, curves, out_base):
    n = len(rows)
    fig, ax = plt.subplots(n, len(cols) + 2, figsize=(7.0, 1.3 * n + 0.5),
                           gridspec_kw=dict(wspace=0.05, hspace=0.42, width_ratios=[1] * len(cols) + [0.16, 1.35]))
    ax = np.atleast_2d(ax)
    for a in ax[:, -2]:
        a.axis("off")                               # spacer between the point panels and the curves
    sc = None
    for i, (name, d) in enumerate(rows):
        gt = d["gt"]
        h = axis_h(gt)
        px = np.concatenate([proj(d[k][-1], h)[0] for k, _ in cols])
        py = np.concatenate([proj(d[k][-1], h)[1] for k, _ in cols])
        bx, by = np.percentile(px, [0.5, 99.5]), np.percentile(py, [0.5, 99.5])
        c = (bx.mean(), by.mean())
        r = max(bx[1] - bx[0], by[1] - by[0]) / 2 * 1.12 + 0.05
        gx, gy = proj(gt[-1], h)
        for j, (k, lab) in enumerate(cols):
            a = ax[i, j]
            a.scatter(gx, gy, s=0.4, c="#D5D9E0", lw=0, rasterized=True)
            e = np.linalg.norm(d[k][-1] - gt[-1], axis=1)
            qx, qy = proj(d[k][-1], h)
            sc = a.scatter(qx, qy, s=0.4, c=np.minimum(e, VMAX), cmap="plasma", vmin=0, vmax=VMAX, lw=0, rasterized=True)
            a.set_xlim(c[0] - r, c[0] + r); a.set_ylim(c[1] - r, c[1] + r); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
            m = float(np.linalg.norm(d[k] - gt, axis=2).mean())
            ttl = (lab + "\n" if i == 0 else "") + ("reference" if k == "gt" else f"error {m:.3f}")
            a.set_title(ttl, fontsize=7, color=BLUE if k == "ours" else INK, pad=2)
            for s in a.spines.values():
                s.set_color(BLUE if k == "ours" else "#C9CED6"); s.set_linewidth(1.4 if k == "ours" else 0.5)
            if j == 0:
                a.set_ylabel(ROWLAB[name], fontsize=7, color=INK)
        a = ax[i, -1]
        t = np.arange(1, 25)
        for k, lab, cc, ls in curves:
            a.plot(t, np.linalg.norm(d[k] - gt, axis=2).mean(axis=1), color=cc, ls=ls, lw=1.4 if k == "ours" else 1.0, label=lab)
        a.axhline(EPS, color=GREY, ls=":", lw=0.8)
        a.set_xlim(1, 24); a.set_ylim(0, None)
        a.tick_params(labelsize=6, length=2, pad=1)
        a.spines[["top", "right"]].set_visible(False)
        if i == 0:
            a.set_title("error per frame", fontsize=7, color=INK, pad=2)
        if i == n - 1:
            a.set_xlabel("frame", fontsize=6.5, labelpad=1)
        if "k" in d:
            a.axvline(d["k"] + 1, color=BLUE, lw=0.8, alpha=.5)
    ax[0, -1].legend(fontsize=5.5, frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
    l0, l1 = ax[-1, 0].get_position(), ax[-1, len(cols) - 1].get_position()
    cax = fig.add_axes([l0.x0, l0.y0 - 0.075 * 3 / n, l1.x1 - l0.x0, 0.022 * 3 / n])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.set_label("distance of each point to the reference simulation at the last frame (grey: reference shape)", fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", out_base)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stand-in", default=None, help="layout check on the deck npz files; output goes to this dir")
    args = ap.parse_args()
    q = Path("results/fig_qualitative")
    if args.stand_in:
        sel, pre, out = json.loads((q / "deck_selection.json").read_text()), "ours_", Path(args.stand_in)
    else:
        sel, pre, out = json.loads((q / "paper_selection.json").read_text()), "paper_", Path("paper/fig")
    rows = {}
    for c in sel["cases"]:
        d = dict(np.load(q / f"{pre}{c['name']}.npz"))
        if args.stand_in and "coarse" in d:
            d["m1"] = d["coarse"]                     # layout check only
        if c.get("decision") == "repair":
            d["k"] = int(c["k"])
        if c["pool"] in ("4o", "p9c"):
            sup = Path(os.path.expandvars("${BIG_ROOT}/generations/gt_super")) / c["split"] / f"{c['qid']}.npz"
            d["sup"] = np.load(sup)["samples"][0]
        rows[c["name"]] = d
    draw([(n, rows[n]) for n in ("shift_soft", "shift_force", "shift_drop")],
         [("gt", "reference simulation"), ("gen", "PhysCtrl"), ("m1", "always tuned M1"), ("ours", "ours")],
         [("gen", "PhysCtrl", ORANGE, "-"), ("m1", "always M1", GREY, "--"), ("ours", "ours", BLUE, "-")],
         out / "qual_shift")
    draw([(n, rows[n]) for n in ("contact_median", "contact_switch")],
         [("gt", "reference simulation"), ("gen", "PhysCtrl"), ("sup", "simulated apart"), ("ours", "ours")],
         [("gen", "PhysCtrl", ORANGE, "-"), ("sup", "simulated apart", GREY, "--"), ("ours", "ours", BLUE, "-")],
         out / "qual_contact")


if __name__ == "__main__":
    main()
