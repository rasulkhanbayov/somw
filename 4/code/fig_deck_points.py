"""Deck figures: point-cloud comparison of our output against the baselines (Oct 16, 2026). CPU.

Reads results/fig_qualitative/ours_<name>.npz (written by fig_ours_traj.py on a GPU) for the cases fixed in
deck_selection.json, and draws the last model frame (sim frame 47) in a side view: y up, horizontal axis =
the exact trajectory's mean horizontal displacement direction (x when it moves < 0.05). Points are coloured by
their distance to the exact simulation; the exact final shape is drawn in light grey behind every panel.
The last column is the mean error per frame. Risk threshold eps = 0.1 is drawn as a dashed line.
  shift    rows soft / force / drop; columns exact | PhysCtrl | coarse simulation only | ours
  contact  rows median scene / mid-switch scene; columns exact | PhysCtrl (two objects composed) |
           simulated separately (gt_super: no contact) | ours
Output: results/fig_qualitative/deck_points_{shift,contact}.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EPS, VMAX = 0.1, 0.4
INK, BLUE, ORANGE, GREY = "#14213D", "#2F6FB0", "#D9762B", "#6A7280"
LABEL = {"shift_soft": "Softer material", "shift_force": "Stronger force", "shift_drop": "Drop onto floor",
         "contact_median": "Contact: median scene", "contact_switch": "Contact: mid-clip switch"}


def axis_h(gt):
    d = (gt[-1] - gt[0]).mean(axis=0)[[0, 2]]
    return d / np.linalg.norm(d) if np.linalg.norm(d) > 0.05 else np.array([1.0, 0.0])


def proj(p, h):
    return p[:, 0] * h[0] + p[:, 2] * h[1], p[:, 1]


def draw(fig_rows, cols, out, curve_keys):
    n = len(fig_rows)
    fig, ax = plt.subplots(n, len(cols) + 1, figsize=(4.0 * (len(cols) + 1), 3.6 * n),
                           gridspec_kw=dict(wspace=0.08, hspace=0.32))
    ax = np.atleast_2d(ax)
    sc = None
    for i, (name, d) in enumerate(fig_rows):
        gt = d["gt"]
        h = axis_h(gt)
        # one square view per row, wide enough for every column's last frame
        px = np.concatenate([proj(d[k][-1], h)[0] for k, _ in cols])
        py = np.concatenate([proj(d[k][-1], h)[1] for k, _ in cols])
        bx, by = np.percentile(px, [0.5, 99.5]), np.percentile(py, [0.5, 99.5])
        c = (bx.mean(), by.mean())
        r = max(bx[1] - bx[0], by[1] - by[0]) / 2 * 1.15 + 0.05
        for j, (k, lab) in enumerate(cols):
            a = ax[i, j]
            gx, gy = proj(gt[-1], h)
            a.scatter(gx, gy, s=2, c="#D5D9E0", lw=0)
            e = np.linalg.norm(d[k][-1] - gt[-1], axis=1)
            px, py = proj(d[k][-1], h)
            sc = a.scatter(px, py, s=2, c=np.minimum(e, VMAX), cmap="plasma", vmin=0, vmax=VMAX, lw=0)
            a.set_xlim(c[0] - r, c[0] + r); a.set_ylim(c[1] - r, c[1] + r); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
            m = float(np.linalg.norm(d[k] - gt, axis=2).mean())
            col = BLUE if k == "ours" else INK
            a.set_title(f"{lab}\nerror {m:.3f}" if k != "gt" else f"{lab}\nreference", fontsize=13, color=col)
            for s in a.spines.values():
                s.set_color(BLUE if k == "ours" else "#C9CED6"); s.set_linewidth(3 if k == "ours" else 1)
            if j == 0:
                a.set_ylabel(LABEL[name], fontsize=14, color=INK)
        a = ax[i, -1]
        t = np.arange(1, 25)
        for k, lab, cc, ls in curve_keys:
            a.plot(t, np.linalg.norm(d[k] - gt, axis=2).mean(axis=1), color=cc, ls=ls, lw=2.4 if k == "ours" else 1.8,
                   label=lab)
        a.axhline(EPS, color=GREY, ls=":", lw=1.2)
        a.set_xlim(1, 24); a.set_ylim(0, None)
        a.set_title("error per frame", fontsize=13, color=INK)
        a.tick_params(labelsize=10)
        a.spines[["top", "right"]].set_visible(False)
        if "k" in d:
            a.axvline(d["k"] + 1, color=BLUE, lw=1, alpha=.5)
            a.text(d["k"] + 1.3, a.get_ylim()[1] * .55, "repair\nstarts", fontsize=10, color=BLUE, va="top")
    hd, lb = ax[0, -1].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", bbox_to_anchor=(0.82, 0.03 + 0.02 * (3 - n)), ncol=3, fontsize=11, frameon=False)
    cb = fig.colorbar(sc, ax=ax[:, :-1], orientation="horizontal", fraction=0.04 / n, pad=0.04, aspect=60)
    cb.set_label("distance of each point to the exact simulation (last frame); grey = exact shape", fontsize=12)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/fig_qualitative")
    args = ap.parse_args()
    root = Path(args.dir)
    sel = json.loads(Path("results/fig_qualitative/deck_selection.json").read_text())
    rows = {}
    for c in sel["cases"]:
        d = dict(np.load(root / f"ours_{c['name']}.npz"))
        if c["decision"] == "repair":
            d["k"] = int(c["k"])
        if c["pool"] == "4o":
            sup = Path(os.path.expandvars("${BIG_ROOT}/generations/gt_super")) / c["split"] / f"{c['qid']}.npz"
            d["sup"] = np.load(sup)["samples"][0]
        rows[c["name"]] = d
    draw([(n, rows[n]) for n in ("shift_soft", "shift_force", "shift_drop")],
         [("gt", "Exact simulation"), ("gen", "PhysCtrl"), ("coarse", "Coarse simulation only"), ("ours", "Ours")],
         root / "deck_points_shift.png",
         [("gen", "PhysCtrl", ORANGE, "-"), ("coarse", "coarse only", GREY, "--"), ("ours", "ours", BLUE, "-")])
    draw([(n, rows[n]) for n in ("contact_median", "contact_switch")],
         [("gt", "Exact simulation"), ("gen", "PhysCtrl (objects composed)"), ("sup", "Simulated separately"),
          ("ours", "Ours")],
         root / "deck_points_contact.png",
         [("gen", "PhysCtrl", ORANGE, "-"), ("sup", "simulated separately", GREY, "--"), ("ours", "ours", BLUE, "-")])


if __name__ == "__main__":
    main()
