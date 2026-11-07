"""Paper figure: method diagram (Oct 20, 2026). CPU, no data. Output: paper/fig/method.{pdf,png}

Top row, the accept path: query -> generate -> check -> certify -> accept -> render.
Bottom row, the repair path: locate the first failing chain window k -> shadow from the true start to k -> re-simulate
frames k..24 (single objects at M2; at contact, the switch to M3 timed by the shadow) -> render.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, BLUE, ORANGE = "#14213D", "#4A5568", "#2a78d6", "#C2410C"
FILL, FILL_R, FILL_B = "#EEF1F4", "#FBEFE4", "#E3ECF6"
W, H = 1.72, 0.80                                     # box size in axis units


def box(ax, x, y, title, body, fill=FILL, edge=None, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill,
                                ec=edge or fill, lw=1.4 if edge else 0))
    ax.text(x + 0.08, y + H - 0.1, title, fontsize=7.0, fontweight="bold", color=tc, va="top")
    ax.text(x + 0.08, y + H - 0.29, body, fontsize=5.9, color=MUTED, va="top", linespacing=1.25)


def arrow(ax, p, q, color=MUTED, label=None, lpos=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=6.5, color=color, fontweight="bold", ha="center")


def main():
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(7.0, 2.15))
    ax.set_xlim(0, 11.3); ax.set_ylim(0, 2.3); ax.axis("off")
    top, bot = 1.38, 0.18
    xs = [0.05 + 1.88 * i for i in range(6)]
    box(ax, xs[0], top, "Query", "object points,\nmaterial, push,\ngravity, floor")
    box(ax, xs[1], top, "1  Generate", "PhysCtrl (frozen)\n24 frames\n0.05$\\times$ cost", tc=BLUE)
    box(ax, xs[2], top, "2  Check", "simulator window\nfrom the generated\nframes: drift $d$", tc=BLUE)
    box(ax, xs[3], top, "3  Certify", "$d < \\tau$ ?\ncalibrated: caps\naccepted failures", fill=FILL_B, edge=BLUE, tc=BLUE)
    box(ax, xs[4], top, "Accept", "keep the\ngenerated motion")
    box(ax, xs[5], top, "7  Render", "PhysCtrl's video\nmodel, unchanged", fill=INK, tc="#E9A35B")
    ax.texts[-1].set_color("#DCE3EC")
    box(ax, xs[3], bot, "4  Locate", "first failed check\nat frame $k$; keep\nframes before $k$", fill=FILL_R, tc=ORANGE)
    box(ax, xs[4], bot, "5  Shadow", "cheap simulation\nfrom the true start\nto $k$ (full state)", fill=FILL_R, tc=ORANGE)
    box(ax, xs[5], bot, "6  Re-simulate", "frames $k$-24 only;\ncontact: switch\nbefore touching", fill=FILL_R, tc=ORANGE)
    yc = top + H / 2
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + W, yc), (b, yc), color=BLUE if a == xs[3] else MUTED)
    ax.text(xs[3] + W + 0.08, top + H + 0.04, "yes", fontsize=6.5, color=BLUE, fontweight="bold", ha="center")
    arrow(ax, (xs[3] + W / 2, top), (xs[3] + W / 2, bot + H), color=ORANGE, label="no", lpos=(xs[3] + W / 2 + 0.15, (top + bot + H) / 2))
    ybc = bot + H / 2
    arrow(ax, (xs[3] + W, ybc), (xs[4], ybc), color=ORANGE)
    arrow(ax, (xs[4] + W, ybc), (xs[5], ybc), color=ORANGE)
    arrow(ax, (xs[5] + W / 2, bot + H), (xs[5] + W / 2, top), color=ORANGE)
    ax.text(0.05, 0.62, "No network is trained.\nOurs: the simulator at\nseveral settings + one\ncalibrated threshold.\nSteps 4-6 run only for\nrejected trajectories.",
            fontsize=6.3, color=MUTED, va="top", linespacing=1.3)
    out = Path(__file__).resolve().parents[1] / "paper" / "fig"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"method.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", out / "method.pdf")


if __name__ == "__main__":
    main()
