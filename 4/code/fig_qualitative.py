"""Qualitative figures for the results deck (Oct 14, 2026). CPU only; reads existing results.

Examples are chosen by a fixed rule, not by eye, so they are typical rather than flattering:
  failure_modes.png  per split, the Phase 2c query with the MEDIAN generation error (4h rows);
                     ground truth vs base generation, side view (x, y; gravity is -y).
  timeline.png       among the 135 Phase 4h/4i suffix queries, the one with the MEDIAN repaired
                     (4i V2) error: the chain's window defects vs tau, and the per-frame error of
                     generation vs certify + repair.
  pareto.png         risk vs cost of every pipeline: Phase 4k (fresh, in distribution) and the
                     exploratory Phase 2c view (300 queries incl. OOD), from the summaries.
Output: results/fig_qualitative/*.png and selection.json
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
from common import load_config, read_jsonl, results_dir  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_summarize import query_inputs  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402

GT, GEN, REP = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#14213D", "#5B6472", "#DDE2E8", "#FBFBF8"
FRAME_DT = 0.0417
SPLITS = [("test_id", "In distribution"), ("ood_E_low", "Softer material"),
          ("ood_force_x3", "3× drag force"), ("ood_gravity_drop", "Drop onto the floor")]
plt.rcParams.update({"font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})


def median_row(rows, key):
    rows = sorted(rows, key=key)
    return rows[(len(rows) - 1) // 2]


def main():
    cfg = load_config("configs/p4i_state.yaml")
    root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    oi = ood_index_from(["p2_build_ood"])
    out = results_dir("p4h_suffix").parent / "fig_qualitative"
    out.mkdir(exist_ok=True)
    h4 = [r for f in sorted(results_dir("p4h_suffix").glob("shard_*/suffix.jsonl")) for r in read_jsonl(f)]
    sel = {}

    # 1. failure modes
    ks = [0, 7, 15, 23]
    fig, axes = plt.subplots(len(SPLITS), len(ks), figsize=(16, 15.5), constrained_layout=True)
    for i, (sp, label) in enumerate(SPLITS):
        r = median_row([x for x in h4 if x["split"] == sp], lambda x: np.mean(x["gen_err_t"]))
        q = query_inputs(r["qid"], sp, root, oi)
        g = np.load(groot / "base" / sp / f"{r['qid']}.npz")["samples"][0]
        sel[sp] = {"qid": r["qid"], "gen_err": float(np.mean(r["gen_err_t"])), "rule": "median generation error in split"}
        pts = np.concatenate([q["gt"][ks].reshape(-1, 3), g[ks].reshape(-1, 3)])
        lo, hi = pts.min(0), pts.max(0)
        c, half = (lo + hi) / 2, (hi - lo)[:2].max() / 2 + 0.25
        for j, k in enumerate(ks):
            ax = axes[i, j]
            ax.scatter(q["gt"][k][:, 0], q["gt"][k][:, 1], s=1.2, c=GT, alpha=0.55, linewidths=0)
            ax.scatter(g[k][:, 0], g[k][:, 1], s=1.2, c=GEN, alpha=0.55, linewidths=0)
            if q["gravity"]:
                ax.axhline(q["floor_height"], color=MUTED, lw=2)
            ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            err = float(np.linalg.norm(g[k] - q["gt"][k], axis=1).mean())
            ax.set_title(f"t = {(2 * k + 1) * FRAME_DT:.2f} s   error {err:.3f}", fontsize=15, color=MUTED)
            if j == 0:
                ax.set_ylabel(label, fontsize=19, color=INK, labelpad=10)
    fig.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=10, color=GT, label="ground truth (exact simulator)"),
                        plt.Line2D([], [], ls="", marker="o", ms=10, color=GEN, label="PhysCtrl base generation")],
               loc="lower center", ncol=2, frameon=False, fontsize=18, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(out / "failure_modes.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 2. timeline of one repaired query
    v = {r["qid"]: r for f in sorted(results_dir("p4i_state").glob("shard_*/state.jsonl")) for r in read_jsonl(f)}
    sfx = [x for x in h4 if x["mode"] == "suffix"]
    r = median_row(sfx, lambda x: np.mean(v[x["qid"]]["v2_err_t"]))
    x = v[r["qid"]]
    scan = next(s for f in sorted(results_dir("p2c_scan").glob("shard_*/scan.jsonl")) for s in read_jsonl(f)
                if s["qid"] == r["qid"] and s["source"] == "base" and s["fidelity"] == "coarse_cfl" and int(s["m"]) == 4)
    win = {s: (np.inf if (b or c or raw is None) else raw) for s, raw, b, c in zip(scan["starts"], scan["raw"], scan["blown"], scan["crashed"])}
    chain_k = [0, 3, 7, 11, 15, 19]
    starts = [0] + [2 * k + 1 for k in chain_k[1:]]
    tau = float(cfg["tau"])
    sel["timeline"] = {"qid": r["qid"], "split": r["split"], "k_j": r["k_j"], "v2_err": float(np.mean(x["v2_err_t"])),
                       "gen_err": float(np.mean(r["gen_err_t"])), "rule": "median repaired error among 4h suffix queries"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1, 1.3]})
    t = (2 * np.arange(24) + 1) * FRAME_DT
    for k, s in zip(chain_k, starts):
        d = win[s]; ok = d < tau
        a1.bar((2 * k + 1) * FRAME_DT + 2 * FRAME_DT * 2, min(d, 0.2), width=4 * FRAME_DT * 0.9, color=GT if ok else GEN, alpha=0.9)
        a1.text((2 * k + 1) * FRAME_DT + 4 * FRAME_DT, min(d, 0.2) + 0.002, "pass" if ok else "fail", ha="center", va="bottom",
                fontsize=15, color=INK)
    a1.axhline(tau, color=INK, lw=1.5, ls="--")
    a1.text(t[-1] + 0.02, tau + 0.002, " threshold τ", va="bottom", fontsize=15, color=INK)
    a1.set_ylim(0, max(min(win[s], 0.2) for s in starts) * 1.25)
    a1.set_ylabel("window defect"); a1.spines[["top", "right"]].set_visible(False)
    a1.set_title("Probe windows: the first failure decides where re-simulation starts", fontsize=17, loc="left", pad=14)
    a2.plot(t, r["gen_err_t"], color=GEN, lw=2.5)
    a2.plot(t, x["v2_err_t"], color=REP, lw=2.5)
    a2.text(t[-1] + 0.02, r["gen_err_t"][-1], " generation", va="center", color=INK, fontsize=15)
    a2.text(t[-1] + 0.02, x["v2_err_t"][-1], " certify + repair", va="center", color=INK, fontsize=15)
    tk = (2 * r["k_j"] + 1) * FRAME_DT
    a2.axvline(tk, color=MUTED, lw=1.5, ls=":")
    a2.text(tk + 0.02, max(r["gen_err_t"]) * 0.95, "re-simulation starts here", color=MUTED, fontsize=15, va="top")
    a2.set_xlabel("time (s)"); a2.set_ylabel("mean point error"); a2.spines[["top", "right"]].set_visible(False)
    a2.grid(axis="y", color=GRID)
    a2.set_xlim(0, t[-1] + 0.45)
    fig.savefig(out / "timeline.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 3. risk vs cost (numbers from the summaries; 2c view recomputed by explore_gated_2c.py)
    k4 = json.loads((results_dir("p4k_gated") / "summary.json").read_text())["by_tau"][str(float(cfg["tau"]))]
    c_ref = 4.666                       # Phase 1 reference rollout (A100 s)
    # Phase 2c view: the values printed by code/explore_gated_2c.py (recorded in PLAN.md "Phase 4k result")
    pts4 = {"generate only": ("gen",), "coarse simulation only": ("coarse",), "certify or simulate": ("cert",),
            "certify + repair": ("gated",)}
    ood = {"generate only": (0.483, 0.236 / c_ref), "coarse simulation only": (0.210, 0.14),
           "certify or simulate": (0.013, 0.84), "certify + repair": (0.017, 0.76)}
    colors = {"generate only": GEN, "coarse simulation only": "#eda100", "certify or simulate": GT, "certify + repair": REP}
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True, constrained_layout=True)
    for ax, title, data in ((axs[0], "Fresh queries, in distribution (Phase 4k)", {n: (k4["rate"][k[0]], k4["cost"][k[0]] / c_ref) for n, k in pts4.items()}),
                            (axs[1], "Phase 2c queries incl. shift (exploratory)", ood)):
        # label offsets (points) and alignment, set so the three low-risk labels do not collide
        off = ({"generate only": (10, 6, "left"), "coarse simulation only": (10, -28, "left"),
                "certify + repair": (0, 22, "center"), "certify or simulate": (0, -30, "center")} if ax is axs[0] else
               {"generate only": (10, 6, "left"), "coarse simulation only": (10, 6, "left"),
                "certify + repair": (-8, 22, "right"), "certify or simulate": (0, -30, "center")})
        for n, (risk, cost) in data.items():
            ax.scatter(cost, risk, s=160, color=colors[n], edgecolors=SURF, linewidths=2, zorder=3)
            dx, dy, ha = off[n]
            ax.annotate(n, (cost, risk), xytext=(dx, dy), textcoords="offset points", fontsize=15, color=INK, ha=ha)
        ax.scatter(1.0, 0.0, s=160, color=MUTED, edgecolors=SURF, linewidths=2, zorder=3)
        ax.annotate("always simulate", (1.0, 0.0), xytext=(8, 22), textcoords="offset points", fontsize=15, color=INK, ha="center")
        ax.set_title(title, fontsize=17, loc="left"); ax.set_xlabel("cost vs always simulate")
        ax.set_xlim(-0.02, 1.2); ax.set_ylim(-0.06, 0.52); ax.grid(color=GRID); ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("risk: share with error > 0.10")
    fig.savefig(out / "pareto.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
