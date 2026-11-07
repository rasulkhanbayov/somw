"""Paper figure: how well each score ranks generation error (Oct 20, 2026, after the paper review). CPU, existing data.

Risk-coverage curves from Phase 7a (results/p7a_metrics/risk_coverage.json, pooled in-distribution + shift pool of
Phase 4m, three-window score of the earlier certificate): accepting the k lowest-scored generations, the fraction of
them that are bad. Reference lines: random acceptance (the overall bad rate; the random-deferral floor of Hybrid
Neural World Models) and a perfect ranking. Legend: AUROC and AURC from results/p7a_metrics/summary.json.
Output: paper/fig/certificate.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLE = {"window (ours)": ("ours", "#2a78d6", "-", 1.8),
         "agreement (needs coarse rollout)": ("coarse agreement", "#1baf7a", "--", 1.1),
         "momentum residual": ("momentum residual", "#eb6834", "-.", 1.1),
         "seed spread": ("seed spread", "#8a8984", ":", 1.3)}


def main():
    curves = json.loads((ROOT / "results/p7a_metrics/risk_coverage.json").read_text())
    sel = {(s, k): (au, aurc) for s, k, au, aurc in json.loads((ROOT / "results/p7a_metrics/summary.json").read_text())["selective"]}
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.3, 1.7))
    n = len(curves["pooled|window (ours)"])
    cov = np.arange(1, n + 1) / n
    base = curves["pooled|window (ours)"][-1]
    nb = int(round(base * n))
    ax.plot(cov, np.full(n, base), color="#C9CED6", lw=1, label=f"random acceptance ({base:.2f})")
    ax.plot(cov, np.maximum(0, (np.arange(1, n + 1) - (n - nb)) / np.arange(1, n + 1)), color="#14213D", lw=0.8,
            ls=(0, (1, 1)), label="perfect ranking")
    for k, (lab, c, ls, lw) in STYLE.items():
        au, aurc = sel[("pooled", k)]
        ax.plot(cov, curves[f"pooled|{k}"], color=c, ls=ls, lw=lw, label=f"{lab} ({au:.3f})")
    ax.set_xlabel("coverage (fraction of generations accepted)")
    ax.set_ylabel("risk among accepted")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.62)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.grid(axis="y", color="#e6e5e0", lw=0.5)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, handlelength=2.2, columnspacing=1.0)
    out = ROOT / "paper" / "fig"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"certificate.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print("wrote", out / "certificate.pdf", "| n", n, "| bad rate", round(base, 3))


if __name__ == "__main__":
    main()
