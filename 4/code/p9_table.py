"""Phase 9 main table and risk-vs-cost figure (descriptive; no new test). CPU, after the Phase 9 verdicts.

Every pipeline on the three fresh Phase 9 benchmarks (in distribution, shift, contact), with risk, mean error and cost,
each object counted once (as in the verdicts: single objects by object_rate, contact scenes by their first object).
The numbers come from the same rows and pipeline code as results/p9_confirm and results/p9c_summary; the contact
tuned-simulator rows are object-weighted here (p9_confirm prints them query-level).

Outputs: results/p9_table/{SUMMARY.md, table.json}, paper/tab/main_table.tex, paper/fig/tradeoff.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, results_dir, start_run  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402
from p4n_summarize import build, pipes  # noqa: E402
from p9_confirm import assemble, ladder  # noqa: E402

PAPER = Path(__file__).resolve().parents[1] / "paper"
SAFE = 0.03                      # the pre-registered absolute risk bound under shift and on contact (C4, CC1)
ROWS = [  # (key, label in the paper); None = not run on that benchmark
    ("generate only", "PhysCtrl"),
    ("coarse only", "Coarse"),
    ("always M1", "Tuned M1"),
    ("always M2", "Tuned M2"),
    ("always M3", "Tuned M3"),
    ("seed-spread deferral", "Seed-spread deferral"),
    ("momentum-residual gate", "Momentum residual"),
    ("certify or simulate", "Certify or simulate"),
    ("always reference", "Reference (simulate all)"),
    ("ours", "Ours"),
]


def stats(e, c, objs, eps, c_ref):
    bad = (e > eps).astype(float)
    return {"risk": object_rate(bad, objs), "err": object_rate(np.minimum(e, 10.0), objs),
            "cost": object_rate(c, objs) / c_ref, "n_bad": int(bad.sum()), "n": int(len(e)), "n_obj": int(len(set(objs)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="configs/p9_confirm.yaml")
    ap.add_argument("--contact-config", required=True, help="configs/p9c_summary.yaml")
    args = ap.parse_args()
    cfg, ccfg = load_config(args.config), load_config(args.contact_config)
    run = start_run("p9_table", {"single": cfg, "contact": ccfg}, synthetic=False)
    eps, full = float(cfg["eps"]), float(cfg["full_substeps"])

    P, A, c_ref, _, _ = assemble(cfg)
    P["certify or simulate"] = P.pop("certify or simulate (one window)")
    P["ours"] = P.pop("ours-v2")
    O = np.asarray(A["obj"])
    shift = np.array([s.startswith(cfg.get("shift_prefix", "ood3_")) for s in A["split"]])
    T = {"in distribution": {}, "shift": {}, "contact": {}}
    for nm, sel in (("in distribution", ~shift), ("shift", shift)):
        for k, _ in ROWS:
            if k in P:
                T[nm][k] = stats(P[k][0][sel], P[k][1][sel], O[sel], eps, c_ref)

    # certificate false-accept rate on bad generations (descriptive): one window (single objects), three windows (contact)
    FAR = {}
    bad_g, acc_g = np.asarray(A["E"]) > eps, np.asarray(A["S_one"]) < float(cfg["tau"])
    for nm, sel in (("in distribution", ~shift), ("shift", shift)):
        FAR[nm] = {"accepted_bad": int((acc_g & bad_g & sel).sum()), "bad": int((bad_g & sel).sum()),
                   "accepted_good": int((acc_g & ~bad_g & sel).sum()), "good": int((~bad_g & sel).sum())}
    Bd = build(ccfg)
    con = [r for r in Bd["R"] if r["kind"] == "contact"]
    Oc = np.array([r["obj"] for r in con])
    Pc = pipes(con, Bd["cg"], Bd["c_ref"])
    lc = ladder(cfg["contact_full_run"], [r["qid"] for r in con], Bd["c_ref"], full, ("M1", "M2", "M3"))
    cmap = {"generate only": Pc["generate only"], "certify or simulate": Pc["cert_ref"], "always reference": Pc["always reference"],
            "ours": Pc["sf"], "chain-timed repair (4k timing)": Pc["chain"],
            **{f"always {f}": lc[f] for f in ("M1", "M2", "M3")}}
    for k, v in cmap.items():
        T["contact"][k] = stats(v[0], v[1], Oc, eps, Bd["c_ref"])
    assert abs(Bd["c_ref"] - c_ref) < 1e-9
    acc_c, bad_c = np.array([r["acc"] for r in con]), np.array([r["E"] for r in con]) > eps
    FAR["contact"] = {"accepted_bad": int((acc_c & bad_c).sum()), "bad": int(bad_c.sum()),
                      "accepted_good": int((acc_c & ~bad_c).sum()), "good": int((~bad_c).sum())}

    # LaTeX table (restructured Oct 20 after the paper review): rows grouped by kind; per-set risk / cost; then the
    # regime-agnostic summary over the three sets (below: largest gap to simulating every query, largest cost).
    # Contact cells of the three baselines not run on the Phase 9 contact set come from Phase 7d (4o scenes), marked.
    best = {b: min((k for k in T[b] if T[b][k]["risk"] <= SAFE and k in dict(ROWS)), key=lambda k: T[b][k]["cost"]) for b in T}
    p7d = {}
    for line in (results_dir("p7d_contact") / "SUMMARY.md").read_text().splitlines():
        cols = [c.strip(" *") for c in line.strip().strip("|").split("|")]
        if len(cols) >= 4 and cols[0] in ("coarse simulation only", "seed-spread deferral (8 samples)", "momentum-residual gate"):
            p7d[{"coarse simulation only": "coarse only", "seed-spread deferral (8 samples)": "seed-spread deferral",
                 "momentum-residual gate": "momentum-residual gate"}[cols[0]]] = (float(cols[2]), float(cols[3]))
    if len(p7d) != 3:
        raise RuntimeError(f"Phase 7d contact rows not found: {p7d}")
    def rc(b, k):
        if k in T[b]:
            return T[b][k]["risk"], T[b][k]["cost"], False
        return (*p7d[k], True) if b == "contact" else (None, None, False)
    # Oct 21 (after Phase 10c; exploratory): the summary over the three sets is the largest gap to simulating every
    # query at the reference step, risk(pipeline) - risk(always reference), over the sets and the three reference
    # steps (results/p10c_explore). It replaces "worst risk / safe", which held against dt only (10c T2 FAIL).
    G = json.loads((results_dir("p10c_explore") / "explore.json").read_text())
    gname = {"generate only": "PhysCtrl", "coarse only": "coarse", "always M1": "M1", "always M2": "M2", "always M3": "M3",
             "seed-spread deferral": "seed spread", "momentum-residual gate": "residual gate",
             "certify or simulate": "certify or simulate", "always reference": "reference (dt)", "ours": "ours"}
    def gaps(k):
        out, dag = [], False
        for b in T:
            g = G[b].get(gname[k])
            if g is not None:
                out += [v[0] for v in g["gap"].values()]
            else:                                    # contact cells from Phase 7d: risk against dt (reference risk 0 there)
                out.append(rc(b, k)[0]); dag = True
        return max(out), dag
    summ = {k: (*gaps(k), max(rc(b, k)[1] for b in T), any(rc(b, k)[2] for b in T)) for k, _ in ROWS}
    GAP = SAFE
    within = [k for k in summ if summ[k][0] <= GAP and k != "always reference"]
    best_cost = min(within, key=lambda k: summ[k][2])
    def cell(b, k):
        r_, c_, dag = rc(b, k)
        r, c = f"{r_:.3f}", f"{c_:.2f}"
        if dag:
            r, c = r + r"$^\dagger$", c + r"$^\dagger$"
        if r_ > SAFE:
            r = r"\textcolor{gray}{" + r + "}"
        if k == best.get(b):
            r, c = r"\textbf{" + r + "}", r"\textbf{" + c + "}"
        return f"{r} & {c}"
    def scell(k):
        g, gdag, mc, dag = summ[k]
        d = r"$^\dagger$" if gdag else ""
        g_ = f"{max(g, 0):+.3f}{d}".replace("+", "$+$")
        if g > GAP:
            g_ = r"\textcolor{gray}{" + g_ + "}"
        mc_ = f"{mc:.2f}" + (r"$^\dagger$" if dag else "")
        if k == best_cost:
            g_, mc_ = r"\textbf{" + g_ + "}", r"\textbf{" + mc_ + "}"
        return f"{g_} & {mc_}"
    groups = [("Learned prior", ["generate only"]),
              ("Fixed simulator setting", ["coarse only", "always M1", "always M2", "always M3", "always reference"]),
              ("Gated", ["seed-spread deferral", "momentum-residual gate", "certify or simulate"]),
              (None, ["ours"])]
    lab = dict(ROWS)
    tex = [r"% generated by code/p9_table.py from the Phase 9 rows (contact cells marked \dagger: Phase 7d); do not edit by hand",
           r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{In distribution} & \multicolumn{2}{c}{Shift} & \multicolumn{2}{c}{Two-object contact} & \multicolumn{2}{c}{All three} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
           r"Pipeline & risk & cost & risk & cost & risk & cost & max gap & max cost \\", r"\midrule"]
    for g, ks in groups:
        if g:
            tex.append(r"\multicolumn{9}{@{}l}{\textit{" + g + r"}} \\")
        else:
            tex.append(r"\midrule")
        for k in ks:
            name = r"\textbf{" + lab[k] + "}" if k == "ours" else lab[k]
            tex.append(name + " & " + " & ".join(cell(b, k) for b in T) + " & " + scell(k) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab").mkdir(exist_ok=True)
    (PAPER / "tab" / "main_table.tex").write_text("\n".join(tex) + "\n")

    # figure: risk vs cost per benchmark; safe zone shaded; ours in the accent colour, baselines grey
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    short = {"generate only": "PhysCtrl", "coarse only": "coarse", "always M1": "M1", "always M2": "M2", "always M3": "M3",
             "seed-spread deferral": "seed spread", "momentum-residual gate": "residual gate",
             "certify or simulate": "certify-or-sim.", "always reference": "reference", "ours": "ours"}
    # label offsets in points (dx, dy, ha), placed by hand against overlaps; default right of the dot
    OFF = {("in distribution", "always M1"): (-4, -1, "right"), ("in distribution", "always M2"): (0, -11, "center"),
           ("in distribution", "always M3"): (-3, 4, "right"), ("in distribution", "certify or simulate"): (3, 4, "left"),
           ("in distribution", "ours"): (5, 4, "left"), ("in distribution", "always reference"): (-3, 4, "right"),
           ("in distribution", "seed-spread deferral"): (2, 5, "left"), ("in distribution", "coarse only"): (-4, -2, "right"),
           ("in distribution", "momentum-residual gate"): (5, -1, "left"),
           ("shift", "always M1"): (-4, 3, "right"), ("shift", "certify or simulate"): (-4, 3, "right"),
           ("shift", "ours"): (-5, 4, "right"), ("shift", "always reference"): (5, 2, "left"),
           ("shift", "momentum-residual gate"): (4, 4, "left"), ("shift", "coarse only"): (0, 5, "center"),
           ("shift", "generate only"): (0, 5, "center"), ("shift", "seed-spread deferral"): (0, 5, "center"),
           ("contact", "always M1"): (-4, 3, "right"), ("contact", "always M3"): (0, 5, "center"),
           ("contact", "ours"): (-5, 4, "right"), ("contact", "always reference"): (-3, 4, "right"),
           ("contact", "certify or simulate"): (2, 5, "left"), ("contact", "generate only"): (4, -3, "left")}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    ticks = [0, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7]
    for ax, b in zip(axs, T):
        ax.set_yscale("function", functions=(lambda y: np.sqrt(np.clip(y, 0, None)), np.square))
        ax.axhspan(0, SAFE, color="#2a78d6", alpha=0.08, lw=0)
        for k, v in T[b].items():
            if k not in short:
                continue
            o = k == "ours"
            ax.scatter(v["cost"], v["risk"], s=40 if o else 16, c="#2a78d6" if o else "#8a8984", zorder=3,
                       edgecolors="white", linewidths=0.8, clip_on=False)
            dx, dy, ha = OFF.get((b, k), (4, -2, "left"))
            lab = short[k] + (" (under ours)" if (b, k) == ("in distribution", "always M2") else "")
            ax.annotate(lab, (v["cost"], v["risk"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                        ha=ha, color="#0b0b0b" if o else "#52514e", fontweight="bold" if o else "normal")
        ax.set_title(b if b != "contact" else "two-object contact", fontsize=8)
        ax.set_xlim(0, 1.3); ax.set_ylim(0, 0.75)
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel("cost (x reference simulation)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
    axs[0].set_ylabel("risk  P(error > 0.10)")
    axs[2].text(1.29, SAFE * 1.15, f"risk <= {SAFE}", ha="right", va="bottom", fontsize=6, color="#2a78d6")
    fig.tight_layout()
    (PAPER / "fig").mkdir(exist_ok=True)
    fig.savefig(PAPER / "fig" / "tradeoff.pdf"); fig.savefig(PAPER / "fig" / "tradeoff.png", dpi=220)

    md = ["# Phase 9 main table (descriptive; object-weighted; fresh Phase 9 data)\n",
          f"Risk = P(error > {eps}); cost relative to one exact rollout ({c_ref:.3f} A100 s). Bold in the LaTeX table: the cheapest "
          f"pipeline with risk <= {SAFE} per benchmark: " + ", ".join(f"{b}: {best[b]}" for b in T) + ".\n"]
    for b in T:
        md += [f"## {b}", "", "| pipeline | risk (bad / n) | mean error | cost |", "|---|---|---|---|"]
        md += [f"| {k} | {v['risk']:.3f} ({v['n_bad']}/{v['n']}, {v['n_obj']} objects) | {v['err']:.4f} | {v['cost']:.2f} |"
               for k, v in T[b].items()] + [""]
    md += ["## Certificate on the generations (descriptive)", "",
           "| benchmark | bad accepted | good accepted |", "|---|---|---|"]
    md += [f"| {b} | {v['accepted_bad']}/{v['bad']} | {v['accepted_good']}/{v['good']} |" for b, v in FAR.items()] + [""]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    run.path("table.json").write_text(json.dumps({"table": T, "best_safe": best, "safe": SAFE, "far": FAR,
                                                    "max_gap": {k: summ[k][0] for k in summ}}, indent=1))
    print(f"[p9_table] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
