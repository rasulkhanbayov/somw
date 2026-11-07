"""Phase 1 summary and GO/NO-GO verdict.

Reads results/p1_cost_{A100,H200}/timings.jsonl and
results/p1_fidelity_A100/shard_*/fidelity.jsonl, writes
results/p1_summary/SUMMARY.md and summary.json.

Decision rule (from configs/p1_summary.yaml):
  For each hardware and each realistic regime, ratio = simulator cost /
  per-sample generation cost (best batch size). For the physctrl regime the
  simulator cost is the matched-accuracy one: the cheapest stable sweep setting
  whose error is no worse than the generator's (falls back to the default
  setting, flagged, if the sweep has no such setting).
  GO if any realistic regime reaches go_ratio on both A100 and H200.
Multi-object generation cost is n_objects x single-object generation, a lower
bound (PhysCtrl generates one object at a time). Dense regimes reuse the
2,048-point generation cost; the upsampling step a real pipeline would need is
not counted, which favours generation.
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.allow_synthetic)
    run = start_run("p1_summary", cfg, synthetic=args.allow_synthetic)
    md, summary = [], {"verdict": None}

    # ---- cost -----------------------------------------------------------------
    gen_best, gen_b1, sim_best, sim_asis, probes = {}, {}, {}, {}, {}
    for hw in cfg["hardware"]:
        try:
            rows = read_jsonl(results_dir(f"p1_cost_{hw}", args.allow_synthetic) / "timings.jsonl")
        except FileNotFoundError:
            md.append(f"> p1_cost_{hw} missing; that hardware is excluded.\n")
            continue
        by_model = defaultdict(list)
        for r in rows:
            if r["kind"] == "generation":
                by_model[r["model"]].append(r)
        for m, rs in by_model.items():
            gen_best[(hw, m)] = min(r["per_sample_s_median"] for r in rs)
            b1 = [r["per_sample_s_median"] for r in rs if r["batch_size"] == 1]
            gen_b1[(hw, m)] = med(b1)
        by_reg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["kind"] == "simulation" and r.get("stable") and "rollout_s" in r:
                by_reg[r["regime"]][r["mode"]].append(r["rollout_s"])
        for reg, modes in by_reg.items():
            meds = {mode: med(v) for mode, v in modes.items()}
            best_mode = min(meds, key=meds.get)
            sim_best[(hw, reg)] = (meds[best_mode], best_mode)
            sim_asis[(hw, reg)] = meds.get("as_is", float("nan"))
        for r in rows:
            if r["kind"] == "probe":
                probes[(hw, r["m_model_frames"])] = r["median_s"]

    md.append("## Generation cost (seconds per trajectory sample)\n")
    md.append("| hardware | model | best batch | batch 1 |\n|---|---|---|---|")
    for (hw, m), v in sorted(gen_best.items()):
        md.append(f"| {hw} | {m} | {v:.4f} | {gen_b1[(hw, m)]:.4f} |")

    md.append("\n## Simulation cost (seconds per 48-frame rollout, median over queries)\n")
    md.append("| hardware | regime | best mode | best (s) | as_is (s) |\n|---|---|---|---|---|")
    for (hw, reg), (v, mode) in sorted(sim_best.items()):
        md.append(f"| {hw} | {reg} | {mode} | {v:.3f} | {sim_asis[(hw, reg)]:.3f} |")

    # ---- fidelity / matched accuracy --------------------------------------------
    matched = {}
    try:
        fid_rows = []
        for f in glob.glob(str(results_dir("p1_fidelity_A100", args.allow_synthetic) / "shard_*" / "fidelity.jsonl")):
            fid_rows.extend(read_jsonl(f))
    except FileNotFoundError:
        fid_rows = []
    if fid_rows:
        sims = defaultdict(list)
        gens = defaultdict(list)
        for r in fid_rows:
            if r["kind"] == "sim":
                key = (r.get("substep_dt"), r.get("n_grid"))
                sims[key].append(r)
            elif r["kind"] == "gen":
                gens[r["model"]].append(r["err_mean"])
        md.append("\n## Simulator fidelity sweep (A100)\n")
        n_crash = sum(1 for r in fid_rows if r.get("crashed"))
        md.append(f"Rows where the rollout blew up and lost the CUDA context: {n_crash} "
                  "(counted as unstable).\n")
        md.append("| substep_dt | n_grid | stable | median err | median rollout (s) |\n|---|---|---|---|---|")
        table = []
        for key, rs in sorted(sims.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0)):
            stable = np.mean([bool(r.get("stable")) for r in rs])
            err = med([r.get("err_mean") for r in rs if r.get("stable")])
            sec = med([r.get("rollout_s") for r in rs if r.get("stable")])
            table.append((key, stable, err, sec))
            md.append(f"| {key[0]} | {key[1]} | {stable:.2f} | {err:.4f} | {sec:.3f} |")
        # reference = finest setting in the sweep (PhysCtrl's own: smallest dt, largest grid)
        ref = min(table, key=lambda t: (t[0][0] or 0, -(t[0][1] or 0)))
        for m, errs in gens.items():
            g_err = med(errs)
            ok = [t for t in table if t[1] == 1.0 and np.isfinite(t[2]) and t[2] <= g_err]
            if ok:
                best = min(ok, key=lambda t: t[3])
                matched[m] = {"config": best[0], "rollout_s": best[3], "sim_err": best[2],
                              "gen_err": g_err, "fallback": False}
            else:
                stable_rows = [t for t in table if t[1] == 1.0]
                cheapest = min(stable_rows, key=lambda t: t[3]) if stable_rows else None
                matched[m] = {"config": cheapest[0] if cheapest else None,
                              "rollout_s": cheapest[3] if cheapest else float("nan"),
                              "sim_err": cheapest[2] if cheapest else float("nan"),
                              "gen_err": g_err, "fallback": True}
            # speed-up of the matched setting over the reference, measured within the sweep
            # (same mode, same queries); applied to each hardware's best physctrl rollout.
            matched[m]["speedup_factor"] = matched[m]["rollout_s"] / ref[3] if ref[3] > 0 else float("nan")
            md.append(f"\nGenerator **{m}** median error {g_err:.4f}. Matched-accuracy simulator: "
                      f"{matched[m]['config']} at {matched[m]['rollout_s']:.3f} s"
                      + (" (FALLBACK: no sweep setting was as inaccurate as the generator; "
                         "this is the cheapest stable one, so the ratio is an upper bound on "
                         "matched-accuracy cost)" if matched[m]["fallback"] else "") + ".")

    # ---- ratios and verdict -------------------------------------------------------
    md.append("\n## Cost ratio: simulation / generation (per sample, best batch)\n")
    md.append("| hardware | regime | model | sim (s) | gen (s) | ratio | note |\n|---|---|---|---|---|---|---|")
    ratios = defaultdict(dict)
    for (hw, reg), (s_best, _) in sorted(sim_best.items()):
        for (hw2, m), g in gen_best.items():
            if hw2 != hw:
                continue
            note = ""
            s = s_best
            if reg == "physctrl" and m in matched and np.isfinite(matched[m]["speedup_factor"]):
                s = s_best * matched[m]["speedup_factor"]
                note = (f"matched-accuracy sim (x{matched[m]['speedup_factor']:.2f} of reference)"
                        + (" (fallback)" if matched[m]["fallback"] else ""))
            n_obj = int(reg[-1]) if reg.startswith("multiobj") else 1
            g_eff = g * n_obj
            if n_obj > 1:
                note = f"gen = {n_obj} x single object (lower bound)"
            if reg.startswith("dense"):
                note = "gen at 2,048 pts; upsampling not counted"
            r = s / g_eff if g_eff > 0 else float("nan")
            ratios[(hw, m)][reg] = r
            md.append(f"| {hw} | {reg} | {m} | {s:.3f} | {g_eff:.4f} | {r:.1f} | {note} |")

    if probes:
        md.append("\n## Probe cost (one window, best mode)\n")
        md.append("| hardware | window (model frames) | seconds | fraction of best full rollout |\n|---|---|---|---|")
        for (hw, m), v in sorted(probes.items()):
            full = sim_best.get((hw, "physctrl"), (float("nan"),))[0]
            md.append(f"| {hw} | {m} | {v:.3f} | {v / full:.3f} |")

    go_ratio = cfg["go_ratio"]
    realistic = cfg["realistic_regimes"]
    hws = sorted({hw for hw, _ in ratios})
    passing = []
    for reg in realistic:
        for m in cfg["models_for_verdict"]:
            vals = [ratios[(hw, m)].get(reg, float("nan")) for hw in hws]
            if vals and all(np.isfinite(v) and v >= go_ratio for v in vals):
                passing.append((reg, m, vals))
    verdict = "GO" if passing else "NO-GO"
    summary.update({"verdict": verdict, "go_ratio": go_ratio, "passing": passing,
                    "ratios": {f"{hw}|{m}": v for (hw, m), v in ratios.items()},
                    "matched": matched})
    md.insert(0, f"# Phase 1 summary: **{verdict}**\n\nRule: simulation >= {go_ratio}x generation "
                 f"(per sample) in some realistic regime ({', '.join(realistic)}) on every hardware "
                 f"measured ({', '.join(hws)}).\n\nPassing: "
                 + (", ".join(f"{r} with {m}" for r, m, _ in passing) or "none") + "\n")
    if args.allow_synthetic:
        md.insert(0, "> SYNTHETIC INPUT. These numbers test the pipeline only and mean nothing.\n")
    (run.path("SUMMARY.md")).write_text("\n".join(md))
    save_json(summary, run.path("summary.json"))
    print("\n".join(md[:6]))
    print(f"[p1_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok", verdict=verdict)


if __name__ == "__main__":
    main()
