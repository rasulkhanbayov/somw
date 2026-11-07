"""Phase 10a: is the reference simulator converged at our error scale? CPU, after job 96.

Pre-registered Oct 20, 2026 (PLAN.md "Phase 10"), before any Phase 10 code ran or data existed.
d = distance (mean over frames and points, the paper's error) between the stored reference rollout (dt 1e-4, grid 125)
and the same setting at dt 5e-5 ("half", every Phase 9 query) or 2.5e-5 ("quarter", a fixed manifest-order subset).
  R1 per set (in distribution, each shift family, contact scenes): converged if the 95th percentile of d_half <= 0.02.
  R2 robustness: the per-point distance obeys the triangle inequality, so against the finer reference each pipeline's
     error lies in [e - d_half, e + d_half]; risk bounds follow. Claims hold if (a) ours' upper bound <= 0.03 in
     distribution, under shift (pooled) and on contact, and (b) under shift every tuned setting's lower bound > 0.03.
     Built-in self-check: with d = 0 the bounds must equal results/p9_table/table.json (run --self-check).
  R3 analytic (ood3_drop, ground truth only): before the lowest point is within two grid cells of the floor (amended, see PLAN), the centre
     of mass follows y0 + v0 t - g t^2 / 2; pass if the largest deviation <= 0.01 (descriptive).
  Also recorded (descriptive, first computed Oct 20 before any Phase 10 data): the frame-time offset of the tuned
     settings (int(0.0417 / dt) dt < 0.0417 s), i.e. the reference sampled at their frame times, as an error.

Output: results/p10a_convergence/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4b_certificate_objects import object_rate  # noqa: E402
from p4n_summarize import build, pipes  # noqa: E402
from p9_confirm import assemble, ladder  # noqa: E402
from verifier import GRAVITY, resolve_fidelity  # noqa: E402

FD = 0.0417
DX = 10.0 / 125


def dist(run, fid):
    """qid -> mean distance to the reference (inf if unstable or crashed); {} if the run does not exist."""
    out = {}
    for f in sorted(results_dir(run).glob("shard_*/fullsim.jsonl")):
        for r in read_jsonl(f):
            if r.get("fidelity") != fid:
                continue
            ok = r.get("stable") and r.get("err_t") is not None and not r.get("crashed")
            out[r["qid"]] = float(np.mean(r["err_t"])) if ok else np.inf
    return out


def freefall(path):
    with h5py.File(path, "r") as h:
        x, v, vol, floor = (np.asarray(h[k], np.float64) for k in ("x", "v", "vol", "floor_height"))
    w = vol / vol.sum()
    y = x[:, :, 1] @ w
    t = np.arange(len(y)) * FD
    pred = y[0] + float(v[0, :, 1] @ w) * t - 0.5 * GRAVITY * t ** 2
    # MPM enforces the floor on grid nodes, so bodies come to rest about one cell above it (0.084 seen): treat the last
    # two cells as the contact layer (amended Oct 20 after a one-cell threshold counted resting frames as free fall)
    free = x[:, :, 1].min(axis=1) - float(floor) > 2 * DX
    free &= np.cumprod(free).astype(bool)                 # frames before the first approach to the floor
    return float(np.abs(y - pred)[free].max()) if free.sum() > 1 else float("nan"), int(free.sum())


def timeshift(path, spec):
    with h5py.File(path, "r") as h:
        x = np.asarray(h["x"], np.float64)
        E, nu = float(np.asarray(h["E"])), float(np.asarray(h["nu"]))
    dt = resolve_fidelity(spec, {"E": E, "nu": nu})["substep_dt"]
    fd = int(FD / dt) * dt
    e = []
    for t in range(24):
        sf = 2 * t + 1
        tt = sf * fd
        j = int(np.floor(tt / FD))
        a = (tt - j * FD) / FD
        e.append(np.linalg.norm((1 - a) * x[j] + a * x[j + 1] - x[sf], axis=1).mean())
    return float(np.mean(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--self-check", action="store_true", help="d = 0 everywhere: bounds must equal the Phase 9 table")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg, ccfg = load_config(cfg["single_config"]), load_config(cfg["contact_config"])
    eps, safe = float(scfg["eps"]), float(cfg["safe"])

    P, A, c_ref, _, _ = assemble(scfg)
    P["certify or simulate"] = P.pop("certify or simulate (one window)")
    P["ours"] = P.pop("ours-v2")
    Bd = build(ccfg)
    con = [r for r in Bd["R"] if r["kind"] == "contact"]
    Pc = pipes(con, Bd["cg"], Bd["c_ref"])
    lc = ladder(scfg["contact_full_run"], [r["qid"] for r in con], Bd["c_ref"], float(scfg["full_substeps"]), ("M1", "M2", "M3"))
    Pc = {"generate only": Pc["generate only"], "certify or simulate": Pc["cert_ref"], "always reference": Pc["always reference"],
          "ours": Pc["sf"], **{f"always {f}": lc[f] for f in ("M1", "M2", "M3")}}

    qs, splits, objs = list(A["qid"]), np.array(A["split"]), np.asarray(A["obj"])
    qc, objc = [r["qid"] for r in con], np.array([r["obj"] for r in con])
    if args.self_check:
        dh_s, dh_c = np.zeros(len(qs)), np.zeros(len(qc))
    else:
        h1, h2 = dist(cfg["half_runs"]["single"], "half"), dist(cfg["half_runs"]["contact"], "half")
        missing = [q for q in qs if q not in h1] + [q for q in qc if q not in h2]
        if missing:
            raise RuntimeError(f"{len(missing)} queries lack a half-step rollout, e.g. {missing[:3]}")
        dh_s, dh_c = np.array([h1[q] for q in qs]), np.array([h2[q] for q in qc])
    sets = {"in distribution": (~np.char.startswith(splits.astype(str), "ood3_"), "single")}
    for fam in sorted({s for s in splits if s.startswith("ood3_")}):
        sets[fam] = (splits == fam, "single")
    sets["shift (pooled)"] = (np.char.startswith(splits.astype(str), "ood3_"), "single")
    sets["contact"] = (np.ones(len(qc), bool), "contact")

    def bounds(PP, d, sel, ob):
        res = {}
        for k, (e, _) in PP.items():
            e, dd, o = e[sel], d[sel], ob[sel]
            res[k] = [object_rate((e - dd > eps).astype(float), o), object_rate((e > eps).astype(float), o),
                      object_rate((e + dd > eps).astype(float), o)]
        return res
    R2 = {nm: bounds(P, dh_s, sel, objs) if kind == "single" else bounds(Pc, dh_c, sel, objc) for nm, (sel, kind) in sets.items()}

    # the built-in check: at d = 0 the bounds collapse onto the point risks of the Phase 9 table
    tab = load_json(results_dir("p9_table") / "table.json")["table"]
    for nm, tk in (("in distribution", "in distribution"), ("shift (pooled)", "shift"), ("contact", "contact")):
        for k, (lo, pt, hi) in R2[nm].items():
            if k in tab[tk] and abs(pt - tab[tk][k]["risk"]) > 1e-9:
                raise RuntimeError(f"{nm}/{k}: point risk {pt} differs from the Phase 9 table {tab[tk][k]['risk']}")
            if args.self_check and not (lo == pt == hi):
                raise RuntimeError(f"self-check: {nm}/{k} bounds do not collapse at d = 0")
    if args.self_check:
        print("[p10a] self-check passed: point risks equal the Phase 9 table and the bounds collapse at d = 0")
        return

    run = start_run(cfg.get("summary_run_name", "p10a_convergence"), cfg, synthetic=False)
    q1, q2 = dist(cfg["quarter_runs"]["single"], "quarter"), dist(cfg["quarter_runs"]["contact"], "quarter")
    stat = lambda v: (float(np.median(v)), float(np.quantile(v, .95)), float(np.max(v))) if len(v) else (np.nan,) * 3  # noqa: E731
    R1, md = {}, ["# Phase 10a: is the reference converged? (pre-registered Oct 20, 2026)\n",
                  f"d = distance to the stored reference (dt 1e-4, grid 125). Rule R1: 95th percentile of d_half <= {cfg['r1_p95']}.\n",
                  "| set | n | d_half median / p95 / max | d_quarter (subset) n, median / p95 / max | R1 |", "|---|---|---|---|---|"]
    for nm, (sel, kind) in sets.items():
        dh = dh_s[sel] if kind == "single" else dh_c[sel]
        ids = [q for q, s in zip(qs if kind == "single" else qc, sel) if s]
        dq = np.array([(q1 if kind == "single" else q2)[q] for q in ids if q in (q1 if kind == "single" else q2)])
        a, b, c = stat(dh); a2, b2, c2 = stat(dq)
        R1[nm] = bool(b <= float(cfg["r1_p95"]))
        md.append(f"| {nm} | {len(dh)} | {a:.4f} / {b:.4f} / {c:.4f} | {len(dq)}, {a2:.4f} / {b2:.4f} / {c2:.4f} | "
                  f"**{'converged' if R1[nm] else 'NOT converged'}** |")
    ok_a = all(R2[nm]["ours"][2] <= safe for nm in ("in distribution", "shift (pooled)", "contact"))
    ok_b = all(R2["shift (pooled)"][f"always {f}"][0] > safe for f in ("M1", "M2", "M3"))
    md += ["", f"## R2: risk against a finer reference, bounds [lower, upper] (point = Phase 9)", "",
           "| pipeline | " + " | ".join(R2) + " |", "|---|" + "---|" * len(R2)]
    for k in P:
        md.append(f"| {k} | " + " | ".join(f"{R2[nm][k][0]:.3f} / {R2[nm][k][1]:.3f} / {R2[nm][k][2]:.3f}" if k in R2[nm] else "—"
                                        for nm in R2) + " |")
    md += ["", f"- (a) ours' upper bound <= {safe} in distribution, under shift and on contact: **{'PASS' if ok_a else 'FAIL'}**",
           f"- (b) under shift every tuned setting's lower bound > {safe}: **{'PASS' if ok_b else 'FAIL'}**",
           f"- **Claims robust to the reference's time-step error: {'YES' if (ok_a and ok_b) else 'NO'}**", ""]
    oi = ood_index_from(scfg["build_runs"])
    ff = [freefall(oi[q]) for q, s in zip(qs, splits) if s == "ood3_drop"]
    dev = np.array([f[0] for f in ff if np.isfinite(f[0])])
    R3 = bool(len(dev) and dev.max() <= float(cfg["freefall_tol"]))
    md += ["## R3: free fall before the floor (ood3_drop ground truth)", "",
           f"{len(dev)} queries with >= 2 free frames; centre-of-mass deviation from y0 + v0 t - g t^2/2: median {np.median(dev):.2e}, "
           f"max {dev.max():.2e}: **{'PASS' if R3 else 'FAIL'}** (tolerance {cfg['freefall_tol']}).", ""]
    full = load_config("configs/p9_full.yaml")["fidelities"]
    md += ["## Recorded: frame-time offset of the tuned settings (descriptive)", "",
           "| setting | split | offset error median / p95 / max |", "|---|---|---|"]
    TS = {}
    for f in ("M1", "M2", "M3"):
        ts = np.array([timeshift(oi[q], full[f]) for q in qs])
        TS[f] = ts.tolist()
        for s in sorted(set(splits)):
            v = ts[splits == s]
            md.append(f"| {f} | {s} | {np.median(v):.4f} / {np.quantile(v, .95):.4f} / {v.max():.4f} |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"R1": R1, "R2": R2, "R2a": ok_a, "R2b": ok_b, "R3": R3, "freefall_max": float(dev.max()),
               "timeshift": TS}, run.path("summary.json"))
    print(f"[p10a] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
