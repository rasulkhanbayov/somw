"""Phase 5a: audited amortisation of the certificate over a query stream. CPU only (dev server).

Pre-registered Oct 16, 2026 (PLAN.md "Phase 5a"), before this code ran. A cheap learned predictor of
the certificate score is trained on past probes; when it is confident a query would be accepted, the
probes are skipped. A random audit fraction a of queries is always probed; the audits (held out from
training) calibrate the skip rule by split conformal, so P(certificate would reject | skipped) <~ delta.
  skip iff  g(x) + q_hat < log10(tau),   g predicts log10(score),   q_hat = (1 - delta) audit quantile
Tests over R random stream orders: A1 probe rate falls (first third - last third > 0, 2.5th pct > 0);
A2 mean risk(amortised) - risk(cert_ref) <= 0.02; A3 cost saving > 0 (2.5th pct > 0).

Output: results/p5a_amortise/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p4a_certificate import ood_index_from  # noqa: E402
from p4d_pipeline import gen_costs  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402
from regimes import _base_drag_coeff  # noqa: E402

FEATURES = ["log10_E", "nu", "force_coeff", "mask_frac", "gravity", "floor_gap", "centroid_disp", "max_disp",
            "mean_speed", "max_speed", "max_centroid_acc", "bbox_ratio"]


def features(q: dict, g: np.ndarray) -> list[float]:
    """Fixed feature list (PLAN.md): physics inputs + statistics of generated sample 0 (24, N, 3)."""
    x0 = np.asarray(q["x0"], np.float64)
    vol = np.asarray(q["vol"], np.float64)
    df = np.asarray(q["drag_force"]).reshape(-1, 3)
    dm = np.asarray(q["drag_mask"]).reshape(len(df), -1) if df.size else np.zeros((0, len(x0)))
    if df.size and dm[0].any():
        coeff = float(np.linalg.norm(df[0]) / _base_drag_coeff(vol, dm[0].astype(bool)))
        frac = float(dm[0].mean())
    else:
        coeff, frac = 0.0, 0.0
    traj = np.concatenate([x0[None], g.astype(np.float64)], 0)               # frame 0 + 24 model frames
    c = traj.mean(axis=1)
    v = np.linalg.norm(np.diff(traj, axis=0), axis=2)
    acc = np.linalg.norm(np.diff(c, 2, axis=0), axis=1)
    ext = lambda p: float(np.prod(p.max(0) - p.min(0) + 1e-6))                 # noqa: E731
    return [float(np.log10(q["E"])), float(q["nu"]), coeff, frac, float(q.get("gravity", 0)),
            float(x0[:, 1].min() - q["floor_height"]) if int(q.get("gravity", 0)) else 0.0,
            float(np.linalg.norm(c[-1] - c[0])), float(np.linalg.norm(traj[-1] - traj[0], axis=1).max()),
            float(v.mean()), float(v.max()), float(acc.max()) if len(acc) else 0.0, ext(traj[-1]) / ext(traj[0])]


def light_query(path) -> dict:
    """Only what the features and window placement need: frame 0 and the scalar inputs. Reading the full
    HDF5 (x, v, F, C at 49 frames, ~10 MB each) over NFS made the first run I/O bound (Oct 16)."""
    import h5py
    with h5py.File(path, "r") as h:
        g = lambda k, d=None: np.asarray(h[k]) if k in h else d                    # noqa: E731
        return {"x0": np.asarray(h["x"][0], np.float32), "vol": g("vol"), "drag_force": g("drag_force", np.zeros(0)),
                "drag_mask": g("drag_mask", np.zeros(0)), "drag_point": g("drag_point", np.zeros(0)),
                "E": float(g("E")), "nu": float(g("nu")), "floor_height": float(g("floor_height")),
                "gravity": int(g("gravity", 0))}


def load_pool(cfg, a_fit, b_fit):
    spec = cfg["defect_spec"]
    root, groot = Path(cfg["paths"]["data_root"]), gen_dir(cfg, False)
    rows = []
    for pool in cfg["pools"]:
        oi = ood_index_from(pool["build_runs"])
        errs = {r["qid"]: r for f in sorted(results_dir(pool["probe_run"]).glob("shard_*/probes.jsonl"))
                for r in read_jsonl(f) if r["model"] == "base"}
        for f in sorted(results_dir(pool["scan_run"]).glob("shard_*/scan.jsonl")):
            for r in read_jsonl(f):
                if r["source"] != "base" or r["fidelity"] != spec["fidelity"] or int(r["m"]) != spec["m"]:
                    continue
                if r["qid"] not in errs:
                    continue
                path = root / "testset" / f"{r['qid']}.h5" if r["split"] == "test_id" else Path(oi[r["qid"]])
                q = {**light_query(path), "qid": r["qid"], "split": r["split"]}
                s = np.load(groot / "base" / r["split"] / f"{r['qid']}.npz")["samples"]
                w = {st: (np.inf if (bl or cr or v is None) else v) for st, v, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
                sub = dict(zip(r["starts"], r["substeps"]))
                st = starts_for(spec["placement"], s, q, spec["m"])
                rows.append({"qid": r["qid"], "split": r["split"], "pool": pool["name"],
                             "shift": not (r["split"] == "test_id" or r["split"].startswith("id_new_")),
                             "x": features(q, s[0]), "score": max(w[t] for t in st),
                             "probe": sum(a_fit + b_fit * (sub.get(t) or 0) for t in st),
                             "E": float(errs[r["qid"]]["err_mean"])})
    return rows


def run_stream(order, X, y, S, E, P, cfg, c_ref, cg, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    tau, a, delta = float(cfg["tau"]), float(cfg["audit_rate"]), float(cfg["delta"])
    rng = np.random.default_rng(10_000 + seed)
    audit = rng.random(len(order)) < a
    train, cal = [], []
    model, qhat = None, np.inf
    out = {"probed": np.zeros(len(order), bool), "err": np.zeros(len(order)), "cost": np.zeros(len(order)),
           "skipped": np.zeros(len(order), bool)}
    for t, i in enumerate(order):
        if t % int(cfg["retrain_every"]) == 0 and len(train) >= 20:
            model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[train], y[train])
            if len(cal) >= int(cfg["min_audits"]):
                res = np.sort(y[cal] - model.predict(X[cal]))
                k = int(np.ceil((len(res) + 1) * (1 - delta)))
                qhat = res[k - 1] if k <= len(res) else np.inf
        skip = (not audit[t]) and model is not None and len(cal) >= int(cfg["min_audits"]) and \
            np.isfinite(qhat) and model.predict(X[i:i + 1])[0] + qhat < np.log10(tau)
        if skip:
            out["skipped"][t] = True
            out["err"][t], out["cost"][t] = E[i], cg
            continue
        out["probed"][t] = True
        (cal if audit[t] else train).append(i)
        acc = S[i] < tau
        out["err"][t] = E[i] if acc else 0.0
        out["cost"][t] = cg + P[i] + (0.0 if acc else c_ref)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("run_name", "p5a_amortise"), cfg, synthetic=False)
    spec, eps, tau = cfg["defect_spec"], cfg["eps"], float(cfg["tau"])
    gc, c_ref = gen_costs()
    cg = gc["base"]["best"]
    a_fit, b_fit, _ = fit_probe_cost(cfg["cost_fit_scan_run"], spec["fidelity"], spec["m"])
    rows = load_pool(cfg, a_fit, b_fit)
    n = len(rows)
    X = np.array([r["x"] for r in rows]); S = np.array([r["score"] for r in rows])
    y = np.log10(np.clip(S, 1e-4, 1.0)); E = np.array([r["E"] for r in rows]); P = np.array([r["probe"] for r in rows])
    shift = np.array([r["shift"] for r in rows]); split = np.array([r["split"] for r in rows])
    acc_all = S < tau
    cert_err = np.where(acc_all, E, 0.0); cert_cost = cg + P + (~acc_all) * c_ref
    thirds = np.array_split(np.arange(n), 3)
    res = {k: [] for k in ("rate_first", "rate_last", "d_rate", "risk_am", "risk_cert", "d_risk", "cost_am", "cost_cert",
                           "saving", "skip_rate", "risk_skipped")}
    curves = []
    for seed in range(int(cfg["n_orders"])):
        order = np.random.default_rng(seed).permutation(n)
        o = run_stream(order, X, y, S, E, P, cfg, c_ref, cg, seed)
        rf, rl = o["probed"][thirds[0]].mean(), o["probed"][thirds[-1]].mean()
        ra, rc = float(np.mean(o["err"] > eps)), float(np.mean(cert_err > eps))
        ca, cc = float(o["cost"].mean()), float(cert_cost.mean())
        res["rate_first"].append(rf); res["rate_last"].append(rl); res["d_rate"].append(rf - rl)
        res["risk_am"].append(ra); res["risk_cert"].append(rc); res["d_risk"].append(ra - rc)
        res["cost_am"].append(ca); res["cost_cert"].append(cc); res["saving"].append(cc - ca)
        res["skip_rate"].append(o["skipped"].mean())
        sk = o["skipped"]
        res["risk_skipped"].append(float(np.mean(o["err"][sk] > eps)) if sk.any() else np.nan)
        curves.append([o["probed"][b].mean() for b in np.array_split(np.arange(n), 10)])
    R = {k: np.array(v, float) for k, v in res.items()}
    q = lambda v: f"{np.nanmean(v):.3f} [{np.nanquantile(v, .025):.3f}, {np.nanquantile(v, .975):.3f}]"   # noqa: E731
    A1 = R["d_rate"].mean() > 0 and np.quantile(R["d_rate"], .025) > 0
    A2 = R["d_risk"].mean() <= 0.02
    A3 = R["saving"].mean() > 0 and np.quantile(R["saving"], .025) > 0
    # secondary: drift order (in distribution first, then shifted), seed 0
    drift = np.concatenate([np.random.default_rng(0).permutation(np.nonzero(~shift)[0]),
                            np.random.default_rng(1).permutation(np.nonzero(shift)[0])])
    od = run_stream(drift, X, y, S, E, P, cfg, c_ref, cg, 0)
    d_id, d_sh = ~shift[drift], shift[drift]
    md = ["# Phase 5a: audited amortisation of the certificate (pre-registered Oct 16, 2026)\n",
          f"Stream: {n} single-object queries ({int((~shift).sum())} in distribution, {int(shift.sum())} shifted; pools "
          f"{', '.join(p['name'] for p in cfg['pools'])}). tau = {tau}; audit rate {cfg['audit_rate']}; delta {cfg['delta']}; "
          f"{cfg['n_orders']} random orders. Values: mean [2.5%, 97.5%] over orders. Costs in A100 s (reference {c_ref:.3f}).\n",
          f"- **A1 amortisation:** probe rate first third {q(R['rate_first'])} -> last third {q(R['rate_last'])}; "
          f"difference {q(R['d_rate'])}: **{'PASS' if A1 else 'FAIL'}**",
          f"- **A2 risk:** amortised {q(R['risk_am'])} vs always-probe {q(R['risk_cert'])}; difference {q(R['d_risk'])} "
          f"(mean <= 0.02): **{'PASS' if A2 else 'FAIL'}**",
          f"- **A3 cost:** saving {q(R['saving'])} s/query (always-probe {q(R['cost_cert'])}, amortised {q(R['cost_am'])}): "
          f"**{'PASS' if A3 else 'FAIL'}**",
          f"- **Confirmation: {'CONFIRMED' if (A1 and A2 and A3) else 'NOT CONFIRMED'}**", "",
          f"Skip rate {q(R['skip_rate'])}; risk among skipped queries {q(R['risk_skipped'])}.", "",
          "Probe rate by stream decile (mean over orders): " + ", ".join(f"{v:.2f}" for v in np.mean(curves, 0)), "",
          f"**Drift order (secondary):** in-distribution part skip rate {od['skipped'][d_id].mean():.3f}, "
          f"risk {np.mean(od['err'][d_id] > eps):.3f}; shifted part skip rate {od['skipped'][d_sh].mean():.3f}, "
          f"risk {np.mean(od['err'][d_sh] > eps):.3f} (always-probe risk there {np.mean(cert_err[drift][d_sh] > eps):.3f})."]
    o0 = run_stream(np.random.default_rng(0).permutation(n), X, y, S, E, P, cfg, c_ref, cg, 0)
    sk_row = np.zeros(n, bool)
    sk_row[np.random.default_rng(0).permutation(n)] = o0["skipped"]           # back to row order
    md += ["", "| split (seed-0 order) | n | skip rate | generation risk | certificate rejects |", "|---|---|---|---|---|"]
    for sp in sorted(set(split)):
        m = split == sp
        md.append(f"| {sp} | {m.sum()} | {sk_row[m].mean():.2f} | {np.mean(E[m] > eps):.2f} | {np.mean(~acc_all[m]):.2f} |")
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"A1": bool(A1), "A2": bool(A2), "A3": bool(A3), **{k: [float(np.nanmean(v)), float(np.nanquantile(v, .025)),
               float(np.nanquantile(v, .975))] for k, v in R.items()}, "curve": np.mean(curves, 0).tolist(), "n": n},
              run.path("summary.json"))
    print(f"[p5a] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
