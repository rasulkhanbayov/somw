"""Phase 4n verdict: event-aware fidelity (EAF) repair on fresh contact scenes. CPU (dev server).

Pre-registered Oct 16, 2026 (PLAN.md "Phase 4n"), before any 4n data. Contact scenes; cluster
bootstrap over the first object of each scene; eps = 0.10; coarse spread_event certificate at tau.
  cert_ref  reject -> reference rollout
  EAF       reject -> V2 restart at k_s = max(k_f - 1, 1), k_f = first frame the generated objects come
            within d_s (3 coarse cells); no k_f or k_f <= 1 -> reference rollout
  chain     reject -> first failing coarse chain window (4k timing)
  event     reject -> V2 restart at k_e = max(k_g - 1, 1) (secondary)
  sf        (Phase 4o, primary there) reject -> V2 restart at k_sw = max(k_sh - 1, 1), k_sh = first frame the
            coarse SHADOW's objects come within d_s (from the GPU job); tests use cfg 'primary' (eaf | sf)
  N1 risk(EAF) <= risk(cert_ref) + 0.02     N2 saving cert_ref - EAF: 95% lower bound > 0
  N3 error(chain) - error(EAF): 95% lower bound > 0
True errors are computed here from the scene ground truth and generated sample 0 (no GPU step).

Output: results/p4n_summary/{SUMMARY.md, summary.json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_json, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from backends import query_from_h5  # noqa: E402
from mo_scenes import D_C, gen_contact_frame  # noqa: E402
from p2b_generate import gen_dir  # noqa: E402
from p2c_select import starts_for  # noqa: E402
from p4d_pipeline import gen_costs  # noqa: E402
from p4f_confirm_pipeline import fit_probe_cost  # noqa: E402


def build(cfg):
    """Per-scene rows R of the 4n/4o verdict (also used by Phase 7d) and the cost constants."""
    tau, spec = float(cfg["tau"]), cfg["defect_spec"]
    gc, c_ref = gen_costs()
    cg = 2 * gc["base"]["best"]
    full = float(cfg["full_substeps"])
    a, b, _ = fit_probe_cost(cfg["cost_fit_scan_run"], spec["fidelity"], spec["m"])
    rates = [r["rollout_s"] / r["substeps"] for f in sorted(results_dir(cfg["shadow_rate_run"]).glob("shard_*/fullsim.jsonl"))
             for r in read_jsonl(f) if r.get("fidelity") == "coarse_cfl" and r.get("stable") and r.get("substeps")
             and r.get("rollout_s") and r["split"].startswith("id_new_")]
    rate = float(np.median(rates))
    groot = gen_dir(cfg, False)
    chain_k = [int(k) for k in cfg["chain_k"]]
    d_s = float(cfg["fidelity_restart_dist"])

    man = [e for f in sorted(results_dir(cfg["build_run"]).glob("shard_*/ood_manifest.json")) for e in load_json(f)]
    design = [r for f in sorted(results_dir(cfg["build_run"]).glob("shard_*/design.jsonl")) for r in read_jsonl(f)]
    scan = {r["qid"]: r for f in sorted(results_dir(cfg["scan_run"]).glob("shard_*/scan.jsonl")) for r in read_jsonl(f)
            if r["source"] == "base" and r["fidelity"] == spec["fidelity"] and int(r["m"]) == spec["m"]}
    allk = {r["qid"]: r for f in sorted(results_dir(cfg["allk_run"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}

    R = []
    n_missing = 0
    has_sf = any("k_sh" in x for x in allk.values())            # Phase 4o rows only
    for e in man:
        q = query_from_h5(e["path"], e["split"], e["qid"])
        s = np.load(groot / "base" / e["split"] / f"{e['qid']}.npz")["samples"]
        gt = q["x"][1::2][:24]
        E = float(np.linalg.norm(s[0] - gt, axis=2).mean())
        r = scan[e["qid"]]
        w = {st: (np.inf if (bl or cr or v is None) else v) for st, v, bl, cr in zip(r["starts"], r["raw"], r["blown"], r["crashed"])}
        sub = dict(zip(r["starts"], r["substeps"]))
        wc = lambda st: a + b * (sub.get(st) or 0)                                       # noqa: E731
        cert = starts_for(spec["placement"], s, q, spec["m"])
        if cfg.get("cert_windows") == "one":
            cert = cert[:1]                  # Phase 9: the one-window certificate (the event window comes first)
        x = allk[e["qid"]]

        def restart(k):
            nonlocal n_missing
            v = x.get("k", {}).get(str(k))
            if v is None:
                n_missing += 1
                return 0.0, c_ref                                   # no rollout -> reference fallback
            c = rate * v["shadow_substeps"] + c_ref * v["substeps"] / full
            if v["shadow_blown"] or v["blown"]:
                return 0.0, c + c_ref
            m = float(np.mean(v["err_t"]))
            return (m if np.isfinite(m) else np.inf), c
        kf = gen_contact_frame(s[0], q["object_id"], d_c=d_s)
        kg = gen_contact_frame(s[0], q["object_id"], d_c=D_C)
        k_s = max(kf - 1, 1) if (kf is not None and kf > 1) else None
        k_e = max(kg - 1, 1) if (kg is not None and kg > 1) else None
        W = [w[0]] + [w[2 * k + 1] for k in chain_k]
        j = next((i for i, v in enumerate(W) if not v < tau), None)
        last = len(W) - 1 if j is None else j
        chain_probe = sum(0.0 if st in cert else wc(st) for st in ([0] + [2 * k + 1 for k in chain_k])[:last + 1])
        R.append({"obj": e["qid"].split("__")[0], "qid": e["qid"], "split": e["split"], "kind": e["kind"], "E": E,
                  "acc": max(w[st] for st in cert) < tau, "probe": sum(wc(st) for st in cert),
                  "eaf": restart(k_s) if k_s is not None else (0.0, c_ref),
                  "event": restart(k_e) if k_e is not None else (0.0, c_ref),
                  # Phase 4o: the shadow-timed switch k_sw comes from the GPU job (it needs the shadow itself)
                  "sf": (restart(x["k_sw"]) if x.get("k_sw") is not None else (0.0, c_ref)) if "k_sh" in x else (np.nan, np.nan),
                  "chain": (0.0, chain_probe + c_ref) if j in (None, 0) else
                           (lambda v: (v[0], chain_probe + v[1]))(restart(chain_k[j - 1])),
                  "k_s_off": (k_s - kg) if (k_s is not None and kg is not None) else None})

    return {"R": R, "design": design, "n_missing": n_missing, "has_sf": has_sf, "cg": cg, "c_ref": c_ref,
            "rate": rate, "groot": groot}


def pipes(rows, cg, c_ref):
    """Per-scene (error, cost) of every 4n/4o pipeline."""
    acc = np.array([r["acc"] for r in rows]); E = np.array([r["E"] for r in rows]); P = np.array([r["probe"] for r in rows])
    out = {"generate only": (E, np.full(len(E), cg)), "always reference": (np.zeros(len(E)), np.full(len(E), c_ref)),
           "cert_ref": (np.where(acc, E, 0.0), cg + P + (~acc) * c_ref)}
    for k in ("eaf", "event", "chain", "sf"):
        ee = np.array([r[k][0] for r in rows]); cc = np.array([r[k][1] for r in rows])
        out[k] = (np.where(acc, E, ee), cg + P + np.where(acc, 0.0, cc))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run(cfg.get("summary_run_name", "p4n_summary"), cfg, synthetic=False)
    eps, tau, B = cfg["eps"], float(cfg["tau"]), cfg["n_boot"]
    Bd = build(cfg)
    R, design, n_missing, has_sf, cg, c_ref = Bd["R"], Bd["design"], Bd["n_missing"], Bd["has_sf"], Bd["cg"], Bd["c_ref"]
    con = [r for r in R if r["kind"] == "contact"]
    O = np.array([r["obj"] for r in con])
    P = pipes(con, cg, c_ref)

    def boot(vals):
        vals = np.asarray(vals, float)
        ob = np.array(sorted(set(O)))
        idx = {o: np.nonzero(O == o)[0] for o in ob}
        f = lambda d: np.mean([vals[idx[o]].mean() for o in d])                      # noqa: E731
        rng = np.random.default_rng(1)
        bs = [f(rng.choice(ob, len(ob), replace=True)) for _ in range(B)]
        return float(f(ob)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    risk = lambda k: boot((P[k][0] > eps).astype(float))                              # noqa: E731
    err = lambda k: boot(np.minimum(P[k][0], 10.0))                                    # noqa: E731
    cost = lambda k: boot(P[k][1])                                                     # noqa: E731
    pk = cfg.get("primary", "eaf")                       # 4n: eaf; 4o: sf (pre-registered per phase)
    tp = cfg.get("test_prefix", "N")
    r_eaf, r_cr = risk(pk), risk("cert_ref")
    N1 = r_eaf[0] <= r_cr[0] + 0.02
    if cfg.get("abs_risk_limit") is not None:              # Phase 9 amendment: also an absolute safety bound
        N1 = N1 and r_eaf[0] <= float(cfg["abs_risk_limit"])
    sv = boot(P["cert_ref"][1] - P[pk][1]); N2 = sv[1] > 0
    tm = boot(np.minimum(P["chain"][0], 10) - np.minimum(P[pk][0], 10)); N3 = tm[1] > 0
    f3 = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"                          # noqa: E731
    n_inv = sum(not d.get("valid", False) for d in design)
    offs = [r["k_s_off"] for r in con if r["k_s_off"] is not None]
    md = ["# " + cfg.get("title", "Phase 4n: event-aware fidelity repair on fresh contact scenes (pre-registered Oct 16, 2026)") + "\n",
          f"Valid scenes: {len(R)} ({len(con)} contact, {len(R) - len(con)} control); designed {len(design)}, excluded {n_inv}. "
          f"tau = {tau}; eps = {eps}; d_s = {cfg['fidelity_restart_dist']}. Restarts missing (charged as reference): {n_missing}. "
          f"EAF restart relative to the generated approach: median {np.median(offs) if offs else float('nan'):+.1f} frames.\n",
          f"- **{tp}1 risk:** {pk.upper()} {r_eaf[0]:.3f} vs cert_ref {r_cr[0]:.3f} (limit +0.02): **{'PASS' if N1 else 'FAIL'}**",
          f"- **{tp}2 cost:** saving cert_ref - {pk.upper()} {f3(sv)} s/query: **{'PASS' if N2 else 'FAIL'}**",
          f"- **{tp}3 timing:** error(chain) - error({pk.upper()}) {f3(tm)}: **{'PASS' if N3 else 'FAIL'}**",
          f"- **Confirmation: {'CONFIRMED' if (N1 and N2) else 'NOT CONFIRMED'}**; fidelity timing beats 4k timing: "
          f"**{'supported' if (N1 and N2 and N3) else 'not supported'}**", "",
          "| pipeline (contact scenes) | mean error | risk | cost / always-reference |", "|---|---|---|---|"]
    labs = (("generate only", "generate only (two objects)"), ("cert_ref", "certify or simulate"),
                   ("eaf", "**EAF repair**" if pk == "eaf" else "EAF repair (generation-timed)"),
                   ("sf", "**SF repair (shadow-timed)**"), ("chain", "chain-gated V2 (4k timing)"), ("event", "event-timed V2 (secondary)"),
                   ("always reference", "always simulate"))
    for k, lab in labs:
        if k == "sf" and not has_sf:
            continue                                         # pipeline not run in this phase (no k_sh rows)
        md.append(f"| {lab} | {err(k)[0]:.4f} | {risk(k)[0]:.3f} | {cost(k)[0] / c_ref:.2f} |")
    ctrl = [r for r in R if r["kind"] == "control"]
    if ctrl:
        good = [r for r in ctrl if r["E"] <= eps]
        md += ["", f"Controls: {len(ctrl)}; generation risk {np.mean([r['E'] > eps for r in ctrl]):.3f}; "
               f"falsely rejected among good {np.mean([not r['acc'] for r in good]) if good else float('nan'):.3f}."]
    run.path("SUMMARY.md").write_text("\n".join(md) + "\n")
    save_json({"N1": bool(N1), "N2": bool(N2), "N3": bool(N3), "risk_eaf": r_eaf, "risk_cert": r_cr, "saving": sv,
               "timing": tm, "n_contact": len(con), "n_missing": n_missing, "primary": pk}, run.path("summary.json"))
    print(f"[p4n_summarize] wrote {run.path('SUMMARY.md')}")
    run.finish("ok")


if __name__ == "__main__":
    main()
