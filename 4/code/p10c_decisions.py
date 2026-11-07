"""Phase 10c step 1 (CPU, pre-registered Oct 21, 2026; PLAN.md "Phase 10c"): every Phase 9 pipeline's decision per query,
fixed from the Phase 9 records before the GPU re-run, plus the stored per-frame errors the re-run must reproduce.
Single objects (p9_confirm.assemble): ours (accept / repair at chain frame k / reference), certify-or-simulate, the
momentum-residual gate and seed-spread deferral (accept / reference). Contact (p4n_summarize.build): ours = the
shadow-timed switch (accept / repair at k_sw / reference) and certify-or-simulate.
Output: results/p10c_decisions/decisions.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir, save_json, start_run  # noqa: E402
from p4n_summarize import build  # noqa: E402
from p9_confirm import assemble  # noqa: E402


def full_errs(run):
    return {(r["qid"], r["fidelity"]): r.get("err_t") for f in sorted(results_dir(run).glob("shard_*/fullsim.jsonl"))
            for r in read_jsonl(f)}


def main():
    scfg, ccfg = load_config("configs/p9_confirm.yaml"), load_config("configs/p9c_summary.yaml")
    run = start_run("p10c_decisions", {"single": scfg, "contact": ccfg}, synthetic=False)
    _, A, *_ = assemble(scfg)
    chain_k = [int(k) for k in scfg["chain_k"]]
    allk = {r["qid"]: r for f in sorted(results_dir(scfg["run_name"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    fe = full_errs(scfg["full_run"])
    out = []
    for i, (q, sp) in enumerate(zip(A["qid"], A["split"])):
        if A["acc_one"][i]:
            ours, k, ref = "accept", None, None
        elif A["gs_one"][i]:
            k = chain_k[int(A["kk_one"][i])]
            ours, ref = "repair", allk[q]["k"][str(k)]["err_t"]
        else:
            ours, k, ref = "reference", None, None
        out.append({"qid": q, "set": "single", "split": sp, "obj": str(A["obj"][i]), "ours": ours, "k": k,
                    "ours_err_t": ref, "cert": "accept" if A["acc_one"][i] else "reference",
                    "residual": "accept" if A["acc_r"][i] else "reference",
                    "seed": "accept" if A["acc_s"][i] else "reference",
                    "full_err_t": {f: fe.get((q, f)) for f in ("coarse_cfl", "M1", "M2", "M3")}})
    Bd = build(ccfg)
    callk = {r["qid"]: r for f in sorted(results_dir(ccfg["allk_run"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    cfe = full_errs(scfg["contact_full_run"])
    for r in Bd["R"]:
        if r["kind"] != "contact":
            continue
        q, ksw = r["qid"], callk[r["qid"]].get("k_sw")
        if r["acc"]:
            ours, k, ref = "accept", None, None
        elif ksw is not None:
            v = callk[q]["k"].get(str(ksw))
            ours, k, ref = ("repair", int(ksw), v["err_t"]) if v and not (v.get("blown") or v.get("shadow_blown")) else ("reference", None, None)
        else:
            ours, k, ref = "reference", None, None
        out.append({"qid": q, "set": "contact", "split": r["split"], "obj": r["obj"], "ours": ours, "k": k,
                    "ours_err_t": ref, "cert": "accept" if r["acc"] else "reference",
                    "full_err_t": {f: cfe.get((q, f)) for f in ("M1", "M2", "M3")}})
    save_json({"decisions": out, "chain_k": chain_k}, run.path("decisions.json"))
    n = {s: sum(o["set"] == s for o in out) for s in ("single", "contact")}
    print(f"[p10c] {n}; ours decisions: " + str({d: sum(o['ours'] == d for o in out) for d in ('accept', 'repair', 'reference')}))
    run.finish("ok")


if __name__ == "__main__":
    main()
