"""Paper qualitative figures: case selection on the Phase 9 test sets (Oct 19, 2026). CPU. Rule fixed before any picture.

The deck's median rule (results/fig_qualitative/deck_selection.json), applied to the Phase 9 data:
  shift    one row per family ood3_E_low (softer), ood3_force (stronger drag), ood3_drop (drop onto a floor): the query
           with the lower-median PhysCtrl error (sample 0; sorted by error then qid, index (n - 1) // 2)
  contact  (a) the lower-median PhysCtrl error among all contact scenes; (b) the same among scenes the certificate
           rejects whose shadow-timed switch falls mid-clip (k_sw exists)
Each case records ours-v2's decision on it: accept (ours = generation), repair at chain frame k (single objects:
M1 shadow + M2 suffix) or at the switch frame k_sw (contact: coarse shadow + M3 suffix), or exact (reference fallback).
Output: results/fig_qualitative/paper_selection.json (read by fig_ours_traj.py --selection and fig_paper_points.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_jsonl, results_dir  # noqa: E402
from p4n_summarize import build  # noqa: E402
from p9_confirm import assemble  # noqa: E402


def lower_median(items):
    items = sorted(items, key=lambda t: (t[1], t[0]))
    return items[(len(items) - 1) // 2]


def main():
    cfg, ccfg = load_config("configs/p9_confirm.yaml"), load_config("configs/p9c_summary.yaml")
    _, A, *_ = assemble(cfg)
    chain_k = [int(k) for k in cfg["chain_k"]]
    cases = []
    for fam, name in (("ood3_E_low", "shift_soft"), ("ood3_force", "shift_force"), ("ood3_drop", "shift_drop")):
        idx = [i for i, s in enumerate(A["split"]) if s == fam]
        i = lower_median([(A["qid"][j], float(A["E"][j]), j) for j in idx])[2]
        dec = "accept" if A["acc_one"][i] else ("repair" if A["gs_one"][i] else "exact")
        c = {"name": name, "pool": "p9", "qid": str(A["qid"][i]), "split": fam, "decision": dec, "gen_err": float(A["E"][i])}
        if dec == "repair":
            c["k"] = chain_k[int(A["kk_one"][i])]
        cases.append(c)
    R = [r for r in build(ccfg)["R"] if r["kind"] == "contact"]
    allk = {r["qid"]: r for f in sorted(results_dir(ccfg["allk_run"]).glob("shard_*/allk.jsonl")) for r in read_jsonl(f)}
    def contact_case(name, pool_rows):
        r = lower_median([(x["qid"], float(x["E"]), x) for x in pool_rows])[2]
        ksw = allk[r["qid"]].get("k_sw")
        dec = "accept" if r["acc"] else ("repair" if ksw is not None else "exact")
        c = {"name": name, "pool": "p9c", "qid": r["qid"], "split": r["split"], "decision": dec, "gen_err": float(r["E"])}
        if dec == "repair":
            c["k"] = int(ksw)
        return c
    cases.append(contact_case("contact_median", R))
    cases.append(contact_case("contact_switch", [r for r in R if not r["acc"] and allk[r["qid"]].get("k_sw") is not None]))
    out = Path("results/fig_qualitative/paper_selection.json")
    rule = (__doc__.split("Rule fixed before any picture.")[1].split("Output:")[0]).strip()
    out.write_text(json.dumps({"rule": rule, "fixed": "2026-10-19, before any paper picture", "cases": cases}, indent=1))
    for c in cases:
        print(c)


if __name__ == "__main__":
    main()
