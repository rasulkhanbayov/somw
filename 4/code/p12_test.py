"""Phase 12 test runs (CPU, cpu-all; pre-registered Oct 22, 2026, PLAN.md "Phase 12"): the independent simulator S2
(PolyFEM, p12_fem.py) on the Phase 9 single-object test queries, at the time step chosen by the S0 gate.

Runs only if results/p12_s0_v2/s0.json says S0 PASS (the pre-registered rule: S0 FAIL -> no cross-check); otherwise it
writes nothing and exits 0 with a message.
Queries: the primary sets id_new_22, id_new_23 (in distribution), ood3_E_high, ood3_E_low, ood3_force (500), as in
results/p10c_decisions. Drop (secondary) is not run: our runner has no floor contact (recorded in PLAN before any test
query ran). Each query: run_query at n_div from S0; mesh-rule exclusions and solver failures are recorded, never retried.
Output: $BIG_ROOT/p12_traj/test/<qid>.npz (s2: (24, N, 3) at simulator frames 1, 3, ..., 47; info) and
        results/p12_test/shard_<i>/rows.jsonl (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_ROOT, append_jsonl, load_json  # noqa: E402
from p12_fem import load_query, run_query  # noqa: E402

PRIMARY = ("id_new_22", "id_new_23", "ood3_E_high", "ood3_E_low", "ood3_force")
TRAJ = Path(os.path.expandvars("${BIG_ROOT}/p12_traj/test"))
WORK = Path(os.path.expandvars("${BIG_ROOT}/tmp/p12_test"))


def queries():
    root = Path(RESULTS_ROOT)
    paths = {e["qid"]: e["path"] for f in sorted((root / "p9_build").glob("shard_*/ood_manifest.json")) for e in load_json(f)}
    dec = load_json(root / "p10c_decisions" / "decisions.json")["decisions"]
    return sorted((d["qid"], paths[d["qid"]]) for d in dec if d["set"] == "single" and d["split"] in PRIMARY)


def job(args):
    qid, path, n_div, threads = args
    dst = TRAJ / f"{qid}.npz"
    if dst.exists():
        return {"qid": qid, "status": "done before"}
    q = load_query(path)
    try:
        pos, info = run_query(q, n_div, WORK / qid, threads)
    except RuntimeError as ex:                       # solver failure: recorded, not retried
        return {"qid": qid, "status": "failed", "error": str(ex)[-600:]}
    if pos is None:
        return {"qid": qid, "status": "excluded (mesh rule)", **{k: info[k] for k in ("largest_component_share", "n_cells")}}
    TRAJ.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in info.items() if k not in ("mesh_com",)}
    np.savez_compressed(dst, s2=pos.astype(np.float32), info=json.dumps(keep))
    return {"qid": qid, "status": "ok", "seconds": info.get("seconds")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    s0 = Path(RESULTS_ROOT) / "p12_s0_v2" / "s0.json"
    if not s0.exists():
        print(f"[p12_test] {s0} missing: run the S0 gate first (job 102). Nothing done.")
        return
    g = load_json(s0)
    if not g.get("S0"):
        print("[p12_test] S0 FAILED: by the pre-registered rule there is no cross-check. Nothing done.")
        return
    items = queries()[args.shard::args.num_shards]
    out = Path(RESULTS_ROOT) / "p12_test" / f"shard_{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.jsonl"
    seen = {json.loads(line)["qid"] for line in rows.read_text().splitlines()} if rows.exists() else set()
    todo = [(q, p, int(g["n_div"]), args.threads) for q, p in items if q not in seen]
    print(f"[p12_test] shard {args.shard}/{args.num_shards}: {len(todo)} of {len(items)} queries to run at n_div {g['n_div']}",
          flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(job, todo):
            append_jsonl(r, rows)
            print(f"[p12_test] {r['qid']}: {r['status']}", flush=True)


if __name__ == "__main__":
    main()
