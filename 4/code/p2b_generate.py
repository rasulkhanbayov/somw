"""Phase 2b (part 1, H200): generate K trajectory samples per query.

For every in-distribution elastic test query and every OOD query, draw K
samples from each PhysCtrl model with a fixed seed, and save
  <gen_dir>/<model>/<split>/<qid>.npz : samples (K, 24, N, 3) float32 in
  simulator coordinates (frames 1,3,...,47), seconds (generation wall time).
Sample 0 is "the" output the verifier certifies; samples 1..K-1 give the
seed-spread baseline (the cheapest uncertainty signal a reviewer will ask for).
Resumable: existing files are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_args, load_config, shard_items, start_run, RESULTS_ROOT  # noqa: E402
from backends import make_backend, ood_manifest_entries, testset_manifest_path  # noqa: E402


def gen_dir(cfg: dict, dry_run: bool) -> Path:
    return (RESULTS_ROOT / "SYNTHETIC_p2b_generations") if dry_run else Path(cfg["paths"]["gen_dir"])


def all_entries(cfg: dict, backend, dry_run: bool) -> list[dict]:
    entries = []
    if cfg.get("include_test_id", True):
        entries = backend.manifest(testset_manifest_path(dry_run), split="test_id")[: cfg["n_id_queries"]]
    try:
        ood = ood_manifest_entries(dry_run, backend, tuple(cfg.get("query_runs", ["p2_build_ood"])))
        for split in sorted({e["split"] for e in ood}):
            entries += [e for e in ood if e["split"] == split][: cfg["n_ood_per_split"]]
    except FileNotFoundError as ex:
        print(f"[p2b] OOD queries missing ({ex}); in-distribution only.")
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    shard = args.shard if args.shard is not None else 0
    run = start_run(cfg.get("generate_run_name", "p2b_generate"), cfg, synthetic=args.dry_run, shard=str(shard))
    backend = make_backend(cfg, args.dry_run)
    entries = shard_items(all_entries(cfg, backend, args.dry_run), shard, args.num_shards)
    root = gen_dir(cfg, args.dry_run)
    n_done = 0
    for which in cfg["models"]:
        for e in entries:
            path = root / which / e["split"] / f"{e['qid']}.npz"
            if path.exists():
                continue
            q = backend.load(e)
            t0 = time.perf_counter()
            samples = backend.generate(which, [q], n_samples=cfg["n_samples"], seed=cfg["seed"])[0]
            secs = time.perf_counter() - t0
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, samples=samples.astype(np.float32), seconds=secs)
            n_done += 1
        print(f"[p2b_generate] {which}: done")
    run.finish("ok", n_generated=n_done)


if __name__ == "__main__":
    main()
