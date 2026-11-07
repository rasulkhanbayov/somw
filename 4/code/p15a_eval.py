"""Phase 15a (H200; pre-registered Oct 22, 2026, clarified Oct 24 before data, PLAN.md "Phase 15"): generations on
the 200 development scenes (test objects; exploratory) for the three systems of the kill rule. Sample 0, seed 0.
  composed     PhysCtrl base on A's own 2,048-point file (A's force) and on B's rest file (zero force), each on its
               fixed FPS subset, B translated by t_b (what PhysCtrl does with two objects: it ignores the other one)
  base_joint   PhysCtrl base on the 2,048-point two-object model file, no fine-tuning, no object IDs
  <name>       each fine-tuned checkpoint in cfg["models"] on the model file, with A/B object IDs
Output: <out_dir>/<system>/<split>/<scene>.npz (samples: (1, 24, 2048, 3), simulator coordinates); resumable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_args, load_config, load_json, results_dir, shard_items  # noqa: E402
from p14_eval_generate import make_generator  # noqa: E402


def scenes(cfg, dry_run=False):
    out = [e for f in sorted(results_dir(cfg["build_run"], dry_run).glob("shard_*/ood_manifest.json"))
           for e in load_json(f) if e["set"] == "dev"]
    return sorted(out, key=lambda e: e["qid"])


def one(g, path, seed):
    path = Path(path)
    return g.sample([g.dataset(path.parent, [path.name])[0][0]], 1, seed)[0]          # (1, 24, N, 3)


def main():
    import h5py
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    out = Path(os.path.expandvars(cfg["out_dir"]))
    items = shard_items(scenes(cfg, args.dry_run), args.shard or 0, args.num_shards)
    seed = int(cfg["seed"])
    systems = {"composed": None, "base_joint": None, **cfg["models"]}
    for name, ckpt in systems.items():
        todo = [e for e in items if not (out / name / e["split"] / f"{e['qid']}.npz").exists()]
        if not todo:
            continue
        g = make_generator(cfg, ckpt)
        model = g.pipeline.model
        for e in todo:
            if name == "composed":
                with h5py.File(e["path"], "r") as h:
                    sel_a, sel_b, t_b = np.asarray(h["sel_a"]), np.asarray(h["sel_b"]), np.asarray(h["t_b"])
                ga, gb = one(g, e["a_path"], seed), one(g, e["rest_path"], seed)
                s = np.concatenate([ga[:, :, sel_a], gb[:, :, sel_b] + t_b], axis=2)
            else:
                if ckpt is not None:
                    n = 2048
                    model.obj_ids = (torch.arange(n) >= n // 2).long()[None]
                s = one(g, e["path"], seed)
                if ckpt is not None:
                    model.obj_ids = None
            dst = out / name / e["split"] / f"{e['qid']}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, samples=s.astype(np.float32))
        print(f"[p15a eval] {name}: {len(todo)} generations", flush=True)
        del g


if __name__ == "__main__":
    main()
