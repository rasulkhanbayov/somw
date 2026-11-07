"""Phase 14 (H200): sample 0 of PhysCtrl base and of each fine-tuned checkpoint on the evaluation queries.

Pre-registered Oct 22, 2026 (PLAN.md "Phase 14"). Every model runs the same procedure (one sample, seed 0, the bridge's
Generator.sample), so base is regenerated here rather than read from older runs. A LoRA checkpoint gets its adapters
re-attached (same targets, rank, alpha) before its weights are loaded; a full checkpoint is loaded directly; loading is
strict, so a mismatch stops the job.
Output: <out_dir>/<model>/<split>/<qid>.npz (samples: (1, 24, N, 3), simulator coordinates); resumable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_args, load_config, shard_items  # noqa: E402


def entries(cfg):
    from backends import ood_manifest_entries
    out = [e for e in ood_manifest_entries(False, None, tuple(cfg["query_runs"])) if e["split"] in set(cfg["splits"])]
    return sorted(out, key=lambda e: (e["split"], e["qid"]))


def make_generator(cfg, ckpt):
    import torch
    from fc_model import add_object_id, add_temporal_lora
    from physctrl_bridge import Generator
    g = Generator("base", cfg["paths"]["ckpt_dir"])
    if ckpt:
        st = torch.load(os.path.expandvars(ckpt), map_location="cpu", weights_only=False)
        model = g.pipeline.model
        if st.get("object_id"):
            add_object_id(model)
        if st.get("mode") == "lora":
            for tgt in st["lora"]["targets"]:
                add_temporal_lora(model, rank=st["lora"]["rank"], alpha=st["lora"]["alpha"], target=tgt)
        model.load_state_dict(st["model"], strict=True)
        model.to(g.device).eval()
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    out = Path(os.path.expandvars(cfg["out_dir"]))
    items = shard_items(entries(cfg), args.shard or 0, args.num_shards)
    for name, ckpt in cfg["models"].items():
        todo = [e for e in items if not (out / name / e["split"] / f"{e['qid']}.npz").exists()]
        if not todo:
            continue
        g = make_generator(cfg, ckpt)
        for e in todo:
            path = Path(e["path"])
            item = g.dataset(path.parent, [path.name])[0][0]
            s = g.sample([item], 1, int(cfg["seed"]))[0]
            dst = out / name / e["split"] / f"{e['qid']}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, samples=s.astype(np.float32))
        print(f"[p14 eval] {name}: {len(todo)} generations", flush=True)
        del g


if __name__ == "__main__":
    main()
