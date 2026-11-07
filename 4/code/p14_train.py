"""Phase 14 (H200): certificate-driven simulator distillation, i.e. fine-tune PhysCtrl base on reference rollouts.

Pre-registered Oct 22, 2026 (PLAN.md "Phase 14"); the frozen settings are in configs/p14a_train_<arm>_<variant>.yaml.
PhysCtrl's base training loss as in Phase 4g (x0 prediction, DDPM noise, x0-MSE plus velocity MSE, AdamW, bf16, constant
LR after warm-up), WITHOUT frame conditioning: this is plain supervised training on simulator targets for new physics.
Each batch is half arm data (the acquired rollouts: ACT = certificate-rejected candidates, RND = a random subset of the
same size) and half replay of the original distribution (the 4g in-distribution re-simulations and the stored sample/
trajectories), against forgetting. Variants: full fine-tune, or LoRA on all attention projections (spatial attn1 and
temporal attn_temp). The final checkpoint is the model; the validation loss on held-out training objects is logged only.
lambda_distill > 0 (pilot 14a-v2; LoRA only): the replay half is also pulled towards the frozen base's prediction on the
same noisy input (adapters switched off), the 4g-v2 recipe that removed forgetting there. The arm half is never
distilled (there the base is what is wrong).
object_id: true (Phase 15a): the arm files are two-object scenes (A's 1,024 points, then B's) and get the zero-initialised
object-ID embeddings (fc_model.add_object_id; trainable in every mode); replay rows get none.

Outputs: $DATA_ROOT/checkpoints/<ckpt_name>/{step_*.pt, final.pt, latest.txt}; results/<run_name>/train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_args, append_jsonl, load_config, load_json, results_dir, resume_rows, start_run  # noqa: E402
from fc_model import add_object_id, add_temporal_lora, set_lora  # noqa: E402
from p4g_train_fc import _call, _dummy_setup  # noqa: E402


def _files(cfg):
    """(arm files, replay files) as absolute paths; both lists are fixed by earlier CPU steps."""
    from backends import ood_manifest_entries
    if "arm_list" in cfg:
        arm = load_json(cfg["arm_list"])["files"]
    else:                                   # Phase 15a: the training scenes of the build runs
        arm = [e["path"] for r in cfg["arm_build_runs"] for f in sorted(results_dir(r).glob("shard_*/ood_manifest.json"))
               for e in load_json(f) if e.get("set", "train") == "train"]
    replay = [e["path"] for e in ood_manifest_entries(False, None, tuple(cfg["replay_build_runs"]))]
    replay += [e["path"] for e in json.load(open(cfg["base_manifest"]))]
    return sorted(set(map(str, arm))), sorted(set(map(str, replay)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    run = start_run(cfg["run_name"], cfg, synthetic=args.dry_run)
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler

    log = run.path("train_log.jsonl")
    resume_rows(log, args.resume)
    ckpt_dir = Path(cfg["paths"]["ckpt_root"]) / cfg["ckpt_name"] if not args.dry_run else run.path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    tr = cfg["train"]
    if args.dry_run:      # CPU pipeline test only; the frozen config is untouched
        tr = dict(tr, steps=30, log_every=10, val_every=10, ckpt_every=15, batch_size=4)

    if args.dry_run:
        model, batch_fn, dev, drng = _dummy_setup(cfg)
        y_rule = lambda b: None                                                     # noqa: E731
        half = tr["batch_size"] // 2
        arm_iter = rep_iter = iter(lambda: batch_fn(half, drng), None)
        val_batches = [batch_fn(tr["batch_size"], np.random.default_rng(99))]
        lora_targets = ("attn_temp",)                                               # the stand-in has no attn1
    else:
        from backends import make_backend
        from torch.utils.data import DataLoader, default_collate

        backend = make_backend(cfg, False)
        gen = backend.generator("base")
        model, dev = gen.pipeline.model, gen.device
        arm, replay = _files(cfg)
        objs = sorted({Path(f).stem.split("__")[0] for f in arm + replay})
        val_objs = set(np.random.default_rng(cfg["seed"]).permutation(objs)[: max(1, int(len(objs) * tr["val_frac"]))])
        keep = lambda fs: [f for f in fs if Path(f).stem.split("__")[0] not in val_objs]     # noqa: E731
        collate = lambda b: default_collate([{k: v for k, v in it[0].items() if k != "model"} for it in b])  # noqa: E731
        half = tr["batch_size"] // 2

        def loader(fs, shuffle=True, bs=half):
            return DataLoader(gen.dataset("/", fs), batch_size=bs, shuffle=shuffle, num_workers=tr["num_workers"],
                              collate_fn=collate, drop_last=shuffle)

        def forever(dl):
            while True:
                for b in dl:
                    yield b
        arm_iter, rep_iter = forever(loader(keep(arm))), forever(loader(keep(replay)))
        va = [f for f in arm if Path(f).stem.split("__")[0] in val_objs][: tr["val_samples"]]
        val_batches = list(loader(va, shuffle=False, bs=tr["batch_size"]))
        num_mat = gen.cfg.model_config.get("num_mat", 0)
        y_rule = lambda b: None if num_mat == 0 else b["mat_type"]                  # noqa: E731
        lora_targets = tuple(tr.get("lora_targets", ("attn1", "attn_temp")))
        print(f"[p14_train] arm {len(keep(arm))} / replay {len(keep(replay))} train files; {len(objs)} objects "
              f"({len(val_objs)} val); {len(va)} val files", flush=True)

    mode = tr["mode"]
    objid = bool(cfg.get("object_id", False))
    if objid:
        add_object_id(model)
    if mode == "lora":
        model.requires_grad_(False)
        for tgt in lora_targets:
            add_temporal_lora(model, rank=tr["lora_rank"], alpha=tr["lora_alpha"], target=tgt)
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(n_.endswith(".A") or n_.endswith(".B"))
    elif mode == "full":
        model.requires_grad_(True)
    else:
        raise ValueError(mode)
    if objid:
        model.obj_id_emb.requires_grad_(True)
    model.to(dev).train()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    print(f"[p14_train] mode {mode}: {sum(p_.numel() for p_ in trainable)} trainable parameters", flush=True)
    opt = torch.optim.AdamW(trainable, lr=tr["lr"], betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    sched = DDPMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False)
    step = 0
    latest = ckpt_dir / "latest.txt"
    if args.resume and latest.exists():
        st = torch.load(ckpt_dir / latest.read_text().strip(), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = int(st["step"])
        print(f"[p14_train] resumed at step {step}", flush=True)

    def cat(a, b):
        return {k: (torch.cat([a[k], b[k]], 0) if torch.is_tensor(a[k]) else a[k]) for k in a}

    lam_d = float(tr.get("lambda_distill", 0.0))
    if lam_d > 0 and mode != "lora":
        raise ValueError("lambda_distill needs mode lora (the frozen base is the model with its adapters off)")

    def loss_on(b, n_arm=None):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        lat = b["points_tgt"]
        if objid:                       # two-object layout for arm rows (all rows when n_arm is None: validation)
            n_pts, n_rows = lat.shape[-2], lat.shape[0]
            ids = (torch.arange(n_pts, device=lat.device) >= n_pts // 2).long()[None].expand(n_rows, -1).clone()
            ids[(n_arm if n_arm is not None else n_rows):] = -1
            model.obj_ids = ids
        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (lat.shape[0],), device=lat.device).long()
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16, enabled=dev != "cpu"):
            pred = _call(model, noisy, t, b, y_rule(b))
        l_xyz = F.mse_loss(pred.float(), lat.float())
        l_vel = F.mse_loss((pred[:, 1:] - pred[:, :-1]).float(), (lat[:, 1:] - lat[:, :-1]).float())
        loss = l_xyz + tr["lambda_vel"] * l_vel
        if lam_d > 0 and n_arm is not None:
            with torch.no_grad(), torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16, enabled=dev != "cpu"):
                set_lora(model, False)
                teacher = _call(model, noisy, t, b, y_rule(b))
                set_lora(model, True)
            loss = loss + lam_d * F.mse_loss(pred[n_arm:].float(), teacher[n_arm:].float())
        return loss, l_xyz, l_vel

    while step < tr["steps"]:
        a_ = next(arm_iter)
        loss, lx, lv = loss_on(cat(a_, next(rep_iter)), n_arm=len(a_["points_tgt"]) if "points_tgt" in a_ else half)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        for g in opt.param_groups:                                   # constant LR after warm-up
            g["lr"] = tr["lr"] * min(1.0, (step + 1) / tr["warmup"])
        opt.step()
        step += 1
        if step % tr["log_every"] == 0 or step == tr["steps"]:
            row = {"step": step, "loss": float(loss), "xyz": float(lx), "vel": float(lv)}
            if (step % tr["val_every"] == 0 or step == tr["steps"]) and val_batches:
                model.eval()
                with torch.no_grad():
                    torch.manual_seed(1234)
                    row["val_loss"] = float(np.mean([float(loss_on(vb)[0]) for vb in val_batches]))
                model.train()
            append_jsonl(row, log)
            print(f"[p14_train] {row}", flush=True)
        if step % tr["ckpt_every"] == 0 or step == tr["steps"]:
            name = "final.pt" if step == tr["steps"] else f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "mode": mode, "object_id": objid,
                        "lora": {"rank": tr.get("lora_rank"), "alpha": tr.get("lora_alpha"), "targets": list(lora_targets)}
                        if mode == "lora" else None}, ckpt_dir / name)
            latest.write_text(name)
    run.finish("ok", final=str(ckpt_dir / "final.pt"))


if __name__ == "__main__":
    main()
