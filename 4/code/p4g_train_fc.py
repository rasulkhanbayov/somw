"""Phase 4g (H200): fine-tune PhysCtrl base into a frame-conditioned generator.

Pre-registered Oct 12, 2026 (PLAN.md "Phase 4g"); the frozen settings are in
configs/p4g_train.yaml. Training follows PhysCtrl's own loop for base (x0 prediction, DDPM
noise, x0-MSE plus velocity MSE, AdamW, bf16), with one change: per sample, a random mask
(fc_model.sample_mask) marks known frames. Those frames are written clean into the noisy input
and flagged by the wrapper's known-frame embedding. PhysCtrl's 0.001-weighted deformation loss
is left out (it needs the stored F and C of every frame). The last checkpoint is the model;
the validation loss (held-out training objects) is logged, never used to pick a checkpoint.

Data: PhysCtrl's sample/ objects (100 objects, no overlap with the evaluation objects), their
stored trajectories plus the in-distribution re-simulations built by p2_build_ood.py
(configs/p4g_train_build.yaml).

Outputs: $DATA_ROOT/checkpoints/<ckpt_name>/{step_*.pt, final.pt, latest.txt}; results/p4g_train/train_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import add_common_args, append_jsonl, load_config, read_jsonl, resume_rows, start_run  # noqa: E402
from fc_model import add_temporal_lora, make_fc, sample_mask, set_lora  # noqa: E402


# ----------------------------------------------------------------------------- dry run
def _dummy_setup(cfg):
    """CPU stand-in with the MDM_ST interface the wrapper relies on. Numbers mean nothing."""
    import torch

    class Dummy(torch.nn.Module):
        latent_dim, cond_frame = 16, 1

        def __init__(self):
            super().__init__()
            self.input_encoder = torch.nn.Linear(3, 16)
            self.out = torch.nn.Linear(16, 3)
            self.attn_temp = torch.nn.Module()            # stand-in with the attention layout
            for a in ("to_q", "to_k", "to_v"):
                setattr(self.attn_temp, a, torch.nn.Linear(16, 16))
            self.attn_temp.to_out = torch.nn.ModuleList([torch.nn.Linear(16, 16)])

        def forward(self, x, t, init_pc, *a, **kw):
            h = torch.cat([init_pc.unsqueeze(1), x], 1)
            bs, nf, n, _ = h.shape
            z = self.input_encoder(h.reshape(-1, n, 3)).reshape(bs, nf, n, 16)
            at = self.attn_temp
            z = z + at.to_out[0](at.to_v(z) * torch.sigmoid(at.to_q(z) * at.to_k(z)))
            return self.out(z)[:, 1:] + init_pc.unsqueeze(1)

    rng = np.random.default_rng(0)
    n, k = 64, 24
    def batch_fn(bs, r):
        x0 = r.normal(size=(bs, 1, n, 3)).astype(np.float32)
        traj = x0 + np.cumsum(r.normal(scale=0.01, size=(bs, k, n, 3)), 1).astype(np.float32)
        t = lambda a: torch.from_numpy(a)
        return {"points_src": t(x0), "points_tgt": t(traj)}
    return Dummy(), batch_fn, "cpu", rng


def _call(model_or_fc, noisy, t, batch, y):
    if "force" not in batch:                          # dry-run stand-in
        return model_or_fc(noisy, t, batch["points_src"][:, 0])
    return model_or_fc(noisy, t, batch["points_src"], batch["force"], batch["E"], batch["nu"],
                       batch["mask"][..., :1], batch["drag_point"], batch["floor_height"], batch["gravity"],
                       batch["base_drag_coeff"], y=y, null_emb=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dry_run)
    run = start_run(cfg.get("run_name", "p4g_train"), cfg, synthetic=args.dry_run)
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler

    log = run.path("train_log.jsonl")
    resume_rows(log, args.resume)
    ckpt_dir = Path(cfg["paths"]["ckpt_root"]) / cfg["ckpt_name"] if not args.dry_run else run.path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    mrng = np.random.default_rng(cfg["seed"])
    tr = cfg["train"]
    if args.dry_run:      # CPU pipeline test only; the frozen config is untouched
        tr = dict(tr, steps=30, log_every=10, val_every=10, ckpt_every=15, batch_size=4)

    if args.dry_run:
        model, batch_fn, dev, drng = _dummy_setup(cfg)
        y_rule = lambda b: None
        train_iter = lambda: iter(lambda: batch_fn(tr["batch_size"], drng), None)
        val_batches = [batch_fn(tr["batch_size"], np.random.default_rng(99))]
    else:
        from backends import make_backend, ood_manifest_entries
        from torch.utils.data import DataLoader, default_collate

        backend = make_backend(cfg, False)
        gen = backend.generator("base")
        model, dev = gen.pipeline.model, gen.device
        data_root = Path(cfg["paths"]["data_root"])
        entries = ood_manifest_entries(False, backend, tuple(cfg["train_build_runs"]))
        entries += json.load(open(cfg["base_manifest"]))           # the stored sample trajectories
        files = sorted({str(Path(e["path"]).relative_to(data_root)) for e in entries})
        objs = sorted({Path(f).stem.split("__")[0] for f in files})
        val_objs = set(np.random.default_rng(cfg["seed"]).permutation(objs)[: max(1, int(len(objs) * tr["val_frac"]))])
        tr_files = [f for f in files if Path(f).stem.split("__")[0] not in val_objs]
        va_files = [f for f in files if Path(f).stem.split("__")[0] in val_objs]
        collate = lambda b: default_collate([{k: v for k, v in it[0].items() if k != "model"} for it in b])
        ds = gen.dataset(data_root, tr_files)
        dl = DataLoader(ds, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["num_workers"],
                        collate_fn=collate, drop_last=True)
        vds = gen.dataset(data_root, va_files[: tr["val_samples"]])
        val_batches = list(DataLoader(vds, batch_size=tr["batch_size"], shuffle=False, collate_fn=collate))
        num_mat = gen.cfg.model_config.get("num_mat", 0)
        y_rule = lambda b: None if num_mat == 0 else b["mat_type"]

        def train_iter():
            while True:
                for b in dl:
                    yield b
        print(f"[p4g_train] {len(tr_files)} train / {len(va_files)} val files; {len(objs)} objects ({len(val_objs)} val)")

    lora_cfg = cfg.get("lora") if tr.get("mode", "full") == "lora" else None
    if lora_cfg:                       # v2: frozen base + temporal-attention LoRA
        model.requires_grad_(False)
        add_temporal_lora(model, **lora_cfg)
    fc = make_fc(model).to(dev)
    model.train()
    if lora_cfg:
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(n_.endswith(".A") or n_.endswith(".B"))
    else:
        model.requires_grad_(True)
    trainable = [p_ for p_ in fc.parameters() if p_.requires_grad]
    print(f"[p4g_train] mode {tr.get('mode', 'full')}: {sum(p_.numel() for p_ in trainable)} trainable parameters")
    opt = torch.optim.AdamW(trainable, lr=tr["lr"], betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    sched = DDPMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False)
    step = 0
    latest = ckpt_dir / "latest.txt"
    if args.resume and latest.exists():
        st = torch.load(ckpt_dir / latest.read_text().strip(), map_location="cpu", weights_only=False)
        fc.model.load_state_dict(st["model"]); fc.known_emb.data.copy_(st["known_emb"])
        opt.load_state_dict(st["opt"]); step = int(st["step"])
        mrng = np.random.default_rng(cfg["seed"] + step)
        print(f"[p4g_train] resumed at step {step}")
    elif not args.dry_run:
        # Identity check: with no known frames the wrapped model must be exactly PhysCtrl base.
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in val_batches[0].items()}
        with torch.no_grad(), torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16):
            x = torch.randn_like(b["points_tgt"]); t = torch.full((x.shape[0],), 500, device=dev)
            fc.mask = None; a = _call(model, x, t, b, y_rule(b))
            fc.mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=dev); c = _call(fc, x, t, b, y_rule(b))
        diff = float((a - c).abs().max()); fc.mask = None
        if lora_cfg:                   # adapters on vs off must also agree at init (B = 0)
            with torch.no_grad(), torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16):
                set_lora(model, False); a2 = _call(model, x, t, b, y_rule(b)); set_lora(model, True)
            diff = max(diff, float((a - a2).abs().max()))
        print(f"[p4g_train] identity check at init: max |base - wrapped| = {diff:.2e}")
        if diff > 0:
            raise RuntimeError("frame-conditioned wrapper is not an identity at initialisation")

    def loss_on(b, rng, train=True):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        lat = b["points_tgt"]
        bs = lat.shape[0]
        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (bs,), device=lat.device).long()
        noisy = sched.add_noise(lat, noise, t)
        M = torch.from_numpy(np.stack([sample_mask(rng, lat.shape[1], **cfg["mask"]) for _ in range(bs)])).to(lat.device)
        noisy = torch.where(M[:, :, None, None], lat, noisy)
        fc.mask = M
        with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16, enabled=dev != "cpu"):
            pred = _call(fc, noisy, t, b, y_rule(b))
        fc.mask = None
        l_xyz = F.mse_loss(pred.float(), lat.float())
        l_vel = F.mse_loss((pred[:, 1:] - pred[:, :-1]).float(), (lat[:, 1:] - lat[:, :-1]).float())
        loss = l_xyz + tr["lambda_vel"] * l_vel
        lam_d = float(tr.get("lambda_distill", 0.0))
        if lam_d > 0:                  # v2: stay equal to the frozen base when nothing is known
            pure = sched.add_noise(lat, noise, t)
            with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16, enabled=dev != "cpu"):
                s_ = _call(fc, pure, t, b, y_rule(b))
                with torch.no_grad():
                    set_lora(model, False); t_ = _call(model, pure, t, b, y_rule(b)); set_lora(model, True)
            loss = loss + lam_d * F.mse_loss(s_.float(), t_.float())
        return loss, l_xyz, l_vel

    it = train_iter()
    while step < tr["steps"]:
        loss, lx, lv = loss_on(next(it), mrng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fc.parameters(), 1.0)
        for g in opt.param_groups:                                   # constant LR after warm-up
            g["lr"] = tr["lr"] * min(1.0, (step + 1) / tr["warmup"])
        opt.step()
        step += 1
        if step % tr["log_every"] == 0 or step == tr["steps"]:
            row = {"step": step, "loss": float(loss), "xyz": float(lx), "vel": float(lv),
                   "known_emb_norm": float(fc.known_emb.norm())}
            if step % tr["val_every"] == 0 or step == tr["steps"]:
                model.eval()
                with torch.no_grad():
                    vr = np.random.default_rng(1234)
                    row["val_loss"] = float(np.mean([float(loss_on(vb, vr)[0]) for vb in val_batches]))
                model.train()
            append_jsonl(row, log)
            print(f"[p4g_train] {row}")
        if step % tr["ckpt_every"] == 0 or step == tr["steps"]:
            name = "final.pt" if step == tr["steps"] else f"step_{step}.pt"
            torch.save({"model": fc.model.state_dict(), "known_emb": fc.known_emb.detach().cpu(),
                        "opt": opt.state_dict(), "step": step, "lora": lora_cfg}, ckpt_dir / name)
            latest.write_text(name)
    run.finish("ok", final=str(ckpt_dir / "final.pt"))


if __name__ == "__main__":
    main()
