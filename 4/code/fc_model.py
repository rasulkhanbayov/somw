"""Frame-conditioned PhysCtrl (Phase 4g): the same DiT, told which frames are known.

PhysCtrl's MDM_ST encodes every frame's points with `input_encoder` (frame 0 of its input is
the conditioning point cloud, cond_frame = 1). A forward hook adds one learned vector,
`known_emb`, to the encoded tokens of every frame marked known. known_emb starts at zero, so
the wrapped model is exactly PhysCtrl until it is trained, and with mask None it is always
exactly PhysCtrl. Known frames are also written clean into the noisy input, both in training
and at inference (Generator.sample_constrained with fc=...).

torch is imported lazily so the rest of code/ stays NumPy-only (CLAUDE.md).
"""
from __future__ import annotations

import numpy as np


def sample_mask(rng: np.random.Generator, n_frames: int = 24, window: int = 4, p_none: float = 0.1,
                p_anchor: float = 0.7, max_windows: int = 3) -> np.ndarray:
    """Training mask (Phase 4g, pre-registered): none with p_none; else the anchor (frames
    0..window-1) with p_anchor, plus 0..max_windows random windows of `window` frames."""
    m = np.zeros(n_frames, dtype=bool)
    if rng.random() < p_none:
        return m
    if rng.random() < p_anchor:
        m[:window] = True
    for _ in range(int(rng.integers(0, max_windows + 1))):
        k = int(rng.integers(0, n_frames - window + 1))
        m[k:k + window] = True
    return m


def add_temporal_lora(model, rank: int = 8, alpha: float = 8.0, target: str = "attn_temp") -> list:
    """Phase 4g-v2: wrap to_q/to_k/to_v/to_out.0 of every `target` attention module in a LoRA
    layer (B zero-initialised, so the model is unchanged at init). Returns the LoRA modules;
    switch them off with set_lora(model, False) to recover the frozen base exactly."""
    import torch

    class LoRALinear(torch.nn.Module):
        def __init__(self, base: torch.nn.Linear):
            super().__init__()
            self.base = base
            self.A = torch.nn.Parameter(torch.randn(rank, base.in_features) / rank)
            self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank))
            self.scale, self.enabled = alpha / rank, True

        def forward(self, x):
            y = self.base(x)
            if not self.enabled:
                return y
            return y + self.scale * torch.nn.functional.linear(torch.nn.functional.linear(x, self.A), self.B)

    loras = []
    for name, mod in list(model.named_modules()):
        if name.split(".")[-1] != target:
            continue
        for attr in ("to_q", "to_k", "to_v"):
            lin = getattr(mod, attr)
            setattr(mod, attr, LoRALinear(lin)); loras.append(getattr(mod, attr))
        mod.to_out[0] = LoRALinear(mod.to_out[0]); loras.append(mod.to_out[0])
    if not loras:
        raise RuntimeError(f"no '{target}' attention modules found for LoRA")
    return loras


def set_lora(model, enabled: bool) -> None:
    for m in model.modules():
        if hasattr(m, "enabled") and hasattr(m, "B"):
            m.enabled = enabled


def add_object_id(model, n_objects: int = 2):
    """Phase 15a: zero-initialised object-ID embeddings (one vector per object) added to the input encoder's output
    of every frame. They are a parameter of the model itself (key `obj_id_emb`), so checkpoints load strictly.
    Set `model.obj_ids` to a long tensor (bs or 1, N) before a call: -1 adds nothing (single-object rows), 0 = A,
    1 = B. With obj_ids None, or the embedding at zero, the model is exactly PhysCtrl."""
    import torch

    model.obj_id_emb = torch.nn.Parameter(torch.zeros(n_objects, int(model.latent_dim)))
    model.obj_ids = None

    def hook(module, inputs, out):
        ids = model.obj_ids
        if ids is None:
            return out
        n = out.shape[-2]
        if ids.shape[-1] != n:
            raise ValueError(f"obj_ids has {ids.shape[-1]} points, the encoder output {n}")
        bs = ids.shape[0]
        o = out.reshape(-1, *out.shape[-2:])                    # (bs_total * frames, N, D)
        rows = o.shape[0]
        ids = ids.to(out.device)
        add = model.obj_id_emb.to(out.dtype)[ids.clamp(min=0)] * (ids >= 0)[..., None].to(out.dtype)   # (bs, N, D)
        if bs == 1:
            o = o + add
        else:
            o = (o.reshape(bs, rows // bs, n, -1) + add[:, None]).reshape(rows, n, -1)
        return o.reshape(out.shape)

    model.input_encoder.register_forward_hook(hook)
    return model


def make_fc(model):
    """Wrap a PhysCtrl MDM_ST. Returns an nn.Module holding the model and known_emb."""
    import torch

    class FrameCond(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
            self.known_emb = torch.nn.Parameter(torch.zeros(int(m.latent_dim)))
            self.cond_frame = int(m.cond_frame)
            self.mask = None                         # (bs, n_frames) bool tensor, set per call
            m.input_encoder.register_forward_hook(self._hook)

        def _hook(self, module, inputs, out):
            if self.mask is None:
                return out
            bs, nf = self.mask.shape
            o = out.reshape(bs, nf + self.cond_frame, *out.shape[1:])
            add = torch.zeros(bs, nf + self.cond_frame, device=out.device, dtype=out.dtype)
            add[:, self.cond_frame:] = self.mask.to(out.dtype)
            o = o + add[:, :, None, None] * self.known_emb.to(out.dtype)
            return o.reshape(out.shape)

        def forward(self, *a, **kw):
            return self.model(*a, **kw)

    return FrameCond(model)


def save_fc(fc, path, lora: dict | None = None) -> None:
    import torch
    torch.save({"model": fc.model.state_dict(), "known_emb": fc.known_emb.detach().cpu(), "lora": lora}, path)


def load_fc(gen, path):
    """Load a fine-tuned checkpoint into a Generator's model; returns the wrapper."""
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st.get("lora"):                       # v2 checkpoints: re-create the adapters first
        add_temporal_lora(gen.pipeline.model, **st["lora"])
    fc = make_fc(gen.pipeline.model)
    fc.model.load_state_dict(st["model"], strict=True)
    with torch.no_grad():
        fc.known_emb.copy_(st["known_emb"])
    fc.to(gen.device).eval().requires_grad_(False)
    return fc
