---
license: other
tags: [physics, diffusion, fine-tuning]
---
# Simulate Only What Matters: fine-tuned PhysCtrl checkpoints

Checkpoints trained in the project *Simulate Only What Matters* (frozen Nov 3, 2026). All are fine-tunes of PhysCtrl's
base model and all were **negative results** (see `PLAN.md`, Phases 4g, 4g-v2, 14a):
- `fc_base_v1/`: Phase 4g frame-conditioned full fine-tune (forgetting: +47% in-distribution error).
- `fc_base_v2/`: Phase 4g-v2 temporal-attention LoRA + known-frame embedding (no forgetting, no repair gain).
- `p14a_{ACT,RND}_{F,L}/`: Phase 14a simulator-distillation pilot (KILL by the pre-registered rule).
Each `final.pt` (and `step_*.pt`) holds the state dict, optimizer state, step and LoRA settings; loaders:
`code/fc_model.load_fc` (4g) and `code/p14_eval_generate.make_generator` (14a).
**Licence:** derived from PhysCtrl's released weights, which declare no licence; check PhysCtrl's terms before reuse.
