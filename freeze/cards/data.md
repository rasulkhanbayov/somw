---
license: other
pretty_name: "Simulate Only What Matters: simulated trajectories, generations and renders"
tags: [physics, video-generation, simulation, mpm]
---
# Simulate Only What Matters: data

Simulated reference trajectories (Warp MPM, HDF5), PhysCtrl generations (NPZ), probe and re-scoring trajectories, and
rendered videos of the project *Simulate Only What Matters* (frozen Nov 3, 2026).
Layout: `data/...` restores to `$DATA_ROOT/...` and `big/...` to `$BIG_ROOT/...` (see `freeze/restore.py` in the code
repository). Sizes and sha256 of every file: `manifests/` in the results repository.

**Licence and provenance.** The objects are PhysCtrl's released objects (derived from Objaverse via TRELLIS-500K), and
the generations and videos come from PhysCtrl's released weights and renderer. PhysCtrl (NeurIPS 2025)
declares no licence for its code, weights or data. Our own contribution (the simulations, scripts and analyses) is
released under CC BY 4.0; use of the underlying objects and model outputs is subject to PhysCtrl's and Objaverse's terms.
