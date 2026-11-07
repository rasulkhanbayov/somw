---
license: cc-by-4.0
pretty_name: "Simulate Only What Matters: results, logs and manifests"
tags: [physics, video-generation, simulation, conformal-prediction, reproducibility]
---
# Simulate Only What Matters: results, logs and manifests

Every result folder, every Slurm log and the checksum manifests of the project
*Simulate Only What Matters: Certified Sparse Simulator Probing for Physics-Grounded Video Generation*
(frozen Nov 3, 2026). The code is in the GitHub repository; its `freeze/README.md` explains the whole archive.

- `results/<run>/`: one folder per run (rows as JSONL, `SUMMARY.md` verdicts, `run_meta.json` with the exact command,
  config, host, GPU and commits). Every number in the plan, paper and decks comes from one of these folders.
- `logs/`: Slurm job logs (timings, GPU utilisation, CUDA-loss restarts).
- `manifests/`: path, size and sha256 of every archived file, in this repo and in the data / checkpoint repos, and of
  every third-party file the results depend on (pinned in `third_party/sources.tsv`).
- `env/`: the exact conda and pip environments (`somw`, `somw-fem`).
- `tools/polyfem/PolyFEM_bin`: the PolyFEM binary used in Phase 12 (commit 008a1aea; MIT licence).

Pre-registered experiments, negative results and exploratory analyses are all included and labelled in `PLAN.md`.
Licence: CC BY 4.0 for this repository's own content (change before publishing if you prefer another).
