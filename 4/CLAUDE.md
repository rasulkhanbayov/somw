# CLAUDE.md: Simulate Only What Matters (project 4)

Read this first, then `INDEX.md` (where everything is) and `PLAN.md` (what to do next).

## The project in one paragraph

Physics-grounded video generation today either trusts a learned prior (fast, but fails silently on contacts and out-of-distribution scenes) or simulates every frame (exact, but expensive). We generate the whole 3D point trajectory once with PhysCtrl's diffusion model, then spend exact simulation only on a few short windows where it is likely wrong. Each window restarts the simulator from a state lifted out of the generated frames. The drift between generated and simulated frames, normalised by the spread of perturbed simulations, is a residual-free defect. Conformal calibration turns it into an accept/reject certificate. Rejected trajectories are repaired by re-sampling with verified windows fixed, and verified windows are replayed into the generator so fewer probes are needed over time.

**Core claim:** at matched total compute, sparse probing beats both always-generate and always-simulate, and certifies what it accepts. The source proposal is `proposal/Simulate Only What Matters — Research Proposal.md`.

## Scope decisions (do not widen without the user)

- **Base:** fork of PhysCtrl (NeurIPS 2025), cloned on the cluster at `$PHYSCTRL_ROOT`, never vendored into this folder. A reference snapshot of the files we depend on is in `notes/physctrl_ref_snapshot/`.
- **Materials:** elastic (Warp MPM) and rigid (Genesis) only. Plasticine and sand are out: their state depends on history that positions do not reveal.
- **Same simulator for data and probes.** Ground truth must come from the exact simulator the probes call, at PhysCtrl's settings. Otherwise the defect measures the gap between two simulators, not model error.
- Phases 0–2 are elastic-only. Contact events exist only in the `ood_gravity_drop` split, because PhysCtrl's elastic data has no gravity.

## Hard rules

1. **Never fabricate a number.** Anything unmeasured stays a visible `[RES]` placeholder in any paper text or table. Estimates in PLAN.md are labelled as estimates.
2. **Synthetic outputs never reach paper tables.**
   - `--dry-run` runs write to `results/SYNTHETIC_*`.
   - Summarizers refuse those folders unless `--allow-synthetic` is passed.
   - Never pass `--allow-synthetic` to produce anything shown to the user as a finding.
3. **Never submit Slurm jobs.** Write scripts; the user runs them. Do not run `sbatch`, `srun` or `scancel`.
4. **Every experiment states its falsifier before it runs**, in the script's module docstring and in PLAN.md. Report the outcome whichever way it lands, and never tune a threshold after seeing the result.
5. **Code raises; it does not guess.** Missing data, unexpanded `${VARS}`, missing checkpoint keys and unstable rollouts are errors or recorded failures, never silently filled in.
6. **Keep INDEX.md current.** Whenever a file is added, removed or renamed, update INDEX.md in the same change.
7. Run `python code/selftest.py` after every edit to `code/`. Run `bash slurm/dry_run_all.sh` (with `SOMW_RESULTS` pointing outside `results/`) after changing any script's interface.

## PhysCtrl facts the code depends on

Verified by reading the repo; details are in the `code/physctrl_bridge.py` docstring.

- MPM settings: substep 1e-4 s; `int(0.0417/1e-4)` = 417 substeps per frame; 48 frames; 125³ grid over [0, 10]³. Objects are normalised to size 1 at (5, 5, 5), with 2,048 particles.
- HDF5 keys: `x v F C` (49 frames), `vol floor_height drag_point drag_mask drag_force E nu mat_type gravity`. `mat_type`: 0 elastic, 1 plasticine, 2 sand, 3 rigid.
  - **The released test set (verified Oct 10)** has 100 files with `x v F C vol floor_height(0.2) drag_point drag_mask drag_force E nu`.
  - It has **no `mat_type` and no `gravity` key**. PhysCtrl's `TrajDataset` defaults both to 0, so all 100 files are elastic, drag-only queries.
  - Phase 2 uses all 100 as in-distribution queries. There is no spare in-distribution slice left for calibration; Phase 3 must generate one.
- Generator output: 24 frames = simulator frames 1, 3, …, 47, normalised as (x − 5)/2.
  - base: 4 DDIM steps, elastic only, no gravity input.
  - large: 25 steps, 4 materials, gravity and floor conditioning.
- **Five silent-failure guards** live in `physctrl_bridge.py`. Do not remove them:
  1. strict checkpoint key matching;
  2. set both `particle_F` and `particle_F_trial` on restarts;
  3. keep the stored drag mask on restarts;
  4. remove per-substep syncs (except in `as_is` timing);
  5. advance the solver clock by the simulated frame length.

## Compute conventions

- **GPU split.**
  - **A100s:** simulation-heavy work (rollouts, OOD and data generation, lift floor, probes, fidelity sweep).
  - **H200s:** generation, fine-tuning, the online loop, video rendering.
  - Cost ratios are measured on both, and compared only within one GPU type.
- **Scripts.** `slurm/NN_<phase>_<what>_<GPU>.sh`.
  - Numbering: 00 setup; 01–06 Phases 0–1; 07–11 Phase 2; 20s Phase 3; 30s Phase 4; 40s Phase 5. `_TODO_` in a name marks a stub that exits 1.
  - Headers follow `previous_prj_slurm_files/`: partitions `gpu-A100` / `gpu-H200`; `<A100_ACCOUNT>` and `<A100_QOS>` placeholders filled by `slurm/configure.sh`; 40 GB RAM on A100 and 80 GB on H200; `-c 8`.
- Every job script sources `slurm/_common.sh <A100|H200>`. This sources `slurm/cluster.env`, activates the conda env `somw`, loads the CUDA module, and logs host, GPU, start/end time and git commits. `log_config` prints the exact config into the log. Jobs must be submitted from the project root.
- **Logs** go to `logs/<name>_%j.out|err` (arrays: `%A_%a`). Slurm does not create `logs/`, so it ships with a `.gitkeep`.
- **Panther guidelines** (`../rgcc_guidelines.txt`, Oct 15, 2026):
  - GPU jobs must keep the GPU busy. `_common.sh` logs mean utilization + peak memory to every job log (`gpu usage :` line, CSV in `logs/gpu_*.csv`); check it, and pack several shards per GPU if utilization is low.
  - CPU-only steps (error/seed-spread computation without window strategies, summarizers, scene design) never go into GPU jobs: run them on a dev server (10.4.64.25–28) or `cpu-all`.
  - No conda/Python on the login node.
  - Request wall time close to the expected runtime (≤ 7 days hard limit).
- **Big artefacts** (Oct 14, 2026, user's rule): new large files (video model, renders, big datasets) go under `$BIG_ROOT` = `/export/hbku-research/rakh90100/somw`, not the shared home. Never move data a running job or download is using; move afterwards and leave a symlink.
- **Results** go to `results/<run_name>/[shard_<i>/]`, each with a `run_meta.json` provenance record. The existing large artefacts (OOD HDF5, generations, PhysCtrl checkpoints) live under `$DATA_ROOT`; new big ones go under `$BIG_ROOT` (above).
- **Configs.** One YAML per experiment in `configs/`. Every parameter and seed is explicit, and paths use `${DATA_ROOT}`.

## Go/no-go gates (details in PLAN.md)

- **Phase 0:** replay of stored trajectories < 0.01 units at every frame, and zero missing checkpoint keys. Otherwise stop and fix the bridge.
- **Phase 1:** simulation ≥ 10× per-sample generation in some realistic regime, on both GPU types, with the simulator charged at its matched-accuracy setting.
  - If NO-GO: switch the main regime to one that passes. If none passes, re-centre on the certificate and consult the user before Phase 3.
- **Phase 2:** G2a (lift floor ≤ 25% of the defect on generated trajectories) and G2b (Spearman ≥ 0.5, beats seed spread on OOD, cost ≤ 20% of a rollout).
  - A blown-up window scores defect = +∞ (pre-registered Oct 10; see PLAN.md). Do not change this after results exist.
  - If NO-GO: follow PLAN.md's fallback list and consult the user.

## Code conventions

- Python 3.10. NumPy at the interface; torch/warp only inside `physctrl_bridge.py`.
- Each experiment script: a module docstring with purpose, outputs and falsifier; `add_common_args` (`--config --dry-run --shard --num-shards`); `start_run(...)`; rows appended as JSONL; `run.finish(...)`.
- Every script must run end-to-end under `--dry-run` on a CPU (synthetic backend in `backends.py`).
- **Simulator blow-ups (added Oct 10).** A rollout that blows up can write outside the MPM grid, and the resulting CUDA error poisons the whole process. Scripts that run the simulator therefore:
  - check `cuda_context_lost(ex)`;
  - record the unit as `crashed` (counted as unstable);
  - call `exit_cuda_lost` (exit 75).

  Their Slurm line uses `run_resumable` (`slurm/_common.sh`), which restarts them with `--resume`. Crashed units are never retried. New simulator scripts must follow the same pattern.
- **Generator batch limit.** At most 31 samples per generator call for 2,048 points (`Generator.max_batch`). Above that, the attention launch fails.
- Comments explain why, not what. Match the existing density.
