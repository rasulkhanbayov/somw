# CLAUDE.md: workspace log (cvpr_physics_plausibility)

Read this first, then `INDEX.md`. The project itself lives in `4/` and has its own, more detailed `4/CLAUDE.md`, `4/INDEX.md` and `4/PLAN.md`. Those are authoritative for code, rules and experiments. This file tracks the workspace, the deadline and progress.

## What this workspace is

- **Paper:** *Simulate Only What Matters: Certified Sparse Simulator Probing for Physics-Grounded Video Generation*, for CVPR 2027.
- **Idea:** generate a 3D point trajectory once with PhysCtrl's diffusion model, then run the exact simulator only on a few short windows. These windows verify the trajectory (a residual-free, conformally calibrated defect), repair it, and feed back into the model.
- **Source:** the proposal in the root (`Simulate Only What Matters — Research Proposal.md`, an identical copy is in `4/proposal/`).
- **Deadlines:** registration **Nov 10, 2026**; paper **Nov 16**; supplementary **Nov 23**.
- Folders 1/–3/ (unrelated CVPR drafts) were removed by the user on Oct 10.
- `previous_prj_slurm_files/` holds Slurm headers from an older project ("sgf"). It is a template only.
- `data/` (created by setup) is `DATA_ROOT`: PhysCtrl checkout, checkpoints, test set, OOD data, generations.

## Compute (verified Oct 10, 2026)

- **This machine (`crimv3mgpu015`)** has 1× Tesla P100 16 GB, already used by other users' processes.
  - No Slurm client, and conda is not on PATH (`~/miniconda3` exists, but has no `somw` env yet).
  - It is one of the designated **setup hosts**: it has internet and its home is shared with the cluster. **Do not run experiments here.**
- **Experiments run on the Panther Slurm cluster:**
  - 4× A100 (`-p gpu-A100`) and 4× H200 (`-p gpu-H200`).
  - Submit from the login node. Home (`/export/home/rakh90100`, NFS) is shared, so this folder is visible there.
  - Account/QOS (from `sacctmgr`, Oct 10):
    - A100: `a100_hbku` / `a100_hbku_qos`.
    - H200: `h200_hbku` / `h200_hbku_qos`.
    - (`qcri` / `gpulimit` also exists; it is not used.)
  - **The user submits all Slurm jobs and runs setup.** Claude prepares scripts and reads results.
- **GPU split** (from `4/CLAUDE.md`):
  - A100: simulation (Warp MPM rollouts, OOD/data generation, lift floor, probes).
  - H200: generation, fine-tuning, online loop, video rendering.

## Status board

| Item | Status | Date | Notes |
|---|---|---|---|
| Phase 0–2 code (`4/code/`) | written, untested on GPU | — | |
| CPU self-test (`4/code/selftest.py`) | **30/30 pass** | Oct 15 | adds the Phase 4l scene checks |
| Synthetic dry run (`4/slurm/dry_run_all.sh`) | **passes end to end** | Oct 10 | output kept outside `results/`; its verdicts are synthetic and meaningless |
| `4/slurm/cluster.env` | **filled** | Oct 10 | `PROJECT_ROOT=…/4`, `DATA_ROOT=…/cvpr_physics_plausibility/data` |
| Slurm headers account/QOS | **filled** (16 scripts, partition↔account checked) | Oct 10 | |
| `4/slurm/preflight.sh` | new | Oct 10 | run on the login node before each submit |
| Env setup (`4/slurm/00_setup_env.sh`) | **done** | Oct 10 | env `somw`: torch 2.5.1+cu121, numpy 1.26.1, warp 1.5.0, taichi 1.7.3; checkpoints base 70M / large 413M; test set 100 .h5. Preflight OK on crimv3mgpu015 (sbatch check pending on the login node) |
| Phase 0 (replay, checkpoints) | **PASS** | Oct 10 | replay error about 2e-6; keys OK; matches PhysCtrl `eval.py` within ±3%. Details in `4/PLAN.md` |
| Phase 1 GO/NO-GO (cost) | **GO by rule, narrow** | Oct 10 | only `dense10x`+base passes (22–27×). At matched accuracy in the main regime, sim/gen = 2.3–2.8× (base) and 0.1–0.2× (large). Caveats in `4/PLAN.md` |
| Phase 2 GO/NO-GO (verifier) | v1 **NO-GO** (lift defect); **v2 GO** (`lift_noF`) | Oct 11 | Passes through **base** only: window defect Spearman 0.80–0.86 and beats seed spread on OOD by about +0.75. Large: windows lose to seed spread. Placement and normalisation add nothing; probe cost ≈ one matched-accuracy full rollout. See `4/PLAN.md` |
| Phase 2c scan | **complete**: A FAIL, A′ PASS, B FAIL (m = 2; m = 4 positive), C PASS, D PASS | Oct 11 | the cheap CFL-adaptive probe keeps the signal at 1/8 cost; longer windows help; event placement helps large on contacts |
| Phase 4a certificate | **primary PASS** (P1 weakly: false-accept 0.19 vs α 0.1; P2 +0.20) | Oct 11 | gravity-drop failures accepted: 0/50 by the window certificate, 50/50 by seed spread. Next: object-level conformal, more calibration draws |
| Phase 4a extension (object-level) | **P1′ PASS** (false-accept 0.084); **P2′ FAIL** (in-distribution power not robust) | Oct 11 | Under shift: 0/50 gravity-drop, 0/32 E_low, 0/36 force failures accepted |
| Phase 4b repair | **primary PASS, small effect** (R1 − R2 = +0.0024, 1%) | Oct 11 | x₀ replacement propagates weakly; next: conditioning repair / guidance |
| Phase 4c conditioning repair | **FAIL** (error 0.267 → 0.511; Goodhart: held-out defect fell) | Oct 12 | the defect is a certificate, not an objective; next: certify-or-simulate or anchored repair (user choice) |
| Phase 4d certify-or-simulate | **T1 PASS** (verification beats random fallback, +0.032); the cheap fallback is inaccurate OOD (gravity drop median 0.103) | Oct 12 | exploratory: with a reference fallback, error 0.011 at 0.84× the cost of always-reference (0.65× in distribution) |
| Phase 4e anchored repair | **FAIL** (L = 8: −0.012 vs replace; no extrapolation gain) | Oct 12 | training-free generative repair not supported (4b/4c/4e); remaining route: frame-conditioned LoRA |
| Phase 4f confirmation (contribution 3) | **F1/F2/F3 PASS** | Oct 12 | held out: risk 0.023, cost 0.56× always-reference, beats random fallback |
| Phase 4g frame-conditioned repair | **FAIL** (forgetting: unconstrained FC is 47% worse in distribution; conditioning itself was learned) | Oct 13 | — |
| Phase 4g-v2 (no-forgetting repair) | **FAIL** (non-degradation PASS +0.0046; no propagation, R1 − R7 = −0.0002) | Oct 13 | diagnosis: windows from generated states carry only 20% corrective information (anchor 94%); next: 4h certified-prefix suffix re-simulation |
| Phase 4h certified-prefix repair | **FAIL** (cost PASS 0.71× vs 0.84×; risk 0.070 vs 0.013) | Oct 13 | early restarts lose stress (F = I); late restarts start from drifted positions. Windows certify consistency, not accuracy. superseded by 4i |
| Phase 4i full-state suffix restarts | **V2 PASS** (risk 0.017 vs 0.013; cost 0.74× vs 0.84×); V1 FAIL (risk 0.057) | Oct 13 | first working repair: certified prefix + suffix from a coarse-shadow state. Not yet confirmed on held-out queries |
| Phase 4j V2 confirmation + ablation | **NOT CONFIRMED**: J1 risk PASS (0.032), J2 cost FAIL at the margin (0.888× cert_ref, 97.5th pct 1.034), J3 timing PASS (+0.003 error) | Oct 14 | V2's 6-window accept path costs more than cert_ref's 3 when λ̂ is high; candidate fix (cert-gated chain) needs fresh queries |
| Phase 4k gated V2 on fresh queries | **CONFIRMED** (K1 risk 0.025 vs 0.011; K2 saving +0.54 s/query at τ, +0.26 at τ_hi; K3 timing +0.002) | Oct 14 | cost 0.53× vs 0.65× always-reference; in distribution only. Exploratory OOD (Phase 2c): risk ≤ 0.04 in every split; coarse-only fails (0.26–0.52) |
| Phase 4m fresh shift + strongest baselines | **M1 PASS, M2 PASS; XF does not dominate** | Oct 15 | under shift: gated V2 risk 0.015 at 0.94× vs coarse-only 0.374; certificate FAR 0.012 vs seed spread +0.34; XF lower risk but 0.48–1.0 s/query costlier; uniform windows costlier at similar risk |
| Phase 4l multi-object contact benchmark (item 3) | **H0–H3 FAIL** (contact placement/timing not supported); certificate FAR 0/188 on contact failures | Oct 15 | diagnosis: coarse grid (dx 0.156) cannot resolve contact (gap 0.16); next idea event-aware *fidelity*; contribution-2 decision pending with the user |
| Phase 4l-ref (H0 follow-up) | **H0 PASS** (floor at contact 0.047 → 0.010); H1–H3 still FAIL | Oct 16 | the coarse grid caused the check noise; event-aware *placement* is not supported. Exploratory event-aware-fidelity repair: risk 0.012 at 1.04× vs cert-or-sim 0.000 at 1.19×. Next: pre-register EAF on fresh scenes (4n). GPU 95–97% with 2 shards/GPU |
| Phase 4n event-aware fidelity repair (fresh scenes) | **NOT CONFIRMED**: N1 risk FAIL by 0.004 (0.034 vs 0.010 + 0.02); N2 cost PASS (+0.69 s); N3 timing PASS (beats 4k) | Oct 16 | failures: the switch time is predicted from a wrong generation. The 4k chain repair stays within the risk margin on contact (0.026 at 0.99× vs 1.16×). Candidate: switch time from the coarse shadow itself |
| Phase 4o shadow-timed fidelity switch (fresh scenes) | **CONFIRMED** (O1 risk 0.012 vs 0.000; O2 saving +0.65 s; O3 beats 4k timing by 0.016) | Oct 16 | contribution 2 (event-aware *fidelity*) confirmed; caveat: on contact scenes every certified pipeline costs ≥ 1× always-simulate (SF 1.05×) |
| Phase 6a qualitative videos | **done**: 10/10 cases, 30 renders | Oct 15 | as-given cases all accepted; all 3×-force cases rejected and repaired (error ↓ 1.6–22×); figure case giraffeX ×3 (side-by-side in `$BIG_ROOT/renders/_figures/`) |
| Phase 5a audited amortisation (CPU) | **CONFIRMED, small** (A1 probe rate 0.99 → 0.85; A2 risk +0.000; A3 saving 0.044 s/query, 1.3%) | Oct 16 | drift caveat: if the stream shifts, risk on the shifted part rises to 0.048 (vs 0.008); guarantee holds for stationary streams only |
| Results deck (`JcoKGKL4fA9avdYv5Fx2b6`) | **18 slides** (v18): side-by-side slide now Phase 9 (10 methods incl. tuned simulators), limits and next updated | Oct 19 | job 76 (426640) reproduced the stored errors to ≤ 4e-7 |
| Phase 7a reviewer metrics + baselines (CPU) | **Q1, Q2, Q4 PASS; Q3 FAIL** | Oct 17 | window score best ranker (AUROC 0.977 vs residual 0.886, seed spread 0.671); momentum-residual gate as safe as ours under shift (0.018 vs 0.015) but 0.79× vs 0.44× in distribution → claim = cheapest safe pipeline; oracle best-of-8 risk 0.676 under shift |
| Phase 7b PhysCtrl metrics on its test set | **done** (job 426931; descriptive) | Oct 17 | ours: MSE 1.49e-3 → 4.23e-4, DeformLoss 0.196 → 0.135 (GT 0.074) at 0.52×; cert-or-sim 3.15e-4 / 0.103 at 0.66×; risk 0.03 both |
| Phase 7c targets recalibrated (CPU) | **S2 PASS; S1, S3 FAIL** | Oct 17 | under shift ours is the cheapest pipeline meeting the target in all 6 cells with ε ≤ 0.10; in distribution at strict ε = 0.05 ours misses it (0.125 vs α 0.10; cert-or-sim 0.085), likely the coarse-shadow repair's error floor |
| Phase 7d contact-benchmark baselines | **D1, D2, D3 PASS; XF does not dominate** (job 427021) | Oct 18 | on contact, SF 0.012 vs coarse only 0.318, seed spread 0.086 (1.66×), residual gate 0.290 (misses internal contact, as predicted); XF rejects all (1.22×) |
| Pilot 8a fidelity ladder (exploratory) | **rule (a) not met; THREAT** (job 427074) | Oct 18 | always-simulate M1 (dt≤8e-4, grid 96): risk 0.000 at 0.13× in distribution and 0.020 at 0.13× on contact (dev) → beats our efficiency there; under shift medium fails (0.055–0.085 pooled, drop 0.22). Medium suffix cuts our cost 25–55% (0.30× ID, 0.72× shift). Reframing needs the user's decision |
| Pilot 8b fused certificate (exploratory, CPU) | **NO-GO** | Oct 17 | good acceptance +0.05 (rule +0.10) but held-out false-accept 0.35 (only 16 bad calibration objects); window-only certificate stays |
| Pilot 8d certified cheap simulator (exploratory) | **NOT PROMISING** (job 80) | Oct 18 | CM1: ID 0.000 at 0.39×, contact 0.008 at 0.56× (rules met) but shift 0.065 at 0.70× (rule ≤ 0.02 at ≤ 0.50× failed); leak = late repairs (force ×3 drift passes early windows) and drop restarts from a contact-mishandled M1 state |
| Pilot E1 stiff regime (exploratory) | **NOT PROMISING, narrow** (jobs 81–83) | Oct 19 | E ≥ 1e8: ours 0.037 at 0.54× S* (M3, 8.65 s); one-window 0.38×; certificate 0/8 false accepts — all 4 failures are M2-suffix repairs; latency estimate 0.98 s vs S* 8.65 s |
| Pilot E1b stiff regime round 2 (exploratory) | **NOT PROMISING** (jobs 84–86); piloting stops | Oct 19 | ours 0.059 at 0.37× S*; repairs fixed (0/21 bad) but the certificate accepted 6/8 bad generations at E ≥ 1e8 (E1+E1b: 6/16) → certificate leaks under stiffness shift; C3 = honest cost-of-safety accounting |
| Pilot E2 cheaper verification (exploratory, CPU) | **PROMISING** | Oct 18 | one event window: AUROC 0.955 vs 0.958, same shift FAR 0.008, probe cost 0.148 s vs 0.443 s (−67%) |
| Phase 9 confirmation of ours-v2 | **CONFIRMED** (C1–C5, CC1–CC3 all PASS; jobs 428997–429003) | Oct 19 | fresh data. ID: risk 0.020 at 0.26× (always M1 0.015 at 0.13× is cheaper). Shift: 0.013 at 0.71×, the only pipeline ≤ 0.03 below 0.87× (tuned M1/M2/M3 0.10–0.16). Contact: SF 0.012 at 0.90× (always M3 0.000 at 0.50× is cheaper) |
| Phase 10 (reference convergence; PhysCtrl metrics) | **10a: reference NOT converged except stiff (R1 FAIL 6/7 sets, R2 FAIL, R3 PASS); 10b done** | Oct 21 | reference moves > ε for 21% softer / 18% stronger push / 7% drop queries; tuned-simulator shift failures mostly inside that uncertainty (6/40 robust), PhysCtrl's robust (173/266). 10b: ours vIoU 86.8% vs 77.6%, risk 0.03 vs 0.22 at 0.24×. Claim revision needs the user |
| Phase 10c exact re-scoring vs three reference time steps | **T1 PASS, T2 FAIL, T3 M1/M3 PASS, M2 FAIL** (job 101) | Oct 21 | PhysCtrl worse than ours everywhere; ours ≈ always-simulate at dt against every reference (max gap +0.025, exploratory) at 0.26–0.90×; under shift vs dt/4 ours 0.216 = reference(dt) 0.226 = M2 0.226 (M2 0.29×). Paper reframed |
| Phase 11 video-level evaluation | **judge: V0 FAIL → V1 uninformative** (job 431265); **human study page live** (no responses yet) | Oct 21 | Qwen2-VL-7B near-constant scores (3 distinct triples / 174). Study: https://claude.ai/artifact/EG8oUGHPCvmqrkJbkbKF8c, share with ≥ 10 lab members as Contributors; analysis once at close or Nov 7 |
| Phase 10d strand resolution study (audit R2) | **PARTLY NUMERICAL** (job 431955) | Oct 22 | stretch resolution-stable (median 2.95 at R0 and finest R4); left-behind share time-step sensitive (Δ 0.113 at dt/4 > 0.10). Strands are real for this MPM; the issue is the material/force parameters |
| Phase 12 independent-simulator cross-check | **S0 run 1 FAIL** (load bug, fixed); **S0 rerun + test runs ready** (`114_p12_test_cpu.sh`, 500 queries; verdict `p12_summary.py`, sandbox-tested) | Oct 29 | drop not run (no floor contact in the runner), recorded before data; force family compared before the wall |
| Phase 13 real-world anchor (PokeFlex drops) | **access granted** (link in `PokeFlex/url.txt`); sizes read: drops 25.9 GB, pokes 562.5 GB (588 GB); download script ready, **not started** (waits for the user's go) | Oct 29 | `4/slurm/00d_download_pokeflex.sh format\|drops\|pokes` on a dev server; share has 1,020 GB free (Oct 29) |
| Phase 14 simulator distillation (Option A) | **14a KILL** (all fine-tunes worse: shift risk 0.61 → 0.80–0.86, ID error 0.075 → 0.13–0.15); **pilot 14a-v2 pre-registered**, ready: `bash slurm/submit_phase14a2.sh` (wall-free pool 754, arms 424, overlap 56%) | Oct 24 | diagnosis: 84% of ACT targets were wall impacts, and replay alone did not stop forgetting; v2 = wall-free pool + 4g-v2 distillation, scored before the wall (≈ 7 GPU-h); if it kills, A stops |
| Phase 10e domain wall (exploratory, CPU) | **found**: PhysCtrl's MPM has sticky walls 0.2 from the domain edge; 99% of `ood3_force` and 32% of `ood3_E_low` references hit them (ID 4%) | Oct 24 | PhysCtrl's failures are robust to it (wall-free shift 0.53); tuned-simulator shift failures are mostly wall impacts (force M1 0.20 → 0.01 before the wall); ours' margin over M2 under shift ≈ 0.04, mostly drop. Paper wording needs the user |
| Phase 15 contact-aware generator (Option B) | **15a code written and tested; scenes designed** (train 1,439 valid, dev 187); ready: `bash slurm/submit_phase15a.sh` | Oct 24 | A pushed into B at rest; 1,024 FPS points per object; object-ID embedding; full fine-tune 6k steps; kill rule vs composed PhysCtrl (≤ 0.7× contact error, controls within +0.01); ≈ 6–8 GPU-h |
| Project freeze (`freeze/`) | **COMPLETE**: GitHub `somw` (public) + `somw-private` (private) pushed; 3 public HF repos match the manifests (`freeze/RELEASE.md`); deleted ours + scratch + renderer tiers, 350.6 GB (`freeze/deleted_local.log`); physctrl tier (5 GB) kept | Nov 8 | verification 46/50 identical, 4 traced, no reported number changes; restore: `freeze/COMMANDS.md` "Restore and reproduce" |
| Phase 3, 5 code | contact benchmark done (4l–4o); amortisation done (5a) | — | — |
| Paper (`4/paper/`) | **reframed after Phase 10c**: Table 1 max-gap columns, teaser (b), new Table 3 (three reference steps), abstract/intro/results/conclusion restated; body fits 8 pages | Oct 21 | open: Phase 11 video evaluation, preference-study page |

Update this table whenever something changes. Put real numbers in `4/PLAN.md` / `4/results/`, never here from memory.

## How to run (user)

1. **Setup.** On `crimv3mgpu015` (or shared server 10.4.64.27/.28), not on the login node:
   `cd ~/cvpr_physics_plausibility/4 && bash slurm/00_setup_env.sh`
2. **Preflight, twice:** on a dev server (10.4.64.25–28; `crimv3mgpu015` = 10.4.64.27) for the environment checks, and on the login node for the Slurm checks (it skips conda/Python there, which the Panther guidelines forbid). Both must print `PREFLIGHT OK`.
3. **Phases 0–1:** `bash slurm/submit_phase0_1.sh` (or `submit_phase1.sh` to redo only Phase 1). When everything has finished, tell Claude: it runs the summarizer on crimv3mgpu015. **Never run summarizers on panther-login**, which kills `conda activate` (seen Oct 10).
4. **Phase 2** (after a Phase 1 GO): `bash slurm/submit_phase2.sh` on panther-login. Claude runs `11_p2_summarize.sh` here.

## Decisions and open questions

- **Decided (Oct 10):** the user submits Slurm jobs and runs setup. Rule 3 in `4/CLAUDE.md` stays.
- **Decided (Oct 10):** `DATA_ROOT` is `cvpr_physics_plausibility/data/`, on the shared NFS home (563 GB free on Oct 10).
- **Decided (Oct 10):** keep the existing `4/` plan and code; add the deadline schedule to `4/PLAN.md`.
- **Decided (Oct 10):** `PYTHONNOUSERSITE=1` is set in `cluster.env`. `~/.local` (~270 packages, incl. torch 2.5.1 and numpy 2.2.6) had leaked into the env, and setup skipped installing torch because of it.
- **Decided (Oct 10):** the main `somw` env leaves out `genesis-world`.
  - Its pins (pydantic 2.7.1, numpy 1.26.4, moviepy ≥ 2) cannot coexist with PhysCtrl's (2.8.2, 1.26.1, 1.0.3, the last needed by the renderer).
  - Genesis is only used for rigid data (Phase 3/5b). Give it a separate env (`somw-rigid`) when that starts.
  - torch is pinned with a constraints file. A pip `--dry-run` resolve of the result succeeded.
- **Verified (Oct 10):** the PhysCtrl clone (commit `eaebd53`) is byte-identical to `4/notes/physctrl_ref_snapshot/` (17 files).
- **Decided (Oct 10, user, pre-registered before Phase 2):** a probe window whose rollout blows up scores defect = +∞, an automatic reject. Excluding blow-ups is reported only as a secondary column. See `4/PLAN.md`.
- **Decided (Oct 11, user):** keep the original novelty and all three contributions; do not reframe. Improve the method and the evaluation instead (`4/notes/improvement_analysis_2026-10-11.md`).
- **Decided (Oct 14, user):** large new artefacts (the video model, renders, new big datasets) go to `/export/hbku-research/rakh90100/somw` (`$BIG_ROOT` in `4/slurm/cluster.env`; 4.2 TB free), not the shared home (153 GB free). Existing data stays in `DATA_ROOT`. Every job logs whether `$BIG_ROOT` is mounted on its node. The same folder holds another project's `fwf_*` data; leave it alone.
- **Panther guidelines (`rgcc_guidelines.txt`, checked Oct 15):**
  - No conda/Python/development on the login node; development and analysis go on the shared servers 10.4.64.25–28 (`crimv3mgpu015` is 10.4.64.27).
  - No CPU-only work on GPU nodes (use `cpu-all` or a dev server).
  - GPUs must be used actively (every job now logs its mean GPU utilization and peak memory).
  - ≤ 25 cores per GPU, proportional memory, ≤ 7 days, one node, HTTPS downloads only, big data on `/export/hbku-research`.
  - Never use the h200 and H200_16GPUs accounts at the same time (we only use `h200_hbku`).
- **Decided (Oct 15, user):** contribution 2 may be reframed from event-aware *placement* to **event-aware fidelity** if the evidence supports it. First the pre-registered 4l H0 follow-up (the reference-setting scan) decides whether that route is viable.
- **Settled (Oct 16):** contribution 2 = **event-aware fidelity**: certificate-gated repair whose switch to the reference simulator is timed by the cheap simulator's own state (Phase 4o confirmed). The event-aware *placement* idea is reported as negative (4l, 4l-ref).
- **Open:** Phase 3–5 code is not written. Start it in parallel with the Phase 0–2 runs (see the `4/PLAN.md` schedule).

## Log

- **2026-10-10:**
  - Surveyed the workspace and confirmed that `4/` is the project and 1–3 are unrelated.
  - Found the machine is a P100 setup host, not the GPU nodes.
  - Self-test 16/16 and dry run pass.
  - Wrote this file and `INDEX.md`, and added the dated schedule to `4/PLAN.md`.
  - No GPU job submitted.
- **2026-10-10 (later):**
  - The user removed folders 1–3 and supplied account/QOS values.
  - Filled `cluster.env` and all headers.
  - Guarded `conda activate` against `set -u` in `_common.sh`, `06`, `11` and `00`.
  - Added `slurm/preflight.sh`.
  - Still no job submitted.
- **2026-10-10 (setup failure):**
  - The user's first `00_setup_env.sh` failed: ResolutionImpossible in PhysCtrl's requirements.
  - Fixed: dropped genesis-world, added a torch constraints file, set `PYTHONNOUSERSITE=1`.
  - Added a `~/.local` leak check to setup and preflight.
- **2026-10-10 (setup done):**
  - The user's second setup run succeeded; preflight passes, and the env loads nothing from `~/.local`.
  - Found that the test set lacks `mat_type`/`gravity`. Our code's default 0 matches PhysCtrl's own loader; this is documented in `4/CLAUDE.md` and in code comments.
  - Ready for Phase 0–1 submission.
- **2026-10-10 (Phase 0–1 run, jobs 419225–419229):**
  - **Phase 0 PASS** on both GPU types, and the bridge matches PhysCtrl's `eval.py`.
  - **Phase 1 invalid**, for three reasons:
    1. Generator batch 32 is 65,536 attention sequences, one over the CUDA limit, so the batch sizes are now 1/8/31 and the bridge raises a clear error.
    2. Coarse-substep MPM blow-ups corrupted the CUDA context. Now handled with exit 75, a `crashed` row and `run_resumable` restarts, also in p2_build_ood, p2a and p2b_probe.
    3. The job logs said "exit 0" for failed runs, because the trap read `$?` after `$(date)`. Fixed.
  - Self-test 18/18, the dry run passes, and resume and injected-crash tests pass.
  - Moved the old run to `4/results/_superseded/` and added `slurm/submit_phase1.sh`.
- **2026-10-10 (Phase 1 rerun, jobs 419234–419236):**
  - All jobs completed; the fidelity sweep went through 12 CUDA-loss restarts, all at dt ≥ 8e-4.
  - The summarizer was killed on panther-login (it kills `conda activate`), so Claude ran it on crimv3mgpu015.
  - **Verdict: GO by the rule, but narrow.** Only `dense10x`+base passes. At matched accuracy in the main regime, simulation is cheap (2.3–2.8× base, 0.1–0.2× large). Details in `4/PLAN.md`.
- **2026-10-10 (Phase 2 prep):**
  - The user chose the blow-up rule: defect = +∞.
  - Implemented `blown` flags in the verifier, p2a and p2b, and `apply_blowup_rule` in `p2_summarize`. Statistics keep +∞ and drop only NaN.
  - Found and fixed a silent bias: `aggregate()` and the JSON writer dropped non-finite windows.
  - Self-test 19/19.
- **2026-10-10 (Phase 2 run, jobs 419290–419296):**
  - All jobs completed, with 43 CUDA-loss restarts.
  - The summarizer first crashed on crashed lift rows (a missing `sigma` key); fixed with `.get()`.
  - **Verdict: NO-GO.** The cause is the F-lifting step, which blows up the simulator even on ground truth; `lift_noF` and the oracle never blow up.
  - Large-model seed spread predicts error well (Spearman 0.82).
  - Proposed a Phase 2b-v2 rerun with `lift_noF`, awaiting the user's decision.
- **2026-10-10 (Phase 2b-v2 prep):**
  - The user approved the `lift_noF` rerun.
  - Added `configs/p2b_v2.yaml`, `configs/p2_summary_v2.yaml`, `slurm/12_*` and `slurm/13_*`. Run names and the G2a floor variant are now config keys.
  - Self-test 19/19; the v2 path was dry-run on synthetic data.
- **2026-10-11 (Phase 2-v2, job 419387):**
  - All 600 rows, 0 CUDA restarts. **GO** under the pre-registered rules, through the base model.
  - Window defects predict true error (Spearman about 0.8) and beat base seed spread on OOD by about +0.75.
  - Caveats: large is not helped; placement and chaos normalisation add nothing; probes cost about as much as a matched-accuracy full simulation.
- **2026-10-11 (diagnosis and improvement plan):**
  - The user asked to keep all three contributions.
  - `diag_p2.py` showed:
    - drag-only errors are global;
    - gravity-drop errors are localised at contact, but the event detector missed 100% of base contacts;
    - the pooled advantage is cross-split scale (the certificate's strength);
    - large's errors drift smoothly and affinely.
  - Implemented the physics-prior contact predictor (`event_union`), the exhaustive scan (`p2c_scan`) with pre-registered verdicts A–D (`p2c_summarize`), AUROC, and the Phase 4a calibration set (`id_new_*`, 400 queries, split by object).
  - Self-test 20/20; the full dry run and the 2c/4a dry runs pass.
- **2026-10-11 (Phase 2c run 1 + Phase 4a data, jobs 419393–419396):**
  - The 4a data is complete.
  - The 2c scan stopped at the 50-restart cap. The fixed coarse dt violates CFL for E = 10^7.5: crashes had CFL ≥ 1.18, clean runs ≤ 1.03.
  - Added `verifier.cfl_number` and `resolve_fidelity`, a CFL gate (+∞ without running), and a new pre-registered A′ (`coarse_cfl`: dt = min(8e-4, 0.5·dx/c)).
  - Added resume jobs 16 and 35.
  - Self-test 21/21; the resume test kept every row byte-identical and added only `coarse_cfl` rows.
  - No 2c verdict computed yet.
- **2026-10-11 (Phase 2c verdicts, signal selection, Phase 4a certificate):**
  - The resume jobs completed the scans (8,100 and 7,200 rows).
  - 2c: A FAIL, A′ PASS, B FAIL (m = 4 secondary positive), C PASS, D PASS.
  - Found that v2's G2a margin was overstated (it compared a median against a max).
  - Selected the signals on 2c by a pre-stated rule (`p2c_select.py`): base `coarse_cfl`/m4/`spread_event`; large seed spread.
  - Pre-registered and ran the 4a certificate on the held-out calibration set: primary PASS.
  - Headline: under shift, the window certificate rejects all 50 gravity-drop failures; base seed spread accepts all 50.
- **2026-10-11 (object-level certificate + repair prepared):**
  - The user approved both.
  - Pre-registered and implemented the object-level conformal certificate (`p4b_certificate_objects.py`) and 8 more calibration draws per object (jobs 36–38).
  - Implemented training-free repair (`Generator.sample_constrained`: x₀ replacement in PhysCtrl's DDIM loop, with a GPU equivalence check), `p4b_repair.py` and `p4b_summarize.py`.
  - Self-test 24/24; the repair dry run passes.
  - Not run on real data yet.
- **2026-10-11 (object-level certificate and repair results, jobs 419953–419956):**
  - Object-level certificate: valid (mean false-accept 0.084 ≤ 0.11); in-distribution power gain not robust (P2′ FAIL); under shift it rejects every gravity-drop, E_low and force failure.
  - Repair: the equivalence check was exact (0). Constrained re-sampling beats replacement by only 1% (PASS); propagation is about 1–2%.
  - Proposed stronger repairs (conditioning repair, guidance).
- **2026-10-12 (Phase 4c conditioning repair prepared):**
  - The user chose conditioning repair.
  - Added `adjust`/`shared_noise` to `Generator.sample_constrained`, `generate_conditioned` in both backends, `p4c_repair_conditioning.py`, `p4c_summarize.py`, `configs/p4c_repair.yaml` and `slurm/43_*`.
  - Pre-registered in `4/PLAN.md`.
  - Self-test 25/25; dry run passes (identity difference 0).
  - Not run on real data yet.
- **2026-10-12 (Phase 4c result, jobs 420304 and 420307):**
  - The script was submitted twice; there was no overlap per shard, and the rows are complete (600, no duplicates).
  - **FAIL:** conditioning repair nearly doubled the error. Selecting by the window defect is noise-dominated (a winner's curse); the anchor term was too small to matter.
  - Lesson recorded in `4/PLAN.md`: do not optimise against the certificate.
- **2026-10-12 (Phases 4d and 4e prepared):**
  - The user asked for both certify-or-simulate and anchored repair.
  - Wrote and pre-registered `p4d_fullsim` + `p4d_pipeline` (sandbox-tested with fake fallback rows; nothing written to `results/`) and `p4e_repair_anchored` + `p4e_summarize` (non-finite repairs score +∞).
  - Self-test 25/25; dry runs pass.
  - Not run on real data yet.
- **2026-10-12 (jobs 420738/420739 lost):**
  - The user submitted 44 and 45 from `4/slurm/`. The relative `#SBATCH -o logs/…` pointed to the non-existent `4/slurm/logs/`, and `_common.sh` rejects that submit directory, so both jobs died silently with no logs or results.
  - Fixed: all 30 job scripts now use absolute log paths, and `_common.sh` accepts a submission from `slurm/`. Simulated job starts from both directories pass.
  - Resubmission needed.
- **2026-10-12 (Phase 4d/4e results, jobs 421205–421206):**
  - 4d: verification beats random fallback (T1 PASS). The cheap `coarse_cfl` fallback is inaccurate OOD; with a reference fallback (exploratory) the pipeline reaches error 0.011 at 0.84× the reference cost.
  - 4e: FAIL; matching an exact start does not improve the generated remainder.
  - Three repair attempts agree: training-free generative repair does not work for PhysCtrl.
- **2026-10-12 (Phase 4f, pre-registered, CPU):**
  - The user set the work queue: confirm → repair → event-aware probing → amortisation, with a novelty check at each step.
  - Certify-or-simulate with a reference fallback is confirmed on held-out objects: risk 0.023, cost 0.56×, beats random (+0.053 risk).
  - Started `4/notes/paper_story.md`.
- **2026-10-12 (Phase 4g prepared):**
  - Hypothesis: training-free repair failed because PhysCtrl never learned to use clean frames.
  - Pre-registered a frame-conditioned fine-tune (zero-initialised known-frame embedding, random masks, in-distribution data on the 100 sample/ objects, 0 overlap with the evaluation objects) and its evaluation against the 4b baselines.
  - Code: `fc_model.py`, `p4g_train_fc.py`, `p4g_repair_fc.py`, `p4g_summarize.py`, `make_manifest.py`; `p2_build_ood` gained `base_manifest`.
  - Self-test 26/26; the training (incl. resume), repair and summary dry runs pass.
- **2026-10-13 (Phase 4g result, jobs 421441–421443):**
  - FAIL: R7 0.243 vs replace 0.230.
  - Cause: the full fine-tune degraded PhysCtrl (unconstrained 0.070 → 0.103 in distribution), although constraints did help relative to the fine-tuned model's own baseline.
  - Next: 4g-v2 that prevents forgetting.
- **2026-10-13 (Phase 4g-v2 prepared):**
  - Pre-registered and implemented the no-forgetting variant: frozen base, LoRA (r8) on temporal attention only, known-frame embedding, and distillation to the frozen base. The non-degradation check (|R7u − R0| ≤ 0.005 on `test_id`) is a new secondary.
  - Self-test 27/27; v2 training (incl. resume), evaluation and summary dry runs pass.
- **2026-10-13 (Phase 4g-v2 result, jobs 422192–422193):**
  - Forgetting fixed; still no propagation.
  - Diagnosis: windows restarted from generated states reduce error at their own frames by only 20%, against 94% for the true-start anchor. Verification is not correction.
  - Pre-registered Phase 4h, certified-prefix + re-simulated suffix.
- **2026-10-13 (Phase 4h prepared):**
  - Implemented certified-prefix repair (`p4h_suffix_repair.py`, `p4h_summarize.py`, `slurm/51_*`).
  - The chain decision was tested on the real scan rows; that is a code test only, and it showed the first-failure distribution (anchor fails 85/300, including all 50 gravity drops; 80 accepted; 135 suffixes). No outcome was computed.
  - Dry run and summarizer sandbox (fake rows, outside `results/`) pass.
- **2026-10-13 (4h queue):**
  - A100 job 422879 is pending (Priority), estimated start Oct 15 08:55.
  - Added the H200 copy `slurm/52_p4h_suffix_H200.sh`. Pre-registration amended before any result: suffix cost = its share of the measured A100 reference rollout (c_ref × substeps / 20,016).
- **2026-10-13 (Phase 4h result, H200 job 423151):**
  - 300/300 rows, no crashes. **FAIL:** cost saving PASS (+0.64 s/query, 0.71× vs 0.84× always-reference); risk FAIL (0.070 vs 0.013 + 0.02).
  - Exploratory diagnosis: early restarts lose the stress state (F = I floor 0.070 at k = 3, mostly `ood_E_low`); late restarts begin from positions already 0.07–0.09 off although earlier windows passed.
  - Sixth failed repair attempt. The user chose: run both full-state variants (4i), then queue item 3.
- **2026-10-13 (Phase 4i prepared):**
  - Pre-registered in `4/PLAN.md` before any code ran: V1 stable F-lift (SVD projection, stretches clamped), V2 coarse-shadow state from the true start; same 4h protocol; Bonferroni α = 0.025 per variant; blow-ups fall back to a full reference rollout.
  - F-clamp bounds computed on CPU from the 100 training objects only (fixed rule, 0.1/99.9% quantiles): [0.880, 1.353].
  - Code: `lift.project_F`, `p4i_f_bounds.py`, `p4i_state_repair.py`, `p4i_summarize.py`, `configs/p4i_state.yaml`, `slurm/53_p4i_state_H200.sh`.
  - Self-test 28/28; synthetic dry run, resume (byte-identical) and summarizer sandbox (fake rows outside `results/`) pass.
- **2026-10-13 (Phase 4i result, H200 job 423187):**
  - 135/135 rows, no crashes or blow-ups.
  - **V2 (coarse shadow) PASS:** risk 0.017 vs 0.013 + 0.02; cost saving +0.505 s/query, 97.5% CI [+0.335, +0.661] (0.74× vs 0.84× always-reference).
  - **V1 (stable F-lift) FAIL** on risk (0.057): it fixes the restart from true positions (floor 0.033 → 0.007) but not the drift of generated positions.
  - Caveat: V2 was designed from the 4h diagnosis on the same queries; it needs a held-out confirmation.
- **2026-10-13 (Phase 4j prepared):**
  - The user chose: confirm V2 on held-out queries, then queue item 3.
  - Pre-registered in `4/PLAN.md`: the 4f protocol (500 object splits, same seeds and λ̂) on the 1,200 calibration queries; J1 risk, J2 cost vs certify-or-simulate, J3 shuffled-restart ablation (does the certificate's timing matter?).
  - The GPU job runs V2 at all five restart points per query (one shadow pass + five reference suffixes), so the confirmation and the ablation share one run.
  - Code: `p4j_allk.py`, `p4j_confirm.py`, `configs/p4j_confirm.yaml`, `slurm/54_p4j_allk_H200.sh`. Real entry list = 1,200 queries (checked on CPU).
  - Dry run, resume (byte-identical) and verdict sandbox (fabricated V2 rows, outside `results/`) pass; the sandbox's cert_ref reproduces 4f exactly (risk 0.02275, cost 0.5589).
- **2026-10-14 (Phase 4j result, H200 job 423214):**
  - 1,200/1,200 queries, 6,000 restarts, no crashes; cert_ref reproduced 4f exactly.
  - **NOT CONFIRMED** by the pre-registered rule: J1 risk PASS (0.032 ≤ 0.11; +0.009 vs cert_ref), J2 cost FAIL (mean 0.888× cert_ref, but 97.5th percentile 1.034), J3 timing PASS (shuffled +0.003 error [0.001, 0.006]).
  - Exploratory: V2 is cheaper in 94.4% of splits; it loses only when λ̂ is high and cert_ref accepts most queries with 3 windows while V2 needs 6.
  - In distribution, coarse-only (no generation) has risk 0.042 at 0.12× (secondary baseline), a reviewer risk recorded in `4/notes/paper_story.md`.
- **2026-10-14 (Phase 4k prepared):**
  - The user asked to prepare the fresh-data test of gated V2 before item 3, so both can run.
  - Pre-registered in `4/PLAN.md` (one amendment before any data: K3 uses the exact permutation expectation): 800 fresh queries (id_new_12..19), fixed τ = 0.02614; K1 risk, K2 cost at τ and τ_hi = 0.04501, K3 timing; object-cluster bootstrap.
  - Code: `p4k_gated.py`; `p4j_allk.py` gained `shadow_to_end` (coarse-only from the same shadow, off by default); configs `p4k_*.yaml`; jobs 55–57 + `submit_phase4k.sh`.
  - Self-test 28/28; all five steps pass a synthetic dry run; the verdict was code-checked on already-analysed draws (id_new_4..11, sandbox, exploratory; recorded in PLAN as not the test).
- **2026-10-14 (Phase 4k result, H200 jobs 424063–424065):**
  - 800/800 fresh queries, no build failures, crashes or blown restarts.
  - **CONFIRMED:** K1 risk 0.025 vs cert_ref 0.011 (within +0.02); K2 saving +0.535 s/query [0.479, 0.592] at τ and +0.256 [0.208, 0.305] at τ_hi; K3 timing +0.002 [0.001, 0.003]. Cost 0.53× vs 0.65× always-reference.
  - Exploratory (Phase 2c rows, development queries): gated V2 keeps risk ≤ 0.04 in every OOD split; coarse-only fails under shift (0.26–0.52); under hard shift the cost approaches always-reference.
- **2026-10-14 (results deck for the professor):**
  - Built a 10-slide deck of initial results: https://claude.ai/artifact/JcoKGKL4fA9avdYv5Fx2b6 (private until the user shares it).
  - Only numbers recorded in `4/PLAN.md` / `4/results/`; exploratory numbers labelled; negative results and limits included.
  - Started scoping queue item 3 (multi-object contact benchmark) in parallel.
- **2026-10-14 (deck: qualitative slides; item 3 scoped):**
  - Added 4 slides to the deck (14 total): failure modes (ground truth vs generation, median-error query per split), the goal (risk vs cost), benchmarks (ours and the field's), and one repaired query's timeline. Figures: `4/code/fig_qualitative.py` → `4/results/fig_qualitative/`.
  - Item 3 design memo: `4/notes/item3_contact_benchmark_design_2026-10-14.md` (400 two-object scenes, ≈ 21–33 GPU-h). Awaiting the user's scope decisions; nothing pre-registered.
- **2026-10-14 (senior-review pass; the user asked Claude to decide design questions itself):**
  - Audit (`4/notes/audit_2026-10-14.md`): confirmed / exploratory / failed claims; the top objections are the coarse-simulator baseline, the exploratory shift results, and the compute-value audit (adaptive vs uniform at matched budget).
  - Literature check (`4/notes/literature_2026-10-14.md`): no paper combines our three parts; the closest threats are Hybrid Neural World Models and ANCHOR. gs_physics (cited in the proposal) could not be verified, and the deck now says so.
  - Ranked plan and item-3 decisions in `4/PLAN.md`:
    - (1) Phase 4m;
    - (2) videos;
    - (3) the contact benchmark (un-cut; sticky contact stated; composition + gt_super control; no large model; reference scan on the pilot only);
    - (4) CPU baselines;
    - (5) amortisation;
    - (6) writing.
  - Phase 4m pre-registered and implemented. A sandbox found that XF's subsample calibration degenerates to −∞ when coarse failures are rare; amended before any data (pooled rule on the whole pool), and seed-spread deferral added.
- **2026-10-14 (big-file storage):**
  - The user asked for large files to go to `/export/hbku-research`.
  - Added `BIG_ROOT=/export/hbku-research/rakh90100/somw` (checkpoints/, renders/, datasets/) to `cluster.env`; `_common.sh` logs whether it is mounted on each node.
  - The user's renderer download (`cogshader5B`, into `DATA_ROOT/checkpoints`) was still running from another host at 19:25, so nothing was moved and `00_setup_env.sh` was not edited mid-run.
  - **Then, at the user's request:** deleted the partial renderer download from home (`cogshader5B` 4.8 GB + 6.8 GB partial cache; the PhysCtrl base/large checkpoints kept). `00_setup_env.sh` now downloads the renderer into `$RENDER_CKPT_DIR` = `$BIG_ROOT/checkpoints/checkpoints` (cogshader5B, lgm_fp16, SAM). PhysCtrl takes these paths as arguments, so no symlinks are needed.
- **2026-10-15 (Phase 4m result, H200 jobs 424571–424574):**
  - 598/600 fresh queries (2 unstable ground-truth builds); no crashes; `$BIG_ROOT` mounted on every GPU node.
  - **M1 PASS:** certificate FAR under shift 0.012; seed spread +0.341 higher (accepts 99% of drop failures).
  - **M2 PASS:** gated V2 risk 0.015 under shift; coarse-only 0.374 (+0.359).
  - **M3:** XF does not dominate. It has lower risk but costs 0.48 s (shift) / 1.01 s (in distribution) more per query than gated V2.
  - **M4:** uniform windows are a costlier operating point at similar risk.
  - The shift contrast is now confirmatory.
- **2026-10-15 (Phase 4l prepared, item 3):**
  - Pre-registered in `4/PLAN.md` (H0 lift-floor gate, H1 placement capture, H2 certificate FAR, H3 event-timed repair) with the item-3 decisions. No pilot; validity rules fixed in advance.
  - **Amendment before data:** restarts at every frame 1..22, so the shuffled-timing test is computable (+≈ 8 GPU-h; ≈ 20 total).
  - **Clarification before data:** the sanity rule covers all frames for controls.
  - Code: `mo_scenes.py`, `p4l_mo_build.py`, `p4l_summarize.py`; `mo_comp_` splits; `contact`/`contact_spread` placements; per-object kNN; `object_id` I/O; `event_restart` in `p4j_allk`. Configs `p4l_*.yaml`; jobs 62–65 + `submit_phase4l.sh`.
  - Self-test 30/30. Sandbox on CPU (stand-in spring simulator with softened material and forces, real component files and generations): design → joint build → sanity → composition → errors → 3-source scan → restarts → verdict all run. The sanity rule correctly flagged the stand-in scenes. Nothing written to `results/`.
- **2026-10-15 (Panther guidelines check):**
  - The renderer is downloaded to `$RENDER_CKPT_DIR` on the share (cogshader5B 39 GB, lgm, SAM; rasterizer builds).
  - The user submitted Phase 4l.
  - Compliance check against `rgcc_guidelines.txt`:
    1. `preflight.sh` activated conda and ran Python on the login node (forbidden). Fixed: it skips those checks there.
    2. CPU-only error computation (`p2b_probe` with no window strategies) ran inside GPU jobs (38, 56, 59, and the already-submitted 65). Future jobs run it on a dev server instead.
    3. GPU utilization was never measured. `_common.sh` now logs it per job (tested with a simulated job).
    4. Wall times were requested generously (up to 48 h for ~1–2 h jobs). Future scripts get tighter limits.
  - Everything else complies: 8 cores per GPU, 40/80 GB, one node, no exclusive, HTTPS downloads, one H200 account.
- **2026-10-15 (node capacity, `sinfo` from the user):**
  - H200 nodes `crirdchpxd001–006`: 192 cores, ~2 TB, 8 × H200 141 GB each (24 cores / ~258 GB per GPU).
  - A100: `crirdcmgpu006` (128 cores, ~1 TB, 4 × A100 80 GB) and `crimv3mgpu018` (48 cores, ~1 TB, 3 × A100 80 GB).
  - Our requests (8 cores; 80 GB per H200, 40 GB per A100) are well within the per-GPU share: compliant.
  - Lower-end GPUs (V100, T4, P100, A16) also exist; if our small simulations show low utilization on H200, they may be the more guideline-friendly target (access/QOS for those partitions not checked).
- **2026-10-15 (Phase 6a videos prepared):**
  - Cases and the figure-selection rule fixed in `4/PLAN.md` before any render: 5 elastic PhysCtrl examples × force {1, 3}; generation / certified / ground truth; the main figure is the median-error rejected case.
  - `code/p6_video.py` runs PhysCtrl's own SAM → SV3D → LGM → tracking → DAS, with our query builder (drag box/force exactly as `inference.py`; MPM force = dir × coeff × base_drag_coeff), generation, reference simulation and gated-V2 certify/repair in between.
  - CPU tests:
    - the query builder works for both drag modes;
    - an empty drag box now raises and is recorded as an invalid case;
    - certify/repair runs its accept, anchor-reference and mid-trajectory-repair paths.
  - Environment bugs found by importing PhysCtrl's `inference.py` on CPU:
    - **kiui 0.3.5 ships an empty `kiui/typing.py`** (every import fails);
    - **setuptools 84 lacks `pkg_resources`**.
    - With kiui 0.3.4 + setuptools 75.8.0 (tested in a scratch overlay, env untouched) the import succeeds.
    - Pins added to `00_setup_env.sh` and to the new `00b_download_render_models.sh`, which also caches SV3D on the share.
- **2026-10-15 (videos done; 4l scenes built):**
  - Phase 6a: 10/10 cases and 30 renders. All as-given cases accepted; all 3×-force cases rejected and repaired (e.g. earphone 0.222 → 0.010). The figure case by the fixed rule is giraffeX ×3; side-by-side video and contact sheet made.
  - Phase 4l steps 1–3 done (GPU usage 32–59%). 334 valid scenes; the 47 excluded contact scenes are genuine early contacts, not setup errors (controls exact to 3e-5). Job 65 (errors, scan, restarts) is running.
- **2026-10-15 (Phase 4l verdict):**
  - 334 scenes complete, no crashes, GPU usage 60–72%.
  - **H0 FAIL** (lift floor at contact 0.047 vs 0.026 allowed); **H1 FAIL** (contact placement captures less: −0.08 vs uniform); **H2 FAIL as stated** (FAR 0.000 for every placement over 188 bad scenes, so no difference possible); **H3a FAIL** (event-gated risk 0.071 vs 0); **H3b FAIL** (chain timing beats event timing by 0.016; event beats shuffled by 0.006).
  - Exploratory diagnosis: the generation error is the ignored contact (0.022 → 0.305); defects peak 2–10 frames after contact; the coarse grid cannot resolve the contact gap, so both the check and the coarse-shadow restart are poor near contact (the restart error rises monotonically with restart time).
  - Asked the user about contribution 2 (event-aware probing) before reframing it.
- **2026-10-15 (contribution-2 decision; 4l-ref prepared):**
  - The user approved the event-aware-fidelity reframing, conditional on evidence.
  - Prepared the pre-registered H0 follow-up: reference-setting scan of the 334 scenes (`configs/p4l_scan_ref.yaml`, `slurm/67_p4l_scan_ref_H200.sh`, two shards per GPU; exit-code propagation tested).
  - τ_ref = 0.02894 computed from the calibration pool only (replicate-0 procedure; the same procedure reproduces 0.02614 for coarse) and recorded in PLAN before the run.
- **2026-10-16 (Phase 4l-ref result, job 425725):**
  - 1,002/1,002 rows, no crashes or blown windows; two shards per GPU gave 95–97% utilisation.
  - **H0 PASS** at the reference setting (floor at contact 0.010 ≤ 0.016), so the coarse grid caused the check noise. H1–H3 still FAIL: contact placement captures less, and FAR is slightly worse than uniform.
  - Exploratory (seen data): event-aware fidelity repair (restart at the reference setting before the coarse grid couples the objects, d_s = 3 coarse cells) gives risk 0.012 at 1.04× vs cert-or-sim 0.000 at 1.19×, and halves the 4k chain timing's error. On contact scenes every certified pipeline costs about as much as always-simulate.
  - Next: pre-register EAF on fresh scenes (Phase 4n).
- **2026-10-16 (Phase 4n prepared):**
  - Pre-registered event-aware fidelity (EAF) repair on fresh contact scenes (draws 8–15): N1 risk ≤ cert_ref + 0.02, N2 cost saving > 0, N3 error below the 4k chain timing.
  - The restart point k_s comes from the generation: one frame before the generated objects come within 3 coarse cells (d_s = 0.46875).
  - Code: `p4n_summarize.py` (computes true errors itself, so no CPU-only GPU step); `p4j_allk` `fidelity_restart_dist`; `_common.sh` `run_two_shards` (tested: shard mapping and failure propagation); configs `p4n_*`; jobs 68–71 + `submit_phase4n.sh`.
  - Code check on the seen 4l scenes reproduces the exploratory numbers (EAF risk 0.014 at 1.04×); recorded in PLAN as not the test. Self-test passes.
- **2026-10-16 (Phase 4n result, jobs 426353–426356):**
  - 347 valid fresh scenes (271 contact); no crashes; GPU 84–96%.
  - **NOT CONFIRMED:** EAF risk 0.034 vs cert_ref 0.010 (limit +0.02, exceeded by 0.004); cost saving +0.69 s/query PASS; error below the 4k timing PASS.
  - Secondary: the 4k chain-gated repair stays within the margin on contact scenes (0.026, 0.99× vs 1.16×).
  - Exploratory diagnosis: EAF's extra failures come from a switch time predicted from a wrong generation. Candidate fix: take the switch time from the coarse shadow (true-start simulation).
- **2026-10-16 (Phase 4o prepared; the user chose option A):**
  - Pre-registered the shadow-timed fidelity switch (SF) on fresh draws 16–23: the coarse shadow (true start) records its state every frame, and the reference restart is placed one frame before the shadow's own objects come within 3 coarse cells. Tests O1–O3 as in 4n.
  - Agreed stopping rule: if 4o fails, contribution 2 becomes certificate-gated repair (4k), and the event-aware results are reported as negative/partial.
  - Code: `p4j_allk` `shadow_switch_dist` (CPU sandbox: k_sh = 11 → restart at 10, charged the shadow only up to there); `p4n_summarize.py` gained the `sf` pipeline and `primary`/`test_prefix`/`title` settings; configs `p4o_*`; jobs 72–75 + `submit_phase4o.sh`.
- **2026-10-16 (Phase 4o result, jobs 426465–426468):**
  - 346 valid fresh scenes (270 contact); no crashes; GPU 77–95%. The 4n verdict reproduced identically after the script change.
  - **CONFIRMED:** O1 SF risk 0.012 vs 0.000 (within +0.02); O2 saving +0.649 s/query [0.519, 0.777]; O3 SF beats the 4k chain timing by 0.016 [0.011, 0.020].
  - Contribution 2 settled as event-aware fidelity.
  - Caveats: on contact scenes certified pipelines cost ≥ 1× always-simulate (SF 1.05×); 158/270 contact scenes start within 3 coarse cells and fall back to a full rollout; generation-timed EAF also stayed within the margin in this draw.
- **2026-10-16 (Phase 5a amortisation, CPU):**
  - Pre-registered and ran audited amortisation of the certificate: a learned predictor skips probes when confident, and a random 10% audit calibrates the skip rule by split conformal.
  - **CONFIRMED but small:** probe rate 0.99 → 0.85 over the stream; risk unchanged (0.011); cost −0.044 s/query (−1.3%). It never skips hard-shift queries.
  - **Drift caveat:** in an in-distribution-then-shift stream, risk on the shifted part rises to 0.048 (vs 0.008); the guarantee needs exchangeability.
  - The first run was stopped while still reading data (I/O-bound full-HDF5 reads over NFS) and rerun with a light loader; nothing had been computed. A `pkill` pattern also matched its own shell, losing one edit, which was re-applied.
- **2026-10-16 (deck: qualitative + side-by-side slides):**
  - The user asked for 2–3 qualitative slides showing how ours is better, with other methods side by side.
  - Added to the deck (v13, 16 slides): `qual_video` (PhysCtrl / ours / exact for all 5 rejected 3×-force examples, frame 24, rule fixed before viewing; `code/fig_deck_video.py`) and `sidebyside` (risk · cost of 7 methods on in distribution + shift (4m) and contact (4o), parsed from the SUMMARY files). Ours is the cheapest certified pipeline in each column; on contact, always simulating is still cheaper (1.00× vs 1.05×), stated on the slide.
  - Updated cover date, limits (contact, amortisation) and next (contribution-2 question).
  - Point-cloud slides: cases fixed by the median-error rule in `4/results/fig_qualitative/deck_selection.json` before any picture. Two need our recomputed positions: `code/fig_ours_traj.py` + `slurm/76_fig_ours_traj_H200.sh` (checks against the stored verdict errors 0.0175 / 0.0168; raises on mismatch). `code/fig_deck_points.py` then draws them (layout tested on a scratchpad stand-in only). Self-test 30/30.
- **2026-10-16 (slide 12 reworked):**
  - The user noticed that at frame 24 PhysCtrl's earphone looked better than ours.
  - Checked the whole clip: PhysCtrl keeps the earphone level (a clean pose) while the exact simulation flips it onto its side, and ours follows the simulation. Errors grow over the clip: PhysCtrl reaches 0.51 at the last frame against 0.008 for ours.
  - Replaced the 5-case grid with an earphone strip over time, with per-frame errors and the repair point (`fig_deck_video.py` → `deck_video_earphone.png`). The slide states that the earphone is the largest-error case, chosen for that reason, and lists all five cases in the speaker notes. Only the earphone exceeds 0.10 on average.
  - Deck v14.
- **2026-10-17 (deck point-cloud slides):**
  - Job 76 (426640): our trajectories recomputed for the 5 rule-selected cases; the repairs match the stored verdict errors (shift_soft 2e-7, contact_switch 4e-7). GPU usage was ~0% (one sample, a minutes-long job).
  - Added `qual_shift` (exact / PhysCtrl / coarse only / ours for softer, force, drop; ours 0.018, exact, exact) and `qual_contact` (PhysCtrl's error ≈ simulating the objects without contact; ours exact and 0.017). Deck v16, 18 slides.
- **2026-10-17 (paper started; Phases 7a/7b):**
  - The user asked to start the remaining work: extra metrics, PhysCtrl's own metrics, paper outline.
  - Pre-registered 7a (CPU metrics + ANCHOR-style momentum-residual gate + best-of-8) and 7b (PhysCtrl's MSE and DeformLoss on its test set, descriptive) in `4/PLAN.md` before any code ran.
  - Refactored the 4m verdict into `p4m_confirm.assemble()`; the 4m SUMMARY reproduced byte-identically (first attempt had a missing variable, fixed).
  - **7a:** Q1, Q2, Q4 PASS; **Q3 FAIL**. The residual gate is as safe as ours, but costlier in distribution, so the claim is narrowed to "cheapest safe pipeline". Weak spots recorded: a heavier tail; worse than coarse only at ε = 0.05 in distribution.
  - **7b** ready: `slurm/77_p7b_testset_H200.sh`. `p4j_allk` gained `save_traj_dir` / `physctrl_metrics`; the bridge gained `physctrl_deform_loss` (PhysCtrl's own class, loaded by path). Dry run passes; DeformLoss itself runs only on GPU, untested here.
  - **Paper:** CVPR author kit (2026 version, year set to 2027) in `4/paper/`; `OUTLINE.md`; method and setup drafted. 23 references verified against the arXiv API / Crossref (`main.bib`). Self-test 30/30.
- **2026-10-17 (Phase 7b result, job 426931):** 100/100 queries, GPU 80–87%, the summary's cross-checks pass. On PhysCtrl's own metrics, ours cuts the MSE 3.5× and halves the DeformLoss excess over the ground truth at 0.52× cost. Descriptive only (development data).
- **2026-10-17 (related work drafted):**
  - The user asked to cite PhysWeep (Khanbayov & Kurban, arXiv 2609.06207). It is cited in the third person, which is standard for double-blind review.
  - Drafted `4/paper/sec/2_related.tex` from the verified abstracts only. Five paragraphs: physics-grounded video, evaluation, trusting surrogates, conformal prediction, coarse–fine coupling.
  - Corrected a count in the drafts: five *generative* repairs (best +1%, 4b); the sixth negative attempt (4h) restarted the simulator from generated positions.
- **2026-10-17 (Phase 7c, CPU):**
  - Pre-registered and ran the ε × α grid with every gate recalibrated. The 4m verdict reproduced byte-identically after splitting out `calibration_pool()`, and the ε = α = 0.10 cell reproduced the 4m/7a thresholds.
  - **S2 PASS** (saving +0.59 s/query); **S1 FAIL, S3 FAIL**: at the strict ε = 0.05 in distribution, ours misses the target and is no safer than coarse only.
  - **Under shift**, ours is the cheapest pipeline meeting the target in all six cells with ε ∈ {0.05, 0.10}.
  - At ε = 0.20 the certificate cannot be calibrated for α ≤ 0.10 (too few bad calibration objects).
  - The claim is now stated per target (the paper outline and story are updated).
- **2026-10-17 (Phase 7d prepared):**
  - Pre-registered the four missing contact baselines (coarse only, XF, seed spread, residual gate) against SF, with tests D1–D3 and the XF dominance rule. Thresholds come from the single-object calibration pool, as under shift.
  - Refactored the 4n/4o verdict into `p4n_summarize.build()`/`pipes()`; both SUMMARYs reproduced byte-identically.
  - **Unit check:** on contact-scene ground truth, the momentum residual is about 1e-4 (the collision is internal). So the residual gate is expected to miss failures in which a contact was ignored. This physics prediction is recorded before the data.
  - Job 78 (coarse rollouts only, < 3 h on 4 H200) is ready for the user. Dry run of the coarse-only path passes; self-test 30/30.
- **2026-10-17 (second review pass; the user's "strong, novel paper" prompt):**
  - Audit + verified literature re-check: `4/notes/audit_literature_2026-10-17.md`.
  - **New top weakness:** Phase 1 shows a medium simulator setting (dt 4e-4, grid 96) at error 0.007 for 0.25× the reference cost in distribution. "Always simulate at the cheapest adequate setting" is the baseline to beat; it is unmeasured under shift and contact.
  - **Literature:** no direct competitor. Video-level competitors that compare against PhysCtrl: PhysFlow, NEXUS (contact-rich, closest to contribution 2), 3DPhysVideo. Reviewers will expect VideoPhy-2 AutoEval, a GPT judge, TRAJAN/FVMD, and PhysCtrl's vIoU/CD/Corr.
  - Proposed pilots P1–P4; waiting for the user's go.
- **2026-10-17 (pilots 8a/8b; the user said go):**
  - Defined both pilots in `4/PLAN.md` before any pilot code ran, as exploratory work on development data.
  - `restart()` gained `sim_overrides`, and `p4j_allk` gained `restart_fidelity`. Both are off by default; dry run: the medium suffix uses 1/4 of the reference substeps. Self-test 30/30.
  - Job 79 ready; `p8a_pilot.py` analyses it. 8b runs on CPU.
- **2026-10-18 (jobs 78/79 analysed):**
  - **7d PASS:** on contact, SF beats coarse only, seed spread and the residual gate. The residual gate misses ignored contacts, as predicted from physics before the data.
  - **Pilot 8a:**
    - A tuned cheap simulator (M1: dt ≤ 8e-4, grid 96) is nearly exact in distribution (risk 0.000 at 0.13×) and on contact development scenes (0.020 at 0.13×). That undercuts our efficiency claim there.
    - Under shift every medium setting fails silently (pooled 0.055–0.085; drop up to 0.22), so certification is still needed.
    - A medium-fidelity suffix cuts our own cost 25–55%.
  - Raised the reframing question with the user (contribution 3 would change).
- **2026-10-18 (pilot 8d prepared; the user said start):**
  - Defined in `4/PLAN.md` before any code: certify the cheap simulator M1 (novelty threshold, because M1 has almost no in-distribution failures) and repair it with a reference suffix from M1's own state.
  - **One amendment before data:** the in-distribution cost rule ≤ 0.30× was infeasible by construction (M1 0.13× + 3 M3 windows 0.25×), so it is now ≤ 0.45×. Found by the sandbox test.
  - Code: `p8d_certsim.py` (GPU, dry run passes), `p8d_pilot.py` (sandbox-tested on synthetic rows outside `results/`), job 80. Self-test 30/30.
- **2026-10-18 (pilot 8d result):**
  - Not promising by the pre-stated rule: the shift risk is 0.065.
  - False accepts are rare (1/200); the leak is in the repairs.
  - No single pipeline is cheapest-and-safe everywhere on development data. The candidate safe everywhere is 8a's B2 (gated generation, M1 shadow, M2 suffix): 0.30× in distribution, 0.72× under shift; contact not yet run.
- **2026-10-18 (contribution-3 review; the user's second prompt):**
  - Diagnosis + verified literature: `4/notes/efficiency_c3_2026-10-18.md`.
  - **Soft materials:** generation + windows ≈ M1's whole cost, so we cannot beat a tuned simulator there by construction (break-even formula of 2608.23075).
  - **Stiff materials:** explicit MPM cost grows with √E (measured: M1 0.12× → 0.40×) while generation does not. This is the regime where generate-then-certify can win. The threat there is implicit MPM (HOT, i-PhysGaussian, GeoWarp).
  - Efficiency claims must follow McGreivy & Hakim (NMI 2024).
  - Proposed pilots E1–E3; waiting for the user's go.
- **2026-10-18 (pilots E1–E3; the user said yes):**
  - Defined in `4/PLAN.md` before any code ran.
  - The backend gained `reference_cfl`, a CFL-safe reference that is off by default (no earlier phase sets it). `p2_build_ood` gained the `stiff_` split (E 10^U(7.5, 9.0)).
  - Configs `p8e_*`, jobs 81–83 + `submit_pilot_e1.sh`. The full chain dry run passes; CPU check of the draws: the stiff reference costs 1.2–2.4× and M1 0.56–1.1× of the normal reference. Self-test 30/30.
  - E2 runs on CPU; E3 is computed inside the E1 analysis.
- **2026-10-19 (pilot E1 result):**
  - Narrow miss by the rule: ours 0.037 at 0.54× S*.
  - The regime behaves as predicted: simulation is expensive, generations are accurate, 76% accepted.
  - The certificate accepted 0/8 bad generations. All failures are repairs at M2, which is inaccurate for stiff materials.
  - Proposed round 2 (E1b): fresh stiff draws, one-window certificate, M3 suffix.
- **2026-10-19 (pilot E1b prepared; the user said yes):**
  - Defined in `4/PLAN.md` before any code: fresh stiff draws, the one-window certificate at τ 0.02614, an M3 suffix. This is the last pilot round.
  - `p8e_stiff.py` gained `primary_windows`, labels and title. The E1 verdict reproduces byte-identically.
  - Jobs 84–86 + `submit_pilot_e1b.sh`; generation on 2 GPUs. The dry run passes; self-test 30/30.
- **2026-10-19 (pilot E1b result; contribution-3 piloting ends):**
  - Not promising: risk 0.059 at 0.37× S*.
  - The M3 repair fixed the repairs, but the certificate accepted 6/8 bad generations at E ≥ 1e8. That is a new limitation of the certificate under stiffness shift.
  - By the agreed stopping rule, contribution 3 becomes the honest cost-of-safety accounting.
- **2026-10-19 (Phase 9 pre-registered and prepared; the user asked for it):**
  - **Confirmation of ours-v2 on fresh data:** one-window certificate; single objects repaired with an M1 shadow + M2 suffix; contact with the shadow-timed switch + M3 suffix.
  - Baselines include the tuned simulators M1/M2/M3, the residual gate and seed spread.
  - **Tests:** C1–C5 (99% CIs, Bonferroni over 5), CC1–CC2.
  - **Code:** the builder gained `ood3_` (the ood2 draws reproduce 40/40); `p4n_summarize` gained `cert_windows`; new `p9_confirm.py` (code-tested on development data, output deleted). Configs `p9_*`/`p9c_*`, jobs 87–93 + `submit_phase9.sh`.
- **2026-10-19 (Phase 9 amended before data):**
  - A code check on seen 4o scenes showed the one-window certificate misses ignored-contact failures (SF risk 0.095).
  - Contact therefore keeps the three-window certificate.
  - C1 and CC1 gain absolute risk bounds (≤ 0.035 / ≤ 0.03), so a weakened comparator cannot carry a relative test. Self-test 30/30.
- **2026-10-19 (Phase 9 result, jobs 428997–429003):**
  - All 24 array tasks exited 0; no unstable ground truth; 600/600 single-object queries; 168 valid contact scenes (132 contact, 36 controls; 32 of 200 excluded by the 4l rule). GPU 36–60% (generation), 67–93% (simulation).
  - Probe errors and both verdicts run on crimv3mgpu015.
  - **CONFIRMED.**
    - Single objects: C1 risk 0.020 vs 0.015 (≤ 0.035); C2 0.26× [0.21, 0.31]; C3 saving +1.30 s; C4 shift risk 0.013, always M1 +0.090 [+0.055, +0.128] riskier; C5 saving +0.76 s.
    - Contact: CC1 SF 0.012 vs 0.000; CC2 saving +1.31 s; CC3 beats 4k timing by 0.014. SF costs 0.90× always-reference (was 1.05× with a reference suffix in 4o).
  - Reported as pre-stated: always M1 is cheaper in distribution (0.015 at 0.13×). New, also reported: on contact, always M3 is cheaper (0.000 at 0.50×).
  - Under shift every tuned simulator fails (M1 0.102, M2 0.102, M3 0.155). The ladder is non-monotonic on ood3_E_low/force (M3 worse than M1). Exploratory reading: step-size sensitivity of soft, strongly driven objects; the reference's own convergence is not tested there.
- **2026-10-19 (results section + deck, after Phase 9):**
  - `code/p9_table.py` (new, descriptive): object-weighted table of every pipeline on the three Phase 9 test sets → `results/p9_table/`, `paper/tab/main_table.tex`, `paper/fig/tradeoff.pdf`. `p9_confirm.py` gained `assemble()`; its SUMMARY reproduced byte-identically. Self-test 30/30.
  - Checks recorded in `4/PLAN.md`:
    - one-window validity carries over (E2 threshold 0.03007 ≥ τ);
    - the cost model is consistent across grids (per-substep time 0.23 ms A100, 0.29–0.32 ms H200 at every grid);
    - on contact M1 is just inside the bound (0.029 object-weighted);
    - on contact, chain timing has the same risk as SF at 0.67×, so SF buys accuracy, not risk.
  - Paper: method updated to the confirmed pipeline (one window for single objects, three for contact; M1 shadow + M2 suffix; M3 at the switch); setup lists the Phase 9 test sets and tuned-simulator baselines (McGreivy & Hakim still a `\CITE` placeholder); results drafted with every number tagged by phase. Builds, 7 pages.
  - Deck v18: side-by-side slide on Phase 9; cover date; limits (tuned simulator cheaper where the regime is known; certificate blind spots); next.
- **2026-10-19 (intro/abstract; paper qualitative figures prepared):**
  - Abstract and introduction written around "you cannot know the regime in advance". Every number is tagged with its phase. McGreivy & Hakim (NMI 2024) verified on Crossref and added to `main.bib`. The paper builds at 8 pages.
  - Qualitative figures redone for the Phase 9 pipeline on Phase 9 data. Cases were fixed before any picture (`fig_paper_select.py`; PLAN "Paper qualitative figures").
  - `fig_ours_traj.py` gained `--selection/--prefix`, the p9/p9c pools, M2/M3 suffixes, 'accept' and an always-M1 rollout. `p6_video.py` gained a v2 mode that never overwrites the Phase 6a records.
  - New scripts: `fig_paper_points.py` and `fig_paper_video.py`, layout-checked on stand-in data. New jobs: 94 and 95 + `submit_paper_figs.sh` (≈ 1–1.5 GPU-h). Self-test 30/30.

- **2026-10-20 (5-slide summary deck; the user asked for a short version):**
  - Built https://claude.ai/artifact/UB7gbAZ1omnnio4TwiKqZP (private), separate from the 18-slide deck. Slides: problem, idea + contributions, numbers (Phase 9 table in percentages), pictures (the Phase 6a five-video grid, labelled as the earlier pipeline), takeaways + limits.
  - Numbers are from `4/results/p9_table` and PLAN; the contact residual-gate number is labelled as coming from the earlier set (7d). The limits name the unverified convergence of the reference simulator, which the user raised the same day; the check (≈ 2–3 GPU-h) is proposed, not yet prepared.
- **2026-10-20 (jobs 94/95):**
  - Job 94 (429594) done: the five Phase 9 figure cases recomputed and checked against the stored errors (max diff 1.7e-5). `paper/fig/qual_{shift,contact}.pdf` drawn.
  - Job 95 (429595): shard 0 done (the five as-given cases, all accepted, renders copied). Shard 1 (all five 3×-force cases) FAILED on crirdchpxd006 because the node's /tmp was full, so torch.compile's gcc aborted inside DAS. Not a code bug.
  - Fix: `95_p6_video_v2_H200.sh` now puts TMPDIR and the triton/inductor caches under `$BIG_ROOT/tmp/<job>`, removed after the run (not via trap: `_common.sh` owns the EXIT trap); wall time 3 h. Resubmit only shard 1: `sbatch --array=1 slurm/95_p6_video_v2_H200.sh`.
- **2026-10-20 (Phase 10 pre-registered and prepared; the user said "yes do"):**
  - Video-level metrics: not run; stated as a limitation (Claude's recommendation, accepted).
  - **10a, reference convergence:** the reference at dt/2 on all Phase 9 queries and dt/4 on a fixed subset.
    - R1: the 95th percentile of the distance ≤ 0.02 per set.
    - R2: triangle-inequality risk bounds for every pipeline. The self-check reproduces the Phase 9 table at d = 0.
    - R3: free fall of the drop ground truth. Already **PASS**: max deviation 1.2e-4 over 78 queries. The first output had 2 runaway queries because a one-cell threshold counted resting frames; amended to the two-cell contact layer and recorded as such.
  - **Found while writing the configs:** M1–M3 frames last int(0.0417/dt)·dt = 0.0416 s, not 0.0417 s, so the tuned simulators are compared about 5 ms early by the last frame. Measured: offset error median 0.003–0.009, max 0.065. After subtracting it, the tuned settings still fail under shift (lower bounds M1 ≈ 8.8%, M2 ≈ 9.3%, M3 ≈ 13.5%). Recorded in `p10a_convergence.py`.
  - **10b:** Phase 7b re-run with the final pipeline. `p9_confirm.one_window()` was extracted; the Phase 9 SUMMARY, the p9 table and the Phase 7b SUMMARY all reproduce byte-identically.
  - Jobs 96/97 + `submit_phase10.sh`.
    - Dry runs pass. The DeformLoss step runs on GPU only, as in 7b.
    - Fixed before submission: job 97 must use array 0–3, because `run_two_shards` hard-codes 8 shards.
  - Self-test 30/30.
- **2026-10-20 (writing):**
  - Paper wording: "exact" → "reference" in every section, table and figure label. The method now states that the reference is the motion PhysCtrl imitates, not real-world motion.
  - New "Is the reference converged?" results subsection: R3 and the frame-time offset filled in; R1/R2 as `\RES` until job 96.
  - Limitations and conclusion written: scope, reference, known regimes, stiff materials, conservativeness, no video-level metrics, streams.
  - Method diagram `code/fig_method.py` → `paper/fig/method.pdf`, placed in §3.
  - PhysMAS (2609.07174) and NEXUS (2606.15015) verified on the arXiv API and cited. Correction to Claude's Oct 20 answer to the user: PhysMAS simulates every scene (agent-set MPM materials, candidates screened); it does not certify or repair generated motion.
  - Builds at 9 pages (body ends p. 8) before the qualitative figures.

- **2026-10-20 (job 95 rerun, 430493):**
  - 4 of 5 3×-force cases re-rendered with the Phase 9 pipeline: blanket 0.089 → 0.031 (repair k = 7), chair 0.061 → 0.034 (k = 15), earphone 0.222 → 0.013 (k = 3), giraffeX 0.086 → 0.018 (k = 7). GPU 31%.
  - Penguin ×3 stopped at the rebuild check: the rebuilt ground truth differed from Phase 6a by 1.0e-3, over the 1e-3 limit. GPU MPM is not bit-reproducible.
  - Fix: the limit is now 2e-3, as in `fig_ours_traj.py`, and the deviation is always logged. Finished cases are skipped, so their records match their renders. Leftover temp dir (90 MB) removed.
  - Resubmit: `sbatch --array=1 slurm/95_p6_video_v2_H200.sh`, which renders only the penguin.
- **2026-10-20 (job 95 complete; figures in the paper):**
  - Penguin ×3 rendered: 0.073 → 0.030 (repair k = 3); rebuild deviation 6.0e-4. All 10 cases done.
  - The Phase 6a figure rule on the v2 decisions again picks giraffeX ×3 (0.086 → 0.018, repair k = 7) → `paper/fig/qual_video.pdf`.
  - Paper: contact point clouds and the video figure added with text (3× cases rejected, error down 1.8–18×). The shift point clouds, amortisation and the full negative list moved to `paper/supp/moved_from_main.tex`. The body now ends on p. 8.

- **2026-10-20 (senior-AC paper review; the user's pasted prompt):**
  - Three research agents read the full texts of PhysCtrl, PSIVG, RealWonder, PhysFlow, NEXUS, Hybrid NWM and ANCHOR.
  - Review and plan: `4/notes/paper_review_2026-10-20.md`: 11 weaknesses ranked by reject risk; 12 edits with impact, effort and GPU cost.
  - ★ first:
    1. page-1 teaser (contact example + risk–cost);
    2. reliability framing + regime-agnostic main table (worst-case risk, "safe everywhere");
    3. PhysCtrl's own vIoU / Chamfer / L2 via its `volume_iou.py` on the job-97 outputs;
    4. one ablation table built from the pre-registered phases;
    5. a small video-level evaluation (VLM judge + preference study, ~30 inputs, 15–25 GPU-h), which needs the user's decisions.
  - Useful facts:
    - PhysCtrl's own Table 4 claims parity with a simulator (GPT judge), which our shift/contact data contradict.
    - NEXUS evaluates exactly our swap-the-trajectory-module protocol.
    - Neither gating paper has a guarantee or a tuned-solver comparison.
  - No paper edits yet; waiting for the user's go.
- **2026-10-20 (paper-review edits; the user said "yes do all"):**
  - **★1** page-1 teaser (`code/fig_teaser.py`): the rule-selected contact scene over time, and worst risk vs largest cost over the three sets. The old Fig. 2 was removed (absorbed).
  - **★2** main table regenerated by `p9_table.py`: rows grouped (learned / fixed simulator / gated / ours) and "all three" columns (worst risk, safe, max cost). Ours is the only "yes" under 1×. Contact cells of coarse, seed spread and the residual gate come from 7d (†). The SUMMARY is byte-identical. Abstract and intro rewritten around per-query trust, with PhysCtrl's own "on par with a simulator" claim as the hook and an explicit research question.
  - **★3** PhysCtrl's own metrics, via its `volume_iou.py` imported by path (`p7b_summarize` option; pre-registered as a 10b addition).
    - Our run reproduces PhysCtrl's Table 2: vIoU 77.57% vs 77.59%, Chamfer 0.0028, MSE 0.0015.
    - Earlier pipeline: ours 86.63% / 0.0008. The final-pipeline rows wait for job 97.
  - **★4** ablation table from the pre-registered phases. Generation-timed switch row: same-scene Phase 9 numbers (0.025 vs 0.012), not 4n vs 4o.
  - **★7** certificate figure (`fig_certificate.py`, 7a risk–coverage with a random floor and a perfect ranking).
  - Per-family cost sentence (soft 0.50×, stiff 0.24×, 3× drag 0.99×, drop 1.10×).
  - Supplementary videos (`supp_videos.py` → `$BIG_ROOT/supp_videos`, 10 cases).
  - The full-width contact figure moved to `paper/supp/` (the teaser shows its scene). Body fits 8 pages.
  - **★5 Phase 11** pre-registered and prepared:
    - images from FLUX.1-schnell (amended before any code to 1152×768 → 720×480, PhysCtrl's example size);
    - `p6_video.py` fresh mode, sharded by object;
    - Qwen2-VL-7B judge with PhysCtrl's Appendix-A prompt;
    - `p11_summary.py` (sandbox-tested on fabricated scores outside `results/`);
    - jobs 98–100, the download script `00c`, `submit_phase11.sh`.
    - Frame extraction tested on the job-95 renders. Self-test 30/30.
  - Not done: the title change (kept: "Simulate Only What Matters" already carries the idea); the failure-case figure (no space; covered by limitations); the preference-study page (needs the Phase 11 renders).
- **2026-10-21 (Phase 10 results, jobs 430617/430618):**
  - All rows present; GPU 90–95% (96), 39–67% (97).
  - **10a: the reference is not converged at our ε except for stiff materials.**
    - 95th-percentile change at half the step: ID 0.026, stiff 0.009, soft 0.20, drop 0.12, stronger push 0.43, contact 0.044.
    - A quarter step moves it further.
    - R1 FAIL (6/7 sets), R2 FAIL, R3 PASS.
  - Exploratory follow-up: under shift the tuned simulators' "failures" are mostly inside the reference's own sensitivity (6/40 robust), while PhysCtrl's failures are robust (173/266).
  - So "every tuned setting fails under shift" and the "only safe everywhere below 1×" headline are not supported for soft materials and strong pushes. Revising them weakens contribution 3, so the user is asked.
  - **10b:** ours vIoU 86.81% (PhysCtrl 77.57%), Chamfer 0.0008 (0.0028), risk 0.030 (0.220) at 0.24×.
  - Paper: Table 3 and the §5.5 facts filled in; the claim text is marked pending.
- **2026-10-21 (the user chose "reframe + extra robustness run"; Phase 10c prepared):**
  - Pre-registered in PLAN before any code: exact re-scoring of every Phase 9 pipeline against the reference at dt, dt/2 and dt/4 (positions saved); tests T1–T3 at 98.3% object-cluster CIs.
  - Code:
    - `p10c_decisions.py` (CPU, run): 732 queries; ours 194 accepted / 242 repaired / 296 simulated. `p9_confirm.assemble` now also stashes the residual and seed-spread decisions; its SUMMARY is still byte-identical.
    - `p10c_rescore.py` (GPU job 101): reproduction check ≤ 2e-3 against the stored Phase 9 errors.
    - `p10c_summary.py`: sandbox-tested on fabricated trajectories outside `results/`.
  - CPU check: all 732 entries and generations resolve. Self-test 30/30.
  - The paper reframing waits for 10c, so the restated claims use exact numbers.

- **2026-10-21 (jobs 101 / 98 / 99):**
  - **98 (430788): done.** 25 FLUX images at 720×480, all valid single objects; no curation.
  - **101 (430784): 715/732 queries saved.**
    - Shard 6 stopped at the reproduction check: one strong-push M3 rollout deviated 0.0029 against the 2e-3 limit. All others were ≤ 0.0012.
    - Amended (PLAN): stop only above 1e-2; deviations above 2e-3 are counted. Finished queries are skipped (`--resume` plus an npz check).
    - Rerun: `sbatch --array=3 slurm/101_p10c_rescore_H200.sh`.
  - **99 (430789): every task failed at the first render.** Bug in the fresh mode: `center/scale/fps_idx.npy` were not written, because the v2 guard skipped them. Fixed (written in fresh mode).
    - Rerun: `sbatch slurm/99_p11_render_H200.sh`.
    - The 4 objects already preprocessed are reused.
  - Self-test 30/30.
- **2026-10-21 (jobs 101 and 99 reruns; Phase 10c verdict; paper reframed):**
  - **101 (430885): done.** 732/732 queries; max reproduction deviation 0.0022 (one M3 above 2e-3); GPU 52%.
  - **Phase 10c:**
    - T1 PASS (PhysCtrl riskier than ours in every set, against every reference).
    - T2 FAIL: under shift ours is 0.108 / 0.216 against dt/2 / dt/4, about the same as always-simulate at dt (0.113 / 0.226).
    - T3: M1, M3 PASS; M2 FAIL. By the pre-registered rule, M2 is as safe as ours under shift within the reference's uncertainty (0.29× vs 0.71×).
    - Exploratory (`p10c_explore.py`): ours stays within +0.025 of always-simulate's risk on every set against every reference, the only pipeline below 1× to do so; robust risk under shift ours 0.005, M2 0.048, M1 0.102, M3 0.157, PhysCtrl 0.647.
  - **Paper reframed** around "tracks the simulator it replaces" (Table 1 "All three" = max gap + max cost; teaser (b); new Table 3 from `p10c_table.py`). Certificate figure, frame-offset sentence and most negatives moved to `paper/supp/` for space; body ends on p. 8.
  - **99 (430886): shards 1, 3 done; shards 0, 2 stopped** at an unstable reference (rubber duck and giraffe teether at force 0.40, log E = 4; NaN from model frame 9). Amended (PLAN): such cases are recorded invalid and excluded, as in 4m/9; `p6_video.py` now skips them. 36 cases done; 7 objects left.
    - Rerun: `sbatch --array=0,2 slurm/99_p11_render_H200.sh`.
  - Frames extracted for the 34 finished valid cases. Self-test 30/30.
- **2026-10-21 (job 99 shards 0/2 rerun, 431046):**
  - Both exit 0 (GPU 40% / 14%). The two invalid cases were skipped as designed; all 48 valid Phase 11 cases now have three renders.
  - Trajectory level (descriptive): in range 16 accepted / 9 repaired, ours 1/25 above 0.10 (accepted spatula, 0.111); shifted 23/23 rejected and fully simulated (PhysCtrl median 0.333).
  - Frames extracted for all 58 cases (174 videos). Next: the judge job 100.
- **2026-10-21 (Phase 11 judge, job 431265; study page):**
  - 174/174 videos scored and parsed; job 431397 was a second submission that found nothing left to do.
  - **V0 FAIL** (PC reference − PhysCtrl +0.07 [−0.07, +0.21] on 28 rejected shifted cases) → by the pre-registered rule **V1 is uninformative**, not a loss. Exploratory: the judge is degenerate (3 distinct score triples, SA = PC always, all three videos tied in 51/58 cases).
  - Paper: the video-evaluation limitation now states that a 7B judge could not tell reference videos from PhysCtrl's; body still ends on p. 8.
  - **Human study (V2) prepared and published:** `p11_study_build.py` (28 items, opaque file names, local mapping), `p11_study.py` (sandbox-tested on fabricated responses outside `results/`), rules fixed in PLAN before any response (analysis once at close or Nov 7; ≥ 10 raters; no exclusions). Page: https://claude.ai/artifact/EG8oUGHPCvmqrkJbkbKF8c (private; owner view shows counts only). Rater privacy checked with a probe document (deleted).
- **2026-10-22 (study page: no sign-in needed):**
  - The user's view of the study gave the page storage but no account identity, so nothing could be saved. Fixed in two steps, both recorded in PLAN as amendments before any response existed: an anonymous rater code when there is no account id; then, at the user's request, no sign-in at all: answers kept in the rater's browser and an answer code shown at the end, to be sent to the user. Signed-in raters also save to the store.
  - `p11_study.py` reads the codes (`--codes`, `--items`) with a checksum; round trip tested on codes from the page's own JavaScript (a parsing bug with "-" in partial codes found and fixed). Self-test 30/30. Still 0 responses.
- **2026-10-22 (first rater's interim look; simulator audit):**
  - At the user's request the first rater's answers were shown (logged in PLAN): Q1 PhysCtrl preferred 24/28; Q2 ours 28/28 (5/5 on repaired items).
  - The user asked whether the simulator and metric are wrong. Audit `4/notes/simulator_audit_2026-10-22.md` (code compared line by line with PhysCtrl's data generator; contact sheets of all 28 items; trajectory measurements; literature from a research agent).
  - **Findings:** no bug. The reference uses PhysCtrl's own convention (no gravity, constant push, 1 m objects). Six of 23 shifted items stretch into strands (soft + small push patch); two reference videos have renderer artifacts (bunny head on the table, mushroom landing); 83–85% of trajectory error comes after objects leave the view. Phase 9: 32% of the softer family stretch > 2×; drop is realistic and robust.
  - Recommendations R1–R5 (writing; strand convergence check ~1–2 GPU-h; realistic-regime subset on CPU; point-cloud fidelity study; realistic-settings video study ~5–10 GPU-h) await the user.
- **2026-10-22 (circularity question; Phase 12 pre-registered):**
  - The user suspected the simulator favours ours. Breakdown: ours = the reference on 7% / 52% / 57% of ID / shift / contact queries, so most of the shift and contact advantage is circular by construction; repairs re-simulate 61–75% of frames with the same code. Note: `4/notes/simulator_independence_2026-10-22.md`.
  - Research agents: no MPM is independent of PhysCtrl's (taichi_elements → warp-mpm → PhysGaussian → PhysCtrl); PolyFEM (fixed-corotated, implicit, IPC) is the best independent reference, libuipc the GPU fallback; warp XPBD ignores E/ν. Real data: PokeFlex drops are the only realistic anchor (needs registration).
  - conda `polyfempy` 0.5.2 is a 2023 build without fixed-corotated (smoke test NaN); building PolyFEM 008a1aea from source into `$BIG_ROOT/tools/polyfem` (separate env `somw-fem`; system cmake on this host is broken). Side note: `somw` has pydantic 2.13.4 (the 2.8.2 above was a pin discussion, not the installed version).
  - Phase 12 pre-registered in PLAN before any second-simulator data: mesh, load, time-step and gate rules; tests X1–X3; contact excluded, drop secondary.
  - PolyFEM built from source in 38 min; synthetic-cube smoke test: free fall and interior nodal push agree with theory after accounting for the integrator's zero initial acceleration; rhs sign convention noted. Clarifications recorded in PLAN Phase 12 before any query ran.
- **2026-10-22 (PokeFlex requested; Phase 13 pre-registered; drop correction):**
  - Found that PhysCtrl base has no gravity or floor input (`eval_base.yaml`), so its 98% drop failure is structural. Drop is a certificate result, not a generalisation failure; the audit's "rest realism on drop" advice is withdrawn (notes updated).
  - Phase 13 pre-registered before data access: E fitted on pokes only, drops predicted blind; P1 (MPM closer to reality than PhysCtrl-large), P2 (MPM within 2× the capture repeatability).
- **2026-10-22 (Phase 12 runner and gate):**
  - `code/p12_fem.py` (PolyFEM runner) and `code/p12_s0.py` (gate), job `slurm/102_p12_s0_cpu.sh` (cpu-all).
  - Amendments before any S0/test query (PLAN):
    - mesh = the MPM's occupied grid cells, because PhysCtrl's points are surface samples, so the MPM body is a one-cell shell (sum(vol) = n_cells·dx³ exactly);
    - push = box body force (the stored masks equal the 0.4 box);
    - frames 1, 3, …, 47;
    - mass-weighted centre of mass;
    - portable build (the first used `-march=native`).
  - Code trial on one development query: MPM–FEM 0.0002 (frame 1) and 0.0012 (frame 3). `--smoke` test end to end OK (scratch only). Self-test 30/30.
- **2026-10-22 (the user: "do B"; Phase 10d prepared):**
  - Pre-registered in PLAN before any code: R1/R2 (dt/2, dt/4) and R3/R4 (filled body, grid 125/250). Decision rule on the 6 strand items: converged / numerical / partly numerical.
  - Refining the grid with PhysCtrl's surface particles would thin the shell (a different object), so the spatial refinement fills each occupied cell with 27 particles (same shape and mass).
  - `code/p10d_strands.py`, `configs/p10d_strands.yaml`, `slurm/103_p10d_strands_A100.sh`. CPU `--check`: volume and total force preserved (max 0.2%, rubber boot, float32 cell boundary). GPU path not run here (no experiments on this host).
- **2026-10-22 (jobs 431955 and 431597):**
  - **103 / Phase 10d:** PARTLY NUMERICAL by the pre-registered rule. Stretch is stable under refinement (2.95 → 2.95); left-behind is time-step sensitive. Strands are real in this MPM, so the realism issue is the parameters. The summary excludes the 2 invalid items that had stale query files (`cases()` fixed).
  - **102 / Phase 12 S0:** failed at the first PolyFEM call. The cpu-all node (crimv3srv038) has an older glibc/libstdc++ than the Ubuntu 22.04 build host. Nothing was computed.
  - PolyFEM rebuilt as `build_conda` (gcc 15.3, glibc 2.17 sysroot; roundeven shim); smoke test reproduces the native build exactly; S0 job updated (env libstdc++ first, workers from nproc, 24 h). Ready to resubmit.
- **2026-10-22 (model-paper options; the user's pasted prompt):**
  - Three research agents (physics-grounded generators; simulator-feedback training; learned simulators and verifiers) opened every source.
  - Note: `4/notes/model_paper_options_2026-10-22.md`.
    - Weaknesses ranked.
    - Options A (certificate-driven simulator distillation, recommended, ~35–55 GPU-h), B (contact-aware generator, ~45–60 GPU-h, riskiest), C (amortised verifier, weak alone), D (learned corrector, not recommended), E (benchmark add-on to A).
    - Dead ends with reasons.
  - New threat: PAVG (arXiv 2610.11791, Oct 8, 2026) beats PhysCtrl in distribution.
  - Nothing pre-registered or run. Waiting for the user's decision (option, GPU budget, Objaverse download).
- **2026-10-22 (the user chose "A+B"):**
  - Taken as approval of both option budgets, spent in stages behind pilots.
  - Pre-registered pilots 14a (simulator distillation) and 15a (contact-aware generator) in PLAN before any code.
  - 14a code:
    - `p2_build_ood` `aug_` family (log E 3–8, force coeff 0.02–0.8, random drag point with PhysCtrl's 0.4 box);
    - `p14_train.py` (dry run passes, full and LoRA);
    - `p14a_acquire.py`, `p14_eval_generate.py`, `p14a_summary.py`;
    - configs `p14a_*`; jobs 104–108 + `submit_phase14a.sh`.
  - Found for 15a: PhysCtrl's DiT has a fixed sin-cos positional embedding for 2,048 points and one force, hence 1,024 points per object and a single pushed object.
- **2026-10-24 (14a stage 1, jobs 432496–432498):** all tasks exit 0; 2,950/3,000 candidates (50 unstable), all generated and scanned. Acquisition: 89% rejected, so ACT ∩ RND = 2,316/2,619. Pilot unchanged; a planning note for 14b (selective pool or top-scored budget) is in PLAN before any training. Stage 2 ready.
- **2026-10-24 (jobs 107/108 and 102; the user: "job finished, check"):**
  - **14a: KILL** by the pre-registered rule (the first summary had NaN from a list passed to `object_rate`; fixed before any reading). Every fine-tuned model is worse than base everywhere.
  - Diagnosis (exploratory): training is not broken; conventions match the replay data. The fine-tuned models move the right distance in the wrong direction or timing (centre-of-mass error 0.016 → 0.06–0.09).
  - **Found while diagnosing:** PhysCtrl's MPM has sticky walls 0.2 from every side of the domain. 84% of the ACT targets and 99% of Phase 9's stronger-push references hit them. Phase 10e (exploratory) re-scored Phase 10c without the walls: PhysCtrl's failures remain; most tuned-simulator shift failures disappear except drop. This weakens the "tuned simulators fail under shift" claim, so it goes to the user.
  - **Pilot 14a-v2 pre-registered** (wall-free pool, replay distillation as in 4g-v2, scoring before the wall; ≈ 7 GPU-h). `p14_train` and `p14a_summary` generalised; the 14a verdict reproduces byte-identically; self-test 30/30.
  - **Phase 12 S0 run 1: FAIL** at S0c (the push check). Cause: the force normalisation (particle vs mesh volume). Fixed as a recorded amendment, one rerun only; the smoke test passes (scratch).
- **2026-10-24 (status question; the user said "yes" to starting 15a):**
  - Gave an honest status (framework paper confirmed; the model contribution unproven; one model route can reach a full run before Nov 16; recommendation: decide by Oct 28–29).
  - **Phase 15a code written:** `p15_scenes.py`, `p15_build.py`, `p15a_eval.py`, `p15a_summary.py`; `fc_model.add_object_id`; `p14_train` gained `object_id` and `arm_build_runs`; configs `p15a_*`; jobs 111–113 + `submit_phase15a.sh`.
  - Clarifications recorded in PLAN before data: components from existing rollouts; A's force input; the composed baseline's zero-force B.
  - Scene design run (CPU): train 1,439 valid, dev 187 valid.
  - Code tests (scratch): PhysCtrl's TrajDataset reads the new files; the force input equals A's exactly; a F/C layout bug found and fixed; the hook is identity at init. Self-test 30/30.
- **2026-10-29 (the user: what is left, jobs to run, PokeFlex):**
  - No job had run since Oct 24: 14a-v2, 15a and the S0 rerun were still unsubmitted. One script now submits all of them: `4/slurm/submit_now_oct29.sh`.
  - Phase 12 test runs and verdict written (`p12_test.py`, `p12_summary.py`, job 114). Deviations recorded in PLAN before data: drop and the Phase 11 items are not run. Verdict code-tested on fabricated positions (scratch only).
  - PokeFlex: the new link works. Sizes read with header-only requests: 54 drop takes 25.9 GB, 116 poke takes 562.5 GB. Download script with size checks; nothing downloaded yet. The share has 1,020 GB free after the user's clean-up.
- **2026-11-03 (the user froze the project; asked for GitHub + Hugging Face archive commands, nothing lost, every number reproducible):**
  - Since Oct 29 no job ran (14a-v2, 15a, Phase 12 rerun never submitted); the project is frozen in that state.
  - Archive kit `freeze/`: `areas.py` (every folder → upload area / pinned third party / deliberate exclusion, with a completeness guard), `make_manifest.py` + `check_files.py` (sha256 of every file), `stage_hf.py` (symlink upload trees), `check_remote.py` (remote vs manifest), `restore.py`, `remap_paths.py`, `fetch_third_party.sh` (pinned revisions read from the download metadata), `verify_numbers.py` (re-runs every dev-server verdict, byte comparison), `redact_transcript.py`, env snapshots, PolyFEM build recipe, HF cards, `COMMANDS.md`. No git or hf command was run by Claude.
  - Found: `data/hf_cache` holds a Hugging Face login token (excluded); PokeFlex link only in `PokeFlex/` and `_pokeflex_download.sh` (excluded); PhysCtrl declares no licence (data/checkpoint repos private by default).
  - Exported the claude.ai decks (slide sources) and the study page (byte-identical) to `4/artifacts_export/`; the study store was empty.
  - Inline Oct 24 analyses frozen as `code/p14a_diagnose.py`, `code/p12_s0_diagnose.py`; one number corrected (replay wall share 2.9%, not 1.3%).
- **2026-11-07 (the user asked where things stood and wants to delete after archiving):**
  - State found: step 1 (verification) finished Nov 3; manifests, upload folders, git repo and all uploads did not exist. The HF repos do not exist. Nothing may be deleted yet. The share is 99% full (170 GB free).
  - Traced the 4 non-identical verdicts (`freeze/VERIFY_NOTES.md`): none changes a reported number. Found that job 96 ran twice (430617, 430786), which was never logged. Fixed `p12_s0.py`'s data path.
  - Added `freeze/delete_local.py` (tiers, live guards; sandbox-tested with fake files, guards checked on the real state), the `REMOTE_OK.json` stamp in `check_remote.py` (sandbox-tested with a stub API; listing tested on the real repo chenwang/physctrl, sha256 matches the local testset.zip) and symlink handling (144 render links to input images are uploaded as real files).
  - Started steps 2–4 (redaction, full manifest, upload folders) on crimv3mgpu015. No git or hf write command was run; nothing was deleted.
- **2026-11-07/08 (uploads; the user ran every git and hf command):**
  - Steps 2–4 finished (`PREPARE EXIT 0`). The user pushed GitHub `rasulkhanbayov/somw` (public, HEAD `eda8d1d`, tag on the remote) and uploaded `somw-results` and `somw-checkpoints`.
  - `somw-data` (38,260 files, 221.4 GB): over xet it ran at about 0.2 MB/s, and about 1 MB/s with `HF_XET_HIGH_PERFORMANCE=1`. With `HF_HUB_DISABLE_XET=1` (classic LFS) it averaged about 15 MB/s, despite the free-account limit of 1,000 API requests per 5 minutes; 429s and failed commits were retried automatically. It finished Nov 7 22:51 with 38,260/38,260 committed.
  - Claude ran `check_remote.py` (read-only): every manifest file matches in all three repos (results 2,281, data 38,259, ckpt 26; no extra files); all three are public. It wrote `freeze/REMOTE_OK.json` (revisions fd4ec4f7, ab8dbaf7, a302b3fc).
  - `delete_local.py` dry run: 2 guards unmet, the uncommitted `REMOTE_OK.json` and `somw-private` (step 6b not done). Plan: ours 224.2 GB, scratch 0.8, renderer 125.6, physctrl 5.2 (kept). Nothing deleted.
- **2026-11-08 (the user ran 6b, 8 and the first deletions; Claude checked):**
  - Step 8 pushed (`01fa1a2`, `freeze/RELEASE.md`). The user deleted the ours tier (26 paths, 224.2 GB) and the scratch tier (9 paths, 0.8 GB), with 81.4 GB freed in home and 143.6 GB in hbku-research, all logged in `freeze/deleted_local.log` (not yet committed). Renderer tier (125.6 GB) and physctrl tier (5.2 GB) are still on disk.
  - **Found: `github.com/rasulkhanbayov/somw-private` was created public** (created 22:49Z, pushed 22:53Z, Nov 7; 0 forks or stars at 23:45Z). Scan of the pushed archive: no PokeFlex or Hugging Face token (checked against the real values), no token patterns, no e-mail addresses. Exposed: the redacted session record, the study answer code and the commit author e-mail. The user must switch it to private.
  - `delete_local.py` gained a guard: anonymous GitHub API must answer 404 for the private repo (tested: it refuses now). COMMANDS 6b gained the same check. The local `~/somw-private` checks out (`SHA256SUMS` OK, the zstd archive is intact). The redacted-record stage `freeze_private` was deleted with the scratch tier; `redact_transcript.py` can rebuild it from the raw session files in `~/.claude/projects/` as long as those exist.
- **2026-11-08 (freeze complete):** the user made `somw-private` private (anonymous API and raw link: 404; remote = local `fee1bfa`) and deleted the renderer tier (125.6 GB) after all guards passed; pushed `fd33056`. `$BIG_ROOT` now holds only an empty `checkpoints/`; free space: hbku-research 410 GB, home 237 GB. Kept: physctrl tier, `data/hf_cache` (login), conda envs (about 17 GB), raw session files.

