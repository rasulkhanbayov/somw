# INDEX: where everything is

Update this file whenever a file is added, removed or renamed (rule 6 in CLAUDE.md). The "Phase" column refers to PLAN.md.

## Top level

| Path | What it is | Why it's needed | Phase |
|---|---|---|---|
| `CLAUDE.md` | Standing instructions for Claude sessions in this folder | Scope, hard rules, conventions, gates | all |
| `INDEX.md` | This map | Find any file and its purpose | all |
| `PLAN.md` | Phased plan with goals, scripts, GPUs, estimates, falsifiers, fallbacks | What to run next, and what each outcome means | all |
| `proposal/Simulate Only What Matters — Research Proposal.md` | The source proposal (copy; the original stays in the workspace root) | The idea, related work, risks | reference |
| `notes/paper_story.md` | Running paper story: what each contribution rests on, key figures and tables, open risks | Kept current after every result | all |
| `notes/item3_contact_benchmark_design_2026-10-14.md` | Design memo for the multi-object contact benchmark (queue item 3): construction, scope decisions, H0–H3, code changes, ≈ 21–33 GPU-h | Not pre-registered; awaiting the user's scope decisions | item 3 |
| `notes/audit_2026-10-14.md` | Senior-review status and risk audit: confirmed / exploratory / failed claims; reviewer objections by severity with fixes | Oct 14 | all |
| `notes/literature_2026-10-14.md` | Literature check (2025–2026): threat table, positioning sentences, demanded baselines; gs_physics unverified | Oct 14 | all |
| `notes/audit_literature_2026-10-17.md` | Second review pass: 10 weaknesses with causes, verified 2025–2026 competitors and expected metrics, ideas to borrow, pilots P1–P4 | Oct 17 | all |
| `notes/efficiency_c3_2026-10-18.md` | Contribution 3: where compute goes, why we cannot beat a tuned simulator on soft materials, the stiff-material regime, verified efficiency literature (McGreivy & Hakim rules, break-even), pilots E1–E3 | Oct 18 | C3 |
| `notes/paper_review_2026-10-20.md` | Senior-AC review of the draft against PhysCtrl, PSIVG, RealWonder, PhysFlow, NEXUS, Hybrid NWM, ANCHOR (full-text reads); 11 weaknesses ranked; 12 edits with impact/effort/GPU (★1–★5 first); no method change |
| `notes/simulator_audit_2026-10-22.md` (+ `notes/simulator_audit_2026-10-22/`, contact sheets) | Audit after the first study rater: reference = PhysCtrl's own data simulator (no bug); stretching regime (soft + concentrated push), renderer artifacts, error from off-screen frames; Phase 9 softer family 32% in that regime; ranked recommendations R1–R5 | — |
| `notes/simulator_independence_2026-10-22.md` | Circularity breakdown (ours = reference on 52–57% of shifted/contact queries), independent-simulator survey (PolyFEM recommended; conda build unusable, source build at `$BIG_ROOT/tools/polyfem`), real-world data survey (PokeFlex), Phase 12 pre-registration summary | — |
| `notes/pokeflex_file_sizes_2026-10-29.tsv` | PokeFlex zip sizes from the server (Oct 29, header-only requests): interaction, object, take, bytes (588 GB in all) | — |
| `notes/model_paper_options_2026-10-22.md` | Model-paper options after a verified literature search (3 research agents): weaknesses ranked; options A (certificate-driven simulator distillation, recommended), B (contact-aware generator), C (amortised verifier), D (learned simulator corrector), E (benchmark add-on); new threats PAVG (2610.11791), NEXUS, PhysiFormer, PhysLDM, DSO | — |
| `notes/improvement_analysis_2026-10-11.md` | Diagnosis of the Phase 2-v2 weak points → ranked improvements → pre-registered Phase 2c questions | Keeps all three contributions | 2c+ |
| `notes/physctrl_ref_snapshot/` | Read-only copies of the PhysCtrl files our code depends on (configs, eval, pipeline, dataset, MPM solver, data generators) | Lets the bridge be checked against the real interfaces without the cluster checkout | reference |
| `.gitattributes`, `.gitignore` | Force LF line endings (bash on Linux breaks on CRLF); ignore results and logs | Safe transfer from Windows | — |

## code/: Python (all runnable with `--dry-run` on a CPU)

| File | What it does | Depends on | Phase |
|---|---|---|---|
| `common.py` | Config loading with `${ENV}` expansion (fails on unset vars), `Run` provenance (`run_meta.json`: host, GPU, Slurm ids, git commits, config), JSON/JSONL I/O, sharding, synced timing, Spearman, bootstrap CIs, SYNTHETIC guard | numpy, pyyaml | all |
| `physctrl_bridge.py` | **The only module importing PhysCtrl.** `MPMSim`: elastic Warp MPM scene built exactly like PhysCtrl's data generator, with restartable state and 3 timing modes. `Generator`: base/large checkpoints via their `TrajPipeline` and `TrajDataset`. HDF5 I/O, particle volume, coordinate maps. Holds the five silent-failure guards | `$PHYSCTRL_ROOT`, warp, torch, diffusers | 0–2 |
| `backends.py` | One interface, two backends. `PhysCtrlBackend` (real). `SyntheticBackend` (tiny mass-spring stand-in for dry runs and tests; its numbers mean nothing). Manifest helpers for the test set and OOD sets | `physctrl_bridge` (real only) | 0–2 |
| `lift.py` | State lifting from positions: velocity by 3-point quadratic difference; F and C by per-particle least squares over k neighbours. Exact on affine/quadratic motion (self-tested) | numpy, scipy (optional) | 2 |
| `verifier.py` | Window defect (`run_window`: restart from lifted/oracle/true-initial state, run, compare; chaos sigma from perturbed restarts). Placement strategies: uniform, random, event, disagree, proposal, oracle. Aggregation | `lift` | 2 |
| `regimes.py` | Phase 1 scenes from test objects: `dense10x` (particle splitting with mass-consistent drag), `dense10x_fine` (250³ grid), `multiobj2/3` (stacked objects colliding under gravity) | `physctrl_bridge.particle_volume` | 1 |
| `p0_check.py` | Imports, checkpoints, **test-set manifest**, replay determinism in 3 timing modes, generator sanity; writes filled configs for PhysCtrl's own `eval.py` | backends | 0 |
| `p1_cost.py` | Generation vs simulation wall-clock per regime and mode, probe cost; one GPU type per run (`--tag`) | backends, regimes | 1 |
| `p1_fidelity.py` | Simulator fidelity sweep (substep × grid) vs ground truth, plus generator error on the same queries; sharded | backends | 1 |
| `p1_summarize.py` | Phase 1 tables and **GO/NO-GO**, including the matched-accuracy simulator cost | results of `p1_cost_*`, `p1_fidelity_A100` | 1 |
| `p2_build_ood.py` | OOD elastic queries (E low/high, force ×3, gravity drop) re-simulated from test objects; writes PhysCtrl-format HDF5 and manifests; sharded | backends, test-set manifest | 2 |
| `p2a_lift_floor.py` | Verifier on ground-truth trajectories: 7 lift variants, raw/normalised defects, lifted-state errors; sharded | lift, verifier | 2 |
| `p2b_generate.py` | K samples per query from base and large → `$DATA_ROOT/generations/…/<qid>.npz`; resumable, sharded | backends, manifests | 2 |
| `p2b_probe.py` | True error, seed-spread/motion baselines, window defects for 6 placements, with simulator cost; resumable, sharded | generations, verifier | 2 |
| `p2_summarize.py` | Phase 2 tables and **G2a/G2b verdicts** | results of `p2a_lift_floor`, `p2b_probe` | 2 |
| `diag_p2.py` | Exploratory Phase 2 diagnostics: error localisation in time, error type (shift/affine), jitter, contact timing, per-split Spearman → `results/diag_p2/` | generations, ground truth, probe rows | 2 |
| `p2c_scan.py` | **Phase 2c** exhaustive window scan: every window × m ∈ {1,2,4} × {ref, coarse} × {base, large, gt}; per-window rows; resumable, exit-75 aware | verifier, generations | 2c |
| `p2c_summarize.py` | **Phase 2c** pre-registered verdicts A (fidelity), B (window length + fusion), C (placement capture on contacts), D (single-threshold AUROC) | scan, probe rows | 2c |
| `p2c_select.py` | Phase 2c signal selection (selection set; exploratory): fidelity × m × placement, alone and fused → `results/p2c_select/SELECT.md` | scan, probe rows | 2c |
| `p4a_certificate.py` | **Phase 4a** class-conditional split-conformal certificate on the calibration set (split by object), P1 validity / P2 power / P3 shift → `results/p4a_certificate/` | calibration scan + errors, Phase 2c | 4a |
| `p4b_certificate_objects.py` | **Phase 4a extension** object-level conformal (single-draw subsampling and object-weighted pooling, 500 object splits) → `results/p4b_certificate_objects/` | both calibration sets, Phase 2c | 4a |
| `p4b_repair.py` | **Phase 4b** training-free repair: anchor + certificate windows as hard constraints in PhysCtrl's DDIM loop (R0/R0p/R1/R2/R3); equivalence check first | `Generator.sample_constrained`, verifier | 4b |
| `p4b_summarize.py` | **Phase 4b** verdict: R2 vs R1 on rejected base trajectories, plus propagation and re-acceptance tables | repair rows | 4b |
| `p4c_repair_conditioning.py` | **Phase 4c** conditioning repair: 30 (Δlog₁₀E, force scale) candidates from shared noise, chosen by certificate defect + anchor mismatch; held-out defect; identity check first | `Generator.sample_constrained(adjust=…)`, verifier, Phase 4b rows | 4c |
| `p4c_summarize.py` | **Phase 4c** verdict: R4 vs R1/R2/R0/R4id on rejected base trajectories; chosen adjustments; held-out defect | 4b and 4c rows | 4c |
| `p4d_fullsim.py` | **Phase 4d data**: one full rollout per query from the true initial state at the fallback fidelity, A100-timed | backend, entry sets | 4d |
| `p4d_pipeline.py` | **Phase 4d** certify-or-simulate as a compute-accounted pipeline: gen / random / cert / oracle / sim; T1–T4 | fullsim, Phase 1 costs, 2c scan + probes | 4d |
| `p4e_repair_anchored.py`, `p4e_summarize.py` | **Phase 4e** anchored conditioning repair (selection by a true-start rollout over L = 8/12 frames) and its verdict | 4c grid, verifier, 4b rows | 4e |
| `p4f_confirm_pipeline.py` | **Phase 4f** held-out confirmation of certify-or-simulate with a reference fallback (500 object splits; F1 risk, F2 cost, F3 value) → `results/p4f_confirm/` | calibration pool, Phase 1 costs, 2c A100 timings | 4f |
| `make_manifest.py` | Query manifest for a folder of PhysCtrl-format HDF5 files (used for the 4g training objects) | h5py | 4g |
| `fc_model.py` | **Phase 4g** frame-conditioned PhysCtrl: zero-initialised known-frame embedding through an `input_encoder` hook; the mask recipe; save/load; **v2**: temporal-attention LoRA (`add_temporal_lora`, `set_lora`); **15a**: object-ID embeddings (`add_object_id`) | torch (lazy) | 4g |
| `p4g_train_fc.py` | **Phase 4g** fine-tune of base with random frame masks (PhysCtrl's x₀ + velocity loss); identity check, resume, CPU dry run. `train.mode: lora` (v2) freezes base, trains LoRA + embedding, and adds distillation to the frozen base | fc_model, Generator, TrajDataset | 4g |
| `p4g_repair_fc.py`, `p4g_summarize.py` | **Phase 4g** repair with the fine-tuned generator (R7/R7u/R8, same constraints and seeds as 4b) and its verdict | fc_model, 4b rows | 4g |
| `p4h_suffix_repair.py`, `p4h_summarize.py` | **Phase 4h** certified-prefix repair: first failing window in a temporal chain (from the 2c scan) → re-simulate only the suffix from the lifted state; suffix lift floor from ground truth; verdict vs certify-or-simulate; done (FAIL on risk) | 2c scan, verifier, Phase 1 costs | 4h |
| `p4i_f_bounds.py`, `p4i_state_repair.py`, `p4i_summarize.py` | **Phase 4i** full-state suffix restarts on the 4h suffix queries: F-clamp bounds from the training objects (CPU, done: [0.880, 1.353]); V1 stable F-lift (projected F) + floor, V2 coarse-shadow state from the true start; per-variant verdict vs certify-or-simulate (Bonferroni) | 4h rows, 2c scan, 4d coarse timings, `lift.project_F` | 4i |
| `p4j_allk.py`, `p4j_confirm.py` | **Phase 4j** held-out confirmation of 4i V2 on the 1,200 calibration queries: V2 at every restart point k (one coarse shadow + five reference suffixes per query, GPU); then the 4f 500-split protocol with V2, shuffled-restart ablation, earliest restart and coarse-only baselines (CPU; J1–J3) | 4f code, 4d timings, calibration scans/errors | 4j |
| `p5a_amortise.py` | **Phase 5a** audited amortisation of the certificate over a query stream (learned skip rule + random audits + split conformal); A1–A3; drift-order secondary (CPU) | 4k/4m/2c scans, errors, generations | 5a |
| `p7a_metrics.py` | **Phase 7a** reviewer metrics + literature baselines on the 4m fresh pool: momentum-residual gate (ANCHOR-style), best-of-8 (medoid / residual / oracle), AUROC + AURC, risk at 3 thresholds, tails, last-frame error; Q1–Q4 (CPU) | 4m scans, errors, all-k rows, generations | 7a |
| `p7b_summarize.py` | **Phase 7b** PhysCtrl's own metrics (eval.py MSE, DeformLoss) on its test set for generate / certify-or-simulate / gated V2 / simulate; descriptive (CPU) | job 77 rows + saved trajectories, 2c scan | 7b |
| `p7c_targets.py` | **Phase 7c** ε × α grid with every gate recalibrated (window, seed spread, residual, agreement); S1–S3 at ε = 0.05 in distribution; cheapest pipeline meeting each target (CPU) | 4m pools, 7a residual functions | 7c |
| `p7d_contact.py` | **Phase 7d** contact benchmark: coarse only, XF, seed-spread deferral, momentum-residual gate vs SF; D1–D3 + XF dominance rule (CPU, after job 78) | 4o rows (`p4n_summarize.build`), job 78 coarse rollouts, 4m/7a thresholds | 7d |
| `p8a_pilot.py` | **Pilot 8a** (exploratory): fidelity ladder; medium full rollouts (M1–M3) and gated V2 with a medium suffix vs always-simulate-medium on development data (CPU, after job 79) | job 79 rows, p4d coarse rows, 2c scan, p2b_probe_v2 | 8a |
| `p8b_fused.py` | **Pilot 8b** (exploratory): fused certificate score (windows + residual + seed spread + generation statistics; logistic regression; object-thirds) vs window-only (CPU) | calibration pool, 2c shift splits | 8b |
| `p8d_certsim.py` | **Pilot 8d** (GPU): M1 proposal rollout, M3 check windows from M1's own state at k = 3, 11, 19, reference suffixes | 2c pool, 4l scenes, id_new_0..3 | 8d |
| `p8d_pilot.py` | **Pilot 8d** analysis: certified M1 (novelty threshold, suffix repair) vs M1 alone and the 8a pipelines (CPU) | job 80 rows, 8a summary | 8d |
| `p8e_stiff.py` | **Pilot E1/E3** analysis: stiff regime (E 10^7.5–10^9), ladder vs ours (gated, M2 repair), S*, idealised latency (CPU) | jobs 81–83, p8e_errors | E1 |
| `p8f_cheapcheck.py` | **Pilot E2**: cheaper verification variants (1, 2, 3 windows; m = 2) calibrated on the calibration pool, evaluated on 2c (CPU) | existing scans | E2 |
| `p9_confirm.py` | **Phase 9** single-object verdict: ours-v2 (one-window certificate, M1 shadow, M2 suffix) vs every baseline incl. tuned simulators; C1–C5; contact ladder baselines (CPU). `assemble()` returns every pipeline's per-query (error, cost) | jobs 87–89, 93; 7a/4m thresholds | 9 |
| `p10c_decisions.py` | **Phase 10c** step 1 (CPU): every Phase 9 pipeline's decision per query + stored errors to reproduce → `results/p10c_decisions/decisions.json` | Phase 9 rows | 10 |
| `p10c_rescore.py` | **Phase 10c** step 2 (GPU, job 101): positions of every pipeline's output and of the reference at dt, dt/2, dt/4 → `$BIG_ROOT/p10c_traj`; checks reproduction of Phase 9 errors (≤ 2e-3) | decisions, Phase 9 configs | 10 |
| `p10c_summary.py` | **Phase 10c** verdict (CPU): exact risk of every pipeline against each reference; T1–T3 (`--sandbox` = code test) | `p10c_traj` | 10 |
| `p10c_explore.py` | **Phase 10c, exploratory** (CPU, after the verdict): risk gap to always-simulate at dt per reference, and robust risk (wrong against all three references) → `results/p10c_explore/` | `p10c_traj`, `p10c_summary` helpers | 10 |
| `p10c_table.py` | Phase 10c LaTeX tables: `paper/tab/reference_table.tex` (main rows) and `paper/supp/reference_table_full.tex` (all pipelines) | `results/p10c_summary`, `results/p10c_explore` | 10 |
| `p10d_strands.py` | **Phase 10d** (GPU job 103): are the strands numerical? 23 shifted Phase 11 items at dt/2, dt/4 and on a body-preserving filled body (27 particles per occupied cell) at grid 125/250 → `$BIG_ROOT/p10d`; `--summarize` (CPU) verdict → `results/p10d_summary/`; `--check` (CPU) body construction | `renders_p11/*/query_v2.h5` | 10 |
| `p12_fem.py` | **Phase 12** independent simulator S2 = PolyFEM (fixed-corotated, implicit Newmark), built from source (`$BIG_ROOT/tools/polyfem/build_portable`); occupied-cell mesh = the MPM's body, particles embedded; drag as a box body force; runs in env `somw-fem` (CPU) | PhysCtrl-format HDF5 | 12 |
| `p10e_wall.py` | **Phase 10e** (CPU, exploratory): the MPM domain wall. Wall-reach rule (`wall_frame`), Phase 10c pipelines re-scored on wall-free queries and before the first wall frame → `results/p10e_wall/` | `$BIG_ROOT/p10c_traj`, `p10c_decisions` | 10 |
| `p12_s0.py` | **Phase 12 gate S0** on development queries only: dt rule, free fall, small-strain agreement, push check (v2 after the Oct 24 amendment: snapped-box force, mesh centre of mass) → `results/p12_s0_v2/` (run 1: `results/p12_s0/`, FAIL); `--smoke DIR` = code test | `p12_fem.py`, `data/calib/id_new_0..2` | 12 |
| `p12_s0_diagnose.py` | **Phase 12** S0 run-1 diagnosis (CPU, mesh only): mesh-in-box vs masked-particle volume per query, correlation with the S0c errors → `results/p12_s0_diagnose/` | `results/p12_s0` | 12 |
| `p12_test.py` | **Phase 12** test runs (cpu-all): PolyFEM on the 500 primary Phase 9 queries; runs only if S0 v2 passed → `$BIG_ROOT/p12_traj/test`, `results/p12_test/` | `p12_fem.py`, `p10c_decisions`, `p9_build` | 12 |
| `p12_summary.py` | **Phase 12** verdict (CPU): X0–X3, every pipeline vs FEM and vs MPM on the same (pre-wall) frames → `results/p12_summary/` | `p10c_traj`, `p12_traj/test` | 12 |
| `p14_train.py` | **Phase 14** (H200): fine-tune PhysCtrl base on reference rollouts (simulator distillation), 50% arm data + 50% original-distribution replay; full or LoRA (targets configurable); optional replay distillation to the frozen base (14a-v2); PhysCtrl's base loss | `results/p14a_acquire/arm_*.json`, 4g replay data | 14 |
| `p14a_acquire.py` | **Phase 14a** (CPU): one-window certificate on the candidate pool → arms ACT (rejected) and RND (random, same size); `--wall-free --run-name p14a2_acquire` for pilot 14a-v2 | `p14a_scan`, `p14a_generate`, `p14a_build` | 14 |
| `p14_eval_generate.py` | **Phase 14** (H200): sample 0 of base and each fine-tuned checkpoint (LoRA re-attached, strict load) on evaluation queries | checkpoints, query manifests | 14 |
| `p14a_diagnose.py` | **Phase 14a** diagnosis (CPU, exploratory): wall survey of the pool, arms, replay and test families; motion split of the fine-tuned generations → `results/p14a_diagnose/` | `p14a_acquire`, `generations_p14a` | 14 |
| `p14a_summary.py` | **Phase 14a / 14a-v2** verdict (CPU): kill rule, variant choice, ACT vs RND descriptive; config key `summary` (14a-v2: errors before the wall) | `generations_p14a` | 14 |
| `p15_scenes.py` | **Phase 15a** (CPU): two-object scene design (A pushed into B at rest; pairing, placement, validity, FPS subsets) → `results/p15a_scenes/` | `fc_train`, `data/calib` component rollouts | 15 |
| `p15_build.py` | **Phase 15a** (A100): joint two-object rollouts + 2,048-point model files (PhysCtrl layout, A's force input) + B rest files → `$BIG_ROOT/p15/`, `results/p15a_build/` | `p15a_scenes` | 15 |
| `p15a_eval.py` | **Phase 15a** (H200): composed PhysCtrl, base on the joint input, fine-tuned model on the development scenes → `$BIG_ROOT/generations_p15a` | `p15a_build`, `p15a_FT` checkpoint | 15 |
| `p15a_summary.py` | **Phase 15a** verdict (CPU): kill rule, contact/control errors, error after first approach | `generations_p15a` | 15 |
| `p10a_convergence.py` | **Phase 10a** reference convergence (CPU): R1 per-set 95th percentile of the half-step distance, R2 triangle-inequality risk bounds for every Phase 9 pipeline (`--self-check`: bounds collapse onto the Phase 9 table at d = 0), R3 free fall of the drop ground truth, recorded frame-time offset of M1–M3 | job 96 rows, Phase 9 rows | 10 |
| `p9_table.py` | Phase 9 main table (descriptive, object-weighted, all three test sets), certificate accept counts, the LaTeX table `paper/tab/main_table.tex` and the risk-vs-cost figure `paper/fig/tradeoff.{pdf,png}` → `results/p9_table/` (CPU) | Phase 9 rows; `p9_confirm.assemble`, `p4n_summarize.build/pipes` | 9 |
| `p4n_summarize.py` (also Phase 4o) | Verdict for 4n (`primary: eaf`) and 4o (`primary: sf`, the shadow-timed switch read from the GPU rows); `p4j_allk` gained `shadow_switch_dist` (shadow state at every frame, restart where the shadow's objects come within 3 coarse cells) | **Phase 4n** verdict: event-aware fidelity repair vs certify-or-simulate and the 4k chain on fresh contact scenes (N1 risk, N2 cost, N3 timing); computes true errors itself (dev server). `p4j_allk` gained `fidelity_restart_dist`; `_common.sh` gained `run_two_shards` | 4n/4o runs | 4n, 4o |
| `p6_video.py` | **Phase 6a** videos: PhysCtrl's SAM → SV3D → LGM preprocessing, a PhysCtrl-format query from the reconstruction (drag box/force exactly as `inference.py`), our generation + reference simulation + gated-V2 certify/repair, then PhysCtrl tracking + DAS render of generation / certified / ground truth. `v2: true` (paper re-render, Oct 19): the Phase 9 pipeline, variant `certified_v2`, `trajectories_v2.npz`/`case_v2.json`, Phase 6a records untouched | PhysCtrl `src/inference.py`, renderer weights in `$RENDER_CKPT_DIR` | 6a, 9 |
| `mo_scenes.py`, `p4l_mo_build.py`, `p4l_summarize.py` | **Phase 4l** multi-object contact benchmark: pairing / shared physics / placement design / composition / generated contact frame; scene builder (design, joint 4,096-particle ground truth, sanity check, composed `base` and `gt_super` generations); verdict H0–H3 + restart curve. Also: `p2_build_ood` `mo_comp_` splits, `p2c_select` `contact`/`contact_spread` placements, per-object kNN in `verifier.nbrs_for`, `object_id` in HDF5 I/O, `p4j_allk` `event_restart` | component builds, generations | 4l |
| `p4m_confirm.py` | **Phase 4m** verdict: fresh shifted queries; certificate FAR vs seed spread (M1), pipelines under shift (M2), cross-fidelity agreement baseline XF (M3), uniform placement at matched budget (M4), seed-spread deferral | 4k loader, calibration pool, saved coarse positions | 4m |
| `p4k_gated.py` | **Phase 4k** verdict: cert-gated V2 vs certify-or-simulate on fresh queries (K1 risk, K2 cost at τ and τ_hi, K3 timing via the exact permutation expectation; object-cluster bootstrap). `p4j_allk.py` gained `shadow_to_end` (coarse-only from the same shadow) | 4k runs, 4f cost fit, 4d rates | 4k |
| `explore_gated_2c.py` | Exploratory (after 4k): gated V2 vs certify-or-simulate vs coarse-only per split on the Phase 2c queries (incl. OOD), from 4h/4i rows | 4h, 4i, 2c scan, 4d | — |
| `fig_qualitative.py` | Qualitative figures for the results deck (failure modes, one repaired query's timeline, risk vs cost); examples chosen by a fixed median rule → `results/fig_qualitative/` | 4h/4i/4k results, generations, test set | — |
| `fig_ours_traj.py` | Deck figures: recomputes the positions of our output (repair or exact fallback) for the 5 cases in `results/fig_qualitative/deck_selection.json`; checks them against the stored verdict-run errors (GPU, job 76). `--selection paper_selection.json --prefix paper_` (job 94): the Phase 9 cases with the pool's suffix setting, plus an always-M1 rollout | 4m/4o/p9/p9c scenes, generations, allk rows | — |
| `fig_teaser.py` | Paper Fig. 1: the rule-selected contact scene over time (PhysCtrl / ours / reference) + worst risk vs largest cost over the three Phase 9 test sets → `paper/fig/teaser.{pdf,png}` (CPU) | `paper_contact_switch.npz`, `p9_table`, `p7d_contact` | 9 |
| `fig_certificate.py` | Paper figure: risk–coverage curves (window vs agreement vs residual vs seed spread; random-acceptance floor; perfect ranking) → `paper/fig/certificate.{pdf,png}` (CPU) | `results/p7a_metrics` | 7a |
| `supp_videos.py` | Supplement: labelled PhysCtrl / ours / reference side-by-side MP4s for every case with a Phase 9 record → `$BIG_ROOT/supp_videos/` (CPU, ffmpeg) | renders (job 95, Phase 11) | — |
| `p11_images.py` | **Phase 11** inputs: 25 FLUX.1-schnell object images + rule configs → `$BIG_ROOT/p11_examples` (GPU, job 98; `--dry-run` = configs only) | FLUX.1-schnell (`00c`) | 11 |
| `p11_judge.py` | **Phase 11** Qwen2-VL-7B judge with PhysCtrl's prompt, one video per call (GPU, job 100; `--extract-only` = frames via ffmpeg on a dev server first) | renders | 11 |
| `p11_summary.py` | **Phase 11** verdict: V0 judge validity, V1 ours vs PhysCtrl on rejected shifted cases (CPU) | judge scores, case records | 11 |
| `p11_study_build.py` | **Phase 11** preference-study items (CPU): 28 rejected shifted cases, opaque video names (seed 11) → the study page's files; local mapping → `results/p11_study_design/mapping.json` (never published) | Phase 11 + job 95 renders | 11 |
| `p11_study.py` | **Phase 11 V2** verdict (CPU): share of raters preferring ours on Q1, rater-clustered CI ; reads the store export and rater answer codes (`--codes`, `--items`; `--sandbox` = code test) | exported study responses, mapping | 11 |
| `fig_method.py` | Paper figure: the method diagram (accept path, repair path) → `paper/fig/method.{pdf,png}` (CPU, no data) | — | — |
| `fig_paper_select.py` | Paper figures: fixes the Phase 9 qualitative cases by the median rule before any picture → `results/fig_qualitative/paper_selection.json` (CPU) | Phase 9 rows | 9 |
| `fig_paper_points.py` | Paper figures: point clouds, exact / PhysCtrl / always M1 (shift) or simulated apart (contact) / ours + error per frame → `paper/fig/qual_{shift,contact}.pdf` (CPU; `--stand-in` = layout check to a scratch dir) | `paper_*.npz` (job 94), gt_super | 9 |
| `fig_paper_video.py` | Paper figure: the Phase 6a rule on ours-v2's decisions; PhysCtrl / ours / exact frames + error over the clip → `paper/fig/qual_video.pdf` (CPU; `--stand-in`) | job 95 renders | 9 |
| `fig_deck_points.py` | Deck figures: point-cloud side views, exact vs PhysCtrl vs coarse-only / simulated-separately vs ours, + error per frame → `deck_points_{shift,contact}.png` (CPU) | `ours_*.npz`, gt_super | — |
| `fig_deck_video.py` | Deck figures: frame 24 of the five rejected 3×-force renders → `deck_videos.png`; earphone ×3 over time with per-frame errors → `deck_video_earphone.png` (slide 12) (CPU) | Phase 6a renders | — |
| `selftest.py` | 28 checks (incl. the Phase 4i F projection, incl. exit-75 crash handling, the generator batch limit, the Phase 2 blow-up rule, AUROC and the physics-prior contact frame): lift exactness, velocity rule, statistics, restart exactness, zero defect on own simulator, window validity, synthetic guard, env-var guard, particle volume, frame mapping, frame-time arithmetic | the modules above | before every job |

## configs/: one YAML per experiment

| File | Used by | Key settings |
|---|---|---|
| `_paths.yaml` | reference only | the shared `paths` block every config repeats |
| `p0_check.yaml` | `p0_check.py` | replay 8 queries, tol 0.01, 3 timing modes, generator 8 queries |
| `p1_cost.yaml` | `p1_cost.py` | batch 1/8/32, 5 regimes, 3 modes, probe windows of 1 and 2 model frames |
| `p1_fidelity.yaml` | `p1_fidelity.py` | 40 queries; substep {1e-4…1.6e-3} × grid {64, 96, 125}; 4 generator samples |
| `p1_summary.yaml` | `p1_summarize.py` | `go_ratio` 10; realistic regimes |
| `p2_ood.yaml` | `p2_build_ood.py` | 100 base objects × 4 OOD splits |
| `p2a_lift_floor.yaml` | `p2a_lift_floor.py` | 40 ID + 10 per OOD split; 6 starts × 2 lengths × 7 variants; 4 perturbations |
| `p2b.yaml` | `p2b_generate.py`, `p2b_probe.py` | 100 ID + 50 per OOD split; 8 samples; 3 windows of m = 1; 2 perturbations |
| `p2b_v2.yaml`, `p2_summary_v2.yaml` | `p2b_probe.py`, `p2_summarize.py` | v2 rerun: `use_F: false` (= `lift_noF`); outputs `p2b_probe_v2`, `p2_summary_v2`; G2a floor variant `lift_noF` |
| `p2c_scan.yaml`, `p2c_summary.yaml` | `p2c_scan.py`, `p2c_summarize.py` | Phase 2c: m 1/2/4, ref + coarse (dt 8e-4, grid 64), lift_noF; pre-registered A–D |
| `p4_calib_build.yaml`, `p4_calib_gen.yaml`, `p4_calib_scan.yaml` | `p2_build_ood.py`, `p2b_generate.py` + `p2b_probe.py`, `p2c_scan.py` | Phase 4a calibration set: 100 objects × 4 in-distribution draws → `$DATA_ROOT/calib` |
| `p4_calib2_build.yaml`, `p4_calib2_gen.yaml`, `p4_calib2_scan.yaml` | calibration extension | `id_new_4..11`; scan m = 4 at `ref` + `coarse_cfl` only |
| `p4g_train_build.yaml`, `p4g_train.yaml`, `p4g_repair.yaml` | Phase 4g data / frozen fine-tune / evaluation | sample/ objects × 30 in-distribution draws; 20k steps, lr 5e-5, batch 8 |
| `p4g_train_v2.yaml`, `p4g_repair_v2.yaml` | Phase 4g-v2 frozen fine-tune / evaluation | LoRA r8 on `attn_temp`, λ_distill 1, 10k steps, lr 1e-4; checkpoint `fc_base_v2` |
| `p5a_amortise.yaml` | Phase 5a | pools 4k/4m/2c; audit 0.10; δ 0.05; 50 orders |
| `p7a_metrics.yaml` | Phase 7a | the 4m config + Bonferroni over 4 tests |
| `p7b_testset.yaml` | Phase 7b | test_id; chain restarts with saved positions (`$BIG_ROOT/p7b_traj`) and DeformLoss |
| `p7c_targets.yaml` | Phase 7c | ε, α ∈ {0.05, 0.10, 0.20}; Bonferroni over 3 |
| `p7d_contact_coarse.yaml`, `p7d_contact.yaml` | Phase 7d | coarse-only rollouts of the 4o scenes (`$BIG_ROOT/coarse_traj_4o`); verdict config = 4o summary + coarse run |
| `p8a_full.yaml`, `p8a_restart_b1.yaml`, `p8a_restart_b2.yaml`, `p8a_pilot.yaml`, `p8b_fused.yaml` | Pilots 8a/8b | M1 (≤8e-4, 96), M2 (≤4e-4, 96), M3 (≤2e-4, 125); B1 coarse shadow / B2 M1 shadow, M2 suffix |
| `p8d_certsim.yaml`, `p8d_pilot.yaml` | Pilot 8d | proposal M1, windows M3 (m = 4) at k = 3, 11, 19; threshold queries id_new_0..3 |
| `p8e_build.yaml`, `p8e_gen.yaml`, `p8e_full.yaml`, `p8e_scan.yaml`, `p8e_allk.yaml`, `p8e_stiff.yaml`, `p8f_cheapcheck.yaml` | Pilots E1–E3 | stiff_0/1 (75 objects each, E 10^U(7.5, 9.0)); `reference_cfl` 0.3; data in `$BIG_ROOT/datasets/stiff_e1` |
| `p8g_build.yaml`, `p8g_gen.yaml`, `p8g_full.yaml`, `p8g_scan.yaml`, `p8g_allk.yaml`, `p8g_stiff.yaml` | Pilot E1b | fresh stiff_2/3; M3 repair suffix; one-window primary certificate (`p8e_stiff.py` options `primary_windows`, labels, title) |
| `p9_*.yaml` (build, gen, full, scan, allk, confirm), `p9c_*.yaml` (comp_build, comp_gen, scene_build, scan, allk, full, summary) | Phase 9 | fresh id_new_22/23 + ood3_* (600) and contact draws 24–27; contact verdict via `p4n_summarize.py` (three-window certificate, amended before data; `abs_risk_limit: 0.03`) |
| `p4o_comp_build.yaml`, `p4o_comp_gen.yaml`, `p4o_scene_build.yaml`, `p4o_scan.yaml`, `p4o_allk.yaml`, `p4o_summary.yaml` | Phase 4o fresh draws 16–23 with the shadow-timed switch | `shadow_switch_dist: 0.46875`; `primary: sf` |
| `p4n_comp_build.yaml`, `p4n_comp_gen.yaml`, `p4n_scene_build.yaml`, `p4n_scan.yaml`, `p4n_allk.yaml`, `p4n_summary.yaml` | Phase 4n fresh draws 8–15: components / generations / scenes / coarse scan / restarts (chain, event, fidelity) / verdict | d_s = 0.46875 (3 coarse cells) |
| `p4l_scan_ref.yaml`, `p4l_summary_ref.yaml` | Phase 4l-ref scan / verdict | reference windows; τ_ref = 0.02894 (calibration pool, replicate 0) |
| `p10a_half.yaml`, `p10a_quarter.yaml`, `p10ac_half.yaml`, `p10ac_quarter.yaml`, `p10a_summary.yaml` | Phase 10a | reference setting at dt 5e-5 (all Phase 9 queries) and 2.5e-5 (first 25 per split / 7 per contact draw) |
| `p11_render.yaml`, `p11_judge.yaml` | Phase 11 | renders of the 25 images at force_coeff 0.06 / 0.40 (`fresh`); judge model and render roots |
| `p7b_testset_volume.yaml` | Phase 10b addition | PhysCtrl's own vIoU / Chamfer on the 7b outputs (earlier pipeline) |
| `p10c_rescore.yaml` | Phase 10c | decisions run, trajectory dir, the Phase 9 full/allk configs per set |
| `p10d_strands.yaml` | Phase 10d | paths only (simulation) |
| `p14a_build.yaml`, `p14a_gen.yaml`, `p14a_scan.yaml`, `p14a_train_{ACT,RND}_{F,L}.yaml`, `p14a_eval.yaml` | Phase 14a | candidate pool (sample/ objects × 30 `aug_` draws), its generations and scan, four training runs, development evaluation |
| `p14a2_train_{ACT,RND}_{Ld,Ltd}.yaml`, `p14a2_eval.yaml` | Pilot 14a-v2 | wall-free arms, LoRA (all attention r16 / temporal r8) with replay distillation; evaluation + verdict settings |
| `p15a_scenes.yaml`, `p15a_build.yaml`, `p15a_train.yaml`, `p15a_eval.yaml` | Phase 15a | scene sets (train 1,500 on sample/ objects, dev 200 on test objects), joint build, full fine-tune with object IDs, evaluation |
| `p10b_testset.yaml` | Phase 10b | Phase 7b with the final pipeline (`cert_windows: one`, M1 shadow, M2 suffix) → `p7b_summarize.py` |
| `p6_video_v2.yaml` | Paper re-render with the Phase 9 pipeline (job 95) | as p6_video + `v2`, M1 shadow, M2 suffix |
| `p6_video.yaml` | Phase 6a videos | 5 elastic examples × force {1, 3}; τ = 0.02614; outputs `$BIG_ROOT/renders` |
| `p4l_comp_build.yaml`, `p4l_comp_gen.yaml`, `p4l_scene_build.yaml`, `p4l_errors.yaml`, `p4l_scan.yaml`, `p4l_allk.yaml`, `p4l_summary.yaml` | Phase 4l components / their generations / scenes / errors / 3-source scan / every-frame restarts / verdict | outputs under `$BIG_ROOT` |
| `p4m_build.yaml`, `p4m_gen.yaml`, `p4m_scan.yaml`, `p4m_allk.yaml`, `p4m_calib_coarse.yaml`, `p4m_confirm.yaml` | Phase 4m fresh queries (ood2_E_low/E_high/force/drop, id_new_20/21) / generations / scan / all-k + saved coarse positions / calibration coarse rollouts / verdict | thresholds from the calibration pool |
| `p4k_build.yaml`, `p4k_gen.yaml`, `p4k_scan.yaml`, `p4k_gated.yaml` | Phase 4k fresh queries (id_new_12..19) / base generations + errors / coarse_cfl m4 scan / all-k + verdict | τ = 0.02614 fixed, τ_hi = 0.04501 |
| `p4j_confirm.yaml` | Phase 4j (job + verdict) | calibration pool; chain k ∈ {3, 7, 11, 15, 19}; shadow = 4d `coarse_cfl`; 500 splits; ε = α = 0.10 |
| `p4i_state.yaml` | Phase 4i | F-bound quantiles 0.1/99.9%; shadow = 4d `coarse_cfl`; α = 0.025 |
| `p4h_suffix.yaml` | Phase 4h | chain k ∈ {3, 7, 11, 15, 19}; τ = 0.02614; reference suffix |
| `p4f_confirm.yaml` | Phase 4f | frozen signal; ε = α = 0.10; 500 splits |
| `p4d_fullsim.yaml`, `p4d_pipeline.yaml`, `p4e_repair.yaml` | 4d data + analysis; 4e | see PLAN.md 4d/4e |
| `p4c_repair.yaml` | conditioning repair | 5 × 6 grid; selection signal; λ̂ 0.03007 |
| `p4b_certificate_objects.yaml`, `p4b_repair.yaml` | object-level certificate; repair | see the pre-registration in PLAN.md |
| `p4a_certificate.yaml` | `p4a_certificate.py` | the signals chosen on 2c; ε {0.05, 0.1, 0.2}; α {0.05, 0.1}; primary base defect, ε = α = 0.1 |
| `p2_summary.yaml` | `p2_summarize.py` | primary model large; floor ratio 0.25; rho 0.5; cost cap 0.2 |

## slurm/: you submit these from the project root

| File | What it does | GPU | Needs |
|---|---|---|---|
| `cluster.env` | `PROJECT_ROOT`, `DATA_ROOT` (filled Oct 10: `…/4`, `…/cvpr_physics_plausibility/data`); conda env, CUDA modules, cache dirs | — | — |
| `configure.sh` | Fills `<A100_ACCOUNT>`/`<A100_QOS>`/`<H200_ACCOUNT>`/`<H200_QOS>` in every header (done Oct 10: `a100_hbku`/`a100_hbku_qos`, `h200_hbku`/`h200_hbku_qos`) | — | values from `sacctmgr` |
| `preflight.sh` | Run on the login node before each submit: placeholders, env, imports, self-test, checkpoints, test set, `sbatch --test-only` on 01/02 | — | 00 |
| `_common.sh` | Sourced by every job: env, module, logging, placeholder and submit-dir checks, `log_config` | — | `cluster.env` |
| `00_setup_env.sh` | Interactive on a shared server: conda env, torch 2.5.1 cu121, PhysCtrl clone and requirements, checkpoints, test set, self-test, dry run | — | internet |
| `dry_run_all.sh` | All Phase 0–2 scripts on the synthetic backend (~10 min CPU) | none | — |
| `01_p0_check_A100.sh` | Phase 0 checks + PhysCtrl `eval.py` cross-check | 1 A100 | 00 |
| `02_p0_check_H200.sh` | Phase 0 checks on H200 | 1 H200 | 00 |
| `03_p1_cost_A100.sh` | Phase 1a timing | 1 A100 | 01 |
| `04_p1_cost_H200.sh` | Phase 1a timing | 1 H200 | 01 |
| `05_p1_fidelity_A100.sh` | Phase 1b sweep, array 0–3 | 4 A100 | 01 |
| `06_p1_summarize.sh` | Phase 1 verdict (run with `bash`, no GPU) | none | 03, 04, 05 |
| `submit_phase0_1.sh` | Submits 01–05 with dependencies | — | configured headers |
| `07_p2_build_ood_A100.sh` | OOD queries, array 0–3 | 4 A100 | Phase 1 GO |
| `08_p2a_lift_floor_A100.sh` | Lift floor, array 0–3 | 4 A100 | 07 |
| `09_p2b_generate_H200.sh` | Generations, array 0–1 | 2 H200 | 07 |
| `10_p2b_probe_A100.sh` | Probes, array 0–3 | 4 A100 | 09 |
| `11_p2_summarize.sh` | Phase 2 verdicts (run with `bash`, no GPU) | none | 08, 10 |
| `submit_phase2.sh` | Submits 07–10 with dependencies | — | Phase 1 GO |
| `submit_phase1.sh` | Re-submits Phase 1 only (03–05), for when Phase 0 already passed | — | results of 01 |
| `12_p2b_probe_v2_A100.sh` | Phase 2b-v2: probe rerun with `lift_noF`, array 0–3 | 4 A100 | 07–09 done |
| `14_p2c_scan_A100.sh` | Phase 2c scan, array 0–3 | 4 A100 | Phase 2 data |
| `15_p2c_summarize.sh` | Phase 2c verdicts (run here) | none | 14 |
| `32_p4_build_calib_H200.sh` → `33_p4_generate_calib_H200.sh` → `34_p4_scan_calib_H200.sh` | Phase 4a calibration data chain | 4 H200 | — |
| `submit_phase2c_4a.sh` | Submits 14 and the 32→33→34 chain together | — | — |
| `16_p2c_scan_resume_A100.sh` | Resumes 14 with the CFL gate and the `coarse_cfl` fidelity (keeps every row) | 4 A100 | rows of 14 |
| `36_p4_build_calib2_H200.sh` → `37_*` → `38_*`; `submit_phase4a2.sh` | Calibration extension chain (`id_new_4..11`) | 4 H200 | — |
| `46_p4g_build_train_A100.sh` → `47_p4g_train_H200.sh` → `48_p4g_repair_H200.sh`; `submit_phase4g.sh` | Phase 4g chain (training data → fine-tune → evaluation) | 4 A100, 1 H200, 4 H200 | — |
| `49_p4g_train_v2_H200.sh` → `50_p4g_repair_v2_H200.sh`; `submit_phase4g_v2.sh` | Phase 4g-v2 chain (reuses the 4g data) | 1 H200, 4 H200 | 4g data |
| `51_p4h_suffix_A100.sh`, `52_p4h_suffix_H200.sh` | Phase 4h suffix re-simulation, array 0–3 (A100, or H200; suffix costed as a share of the A100 reference rollout) | 4 A100 / 4 H200 | 2c scan |
| `72_p4o_comp_build_H200.sh` → `73_…` → `74_…` → `75_p4o_scan_allk_H200.sh`; `submit_phase4o.sh` | Phase 4o shadow-timed fidelity switch; done (CONFIRMED) |
| `76_fig_ours_traj_H200.sh` | Deck figures: our trajectories for the 5 qualitative cases (two MPM restarts, minutes) |
| `00c_download_p11_models.sh` → `98_p11_images_H200.sh` → `99_p11_render_H200.sh` → (frames on a dev server) → `100_p11_judge_H200.sh`; `submit_phase11.sh` (98 → 99) | Phase 11 video evaluation (≈ 15–20 GPU-h) |
| `101_p10c_rescore_H200.sh`, `submit_phase10c.sh` | Phase 10c exact re-scoring (≈ 8–9 GPU-h) |
| `102_p12_s0_cpu.sh` | Phase 12 gate S0 on `cpu-all` (env `somw-fem`; 32 cores, ≈ 3–5 h estimated; add an account line if cpu-all needs one) |
| `114_p12_test_cpu.sh` | Phase 12 test runs on `cpu-all` (4 shards; submit after 102) |
| `00d_download_pokeflex.sh format\|drops\|pokes` | PokeFlex download (dev server; link from `../PokeFlex/url.txt`; 5.2 / 25.9 / 562.5 GB) |
| `submit_now_oct29.sh` | submits everything ready on Oct 29: 102 → 114, Phase 15a, Phase 14a-v2 |
| `103_p10d_strands_A100.sh` | Phase 10d strand resolution study (≈ 1 A100-h, estimate) |
| `104_p14a_build_A100.sh` → `105_p14a_generate_H200.sh` → `106_p14a_scan_H200.sh` → (dev server: `p14a_acquire.py`) → `107_p14a_train_H200.sh` → `108_p14a_eval_H200.sh`; `submit_phase14a.sh [stage1|stage2]` | Phase 14a simulator-distillation pilot (≈ 12 GPU-h estimate; KILL Oct 24) |
| `109_p14a2_train_H200.sh` → `110_p14a2_eval_H200.sh`; `submit_phase14a2.sh` | Pilot 14a-v2 (≈ 7 GPU-h estimate) |
| `111_p15a_build_A100.sh` → `112_p15a_train_H200.sh` → `113_p15a_eval_H200.sh`; `submit_phase15a.sh` | Phase 15a contact-aware generator pilot (≈ 12 GPU-h estimate) |
| `96_p10a_convergence_H200.sh`, `97_p10b_testset_H200.sh`, `submit_phase10.sh` | Phase 10 (Oct 20): reference convergence (≈ 3.8 GPU-h); PhysCtrl's metrics with the final pipeline (minutes) |
| `94_fig_paper_traj_H200.sh`, `95_p6_video_v2_H200.sh`, `submit_paper_figs.sh` | Paper figures (Oct 19): Phase 9 cases recomputed + always-M1 rollouts (minutes); the 10 videos re-rendered with the Phase 9 pipeline (≈ 1 GPU-h) |
| `77_p7b_testset_H200.sh` | Phase 7b: shadow + five chain restarts per test-set query, positions saved, PhysCtrl DeformLoss on GPU (4 H200 × 2 shards, < 3 h) |
| `78_p7d_contact_coarse_H200.sh` | Phase 7d: coarse rollout of each 4o contact/control scene to the end, positions saved (4 H200 × 2 shards, < 3 h) |
| `79_p8a_fidelity_H200.sh` | Pilot 8a: medium full rollouts (2c + 4l) and gated-V2 restarts with a medium suffix (2c), 4 H200 × 2 shards, < 4 h |
| `80_p8d_certsim_H200.sh` | Pilot 8d: certify the cheap simulator M1 (4 H200 × 2 shards, < 4 h) |
| `81_p8e_build_H200.sh` → `82_p8e_generate_H200.sh` → `83_p8e_sims_H200.sh`; `submit_pilot_e1.sh` | Pilot E1: stiff build (CFL-safe reference), generations, ladder + scan + restarts (≈ 2–4 GPU-h) |
| `84_p8g_build_H200.sh` → `85_p8g_generate_H200.sh` → `86_p8g_sims_H200.sh`; `submit_pilot_e1b.sh` | Pilot E1b (round 2): fresh stiff draws, M3 repair (≈ 3 GPU-h) |
| `87_p9_build` → `88_p9_generate` → `89_p9_sims`; `90_p9c_comp_build` → `91_p9c_comp_generate` → `92_p9c_scene_build` → `93_p9c_sims`; `submit_phase9.sh` | Phase 9 confirmation (two independent chains, ≈ 7–9 GPU-h) |
| `68_p4n_comp_build_H200.sh` → `69_…` → `70_…` → `71_p4n_scan_allk_H200.sh`; `submit_phase4n.sh` | Phase 4n; done (NOT CONFIRMED: N1 FAIL by 0.004) | 4 H200 | — |
| `67_p4l_scan_ref_H200.sh` | Phase 4l-ref, the pre-registered H0 follow-up; done (H0 PASS) | 4 H200 | 4l scenes |
| `00b_download_render_models.sh` → `66_p6_video_H200.sh` | **Next:** render prerequisites on a dev server (pins kiui 0.3.4 + setuptools 75.8, caches SV3D on the share), then the Phase 6a video job (≈ 3.5 GPU-h) | dev server; 4 H200 | renderer weights |
| `62_p4l_comp_build_H200.sh` → `63_p4l_comp_generate_H200.sh` → `64_p4l_scene_build_H200.sh` → `65_p4l_scan_allk_H200.sh`; `submit_phase4l.sh` | Phase 4l contact benchmark (≈ 20 GPU-h); submitted Oct 15 | 4 H200 | — |
| `58_p4m_build_H200.sh` → `59_p4m_generate_H200.sh` → `60_p4m_scan_allk_H200.sh`; `61_p4m_calib_coarse_H200.sh`; `submit_phase4m.sh` | Phase 4m; done (M1/M2 PASS, XF does not dominate) | 4 H200 | — |
| `55_p4k_build_H200.sh` → `56_p4k_generate_H200.sh` → `57_p4k_scan_allk_H200.sh`; `submit_phase4k.sh` | Phase 4k chain (fresh queries → gated V2 test); done (CONFIRMED) | 4 H200 | — |
| `54_p4j_allk_H200.sh` | Phase 4j all-k V2 rollouts on the calibration pool, array 0–3; done (not confirmed: J2 FAIL) | 4 H200 | calibration generations, 4d |
| `53_p4i_state_H200.sh` | Phase 4i V1/V2 suffix restarts, array 0–3; done (V2 PASS, V1 FAIL) | 4 H200 | 4h rows, `results/p4i_f_bounds/` |
| `44_p4d_fullsim_A100.sh` | Phase 4d fallback rollouts, array 0–3 | 4 A100 | — |
| `45_p4e_repair_anch_H200.sh` | Phase 4e anchored repair, array 0–3 | 4 H200 | 4b rows |
| `43_p4c_repair_cond_H200.sh` | Phase 4c conditioning repair, array 0–3 | 4 H200 | 4b rows |
| `39_p4b_repair_A100.sh` | Phase 4b repair, array 0–3 (supersedes the `31_p4_repair_TODO` stub) | 4 A100 | Phase 2 generations |
| `35_p4_scan_calib_cfl_H200.sh` | Adds `coarse_cfl` to the calibration scan (keeps every row) | 4 H200 | 34 |
| `13_p2_summarize_v2.sh` | Phase 2-v2 verdict (run here, not on the login node) | none | 08, 12 |
| `20_p3_data_gen_TODO_A100.sh` … `42_p5_realworld_TODO_H200.sh` | **Stubs** (exit 1) listing the TODOs for Phases 3–5 | — | Phase 2 GO |

## Data and outputs (created by jobs)

| Path | Contents | Written by |
|---|---|---|
| `$DATA_ROOT/physctrl/` | PhysCtrl checkout (commit in `$DATA_ROOT/physctrl_commit.txt`) | 00 |
| `$DATA_ROOT/checkpoints/checkpoints/physctrl_{base,large}.safetensors` | Generator weights | 00 |
| `$DATA_ROOT/testset/` | Released test trajectories (HDF5) | 00 |
| `$DATA_ROOT/ood/<split>/*.h5` | OOD ground truth in PhysCtrl format | 07 |
| `$DATA_ROOT/generations/<model>/<split>/<qid>.npz` | 8 samples per query | 09 |
| `results/p0_check_{A100,H200}/` | `p0_report.json`, `testset_manifest.json`, `physctrl_eval/` | 01, 02 |
| `results/p1_cost_{A100,H200}/timings.jsonl` | Timing rows | 03, 04 |
| `results/p1_fidelity_A100/shard_*/fidelity.jsonl` | Sweep rows | 05 |
| `results/p1_summary/SUMMARY.md` | **Phase 1 verdict: send back** | 06 |
| `results/p2_build_ood/shard_*/` | OOD manifests and failures | 07 |
| `results/p2a_lift_floor/shard_*/lift_floor.jsonl` | Lift-floor rows | 08 |
| `results/p2b_probe/shard_*/probes.jsonl` | Probe rows | 10 |
| `results/p2_summary/SUMMARY.md` | **Phase 2 verdicts: send back** | 11 |
| `logs/` | Slurm stdout/stderr | all jobs |
| `results/_superseded/`, `logs/_superseded/` | Runs replaced by a fixed rerun (e.g. Phase 1 run 419227, which hit the CUDA-crash and batch-32 bugs). Kept for the record; never cited | — |
| `results/p7b_testset_volume/` | PhysCtrl's own vIoU / Chamfer / MSE on its test set (7b outputs): our run reproduces its reported 77.59% / 0.0028 / 0.0015 | `p7b_summarize.py` |
| `results/p9_table/{SUMMARY.md, table.json}` | Phase 9 main table (object-weighted) + certificate accept counts; `table.json` also has `max_gap` (Phase 10c, exploratory) | `p9_table.py` |
| `results/p10c_decisions/`, `results/p10c_rescore/shard_*/rows.jsonl` | Phase 10c Phase 9 decisions per query; re-score rows (reproduction deviations, unstable rollouts); positions in `$BIG_ROOT/p10c_traj/{single,contact}/` | `p10c_decisions.py`, 101 |
| `results/p10c_summary/{SUMMARY.md, summary.json}` | **Phase 10c verdict (Oct 21): T1 PASS, T2 FAIL, T3 M1/M3 PASS, M2 FAIL** | `p10c_summary.py` |
| `results/p10c_explore/{SUMMARY.md, explore.json}` | Phase 10c exploratory: gap to always-simulate and robust risk | `p10c_explore.py` |
| `results/p11_judge/scores.jsonl`, `results/p11_summary/` | **Phase 11 judge verdict (Oct 21): V0 FAIL → V1 uninformative**; Qwen2-VL-7B gives near-constant scores | job 100, `p11_summary.py` |
| `results/p11_study_design/mapping.json` | Study item → case and which letter is ours (local only) | `p11_study_build.py` |
| `results/p11_study_codes/codes.txt` | Answer codes received from raters, one per line (input to `p11_study.py --codes`); not analysed until the study closes | user |
| `results/p9_confirm/SUMMARY.md`, `results/p9c_summary/SUMMARY.md` | **Phase 9 verdicts (Oct 19): CONFIRMED** (single objects C1–C5; contact CC1–CC3). Inputs: `results/p9_{build,generate,scan,allk,full,errors}/`, `results/p9c_{comp_build,comp_generate,scene_build,scan,allk,full}/`; datasets in `$BIG_ROOT/datasets/{p9,mo_comp}` | 87–93; CPU |
| `results/p2b_probe/shard_*/cuda_crashes.jsonl`, `results/p2_build_ood/shard_*/failures.jsonl` | Units whose rollout lost the CUDA context; read on `--resume` so they are not retried | 10, 07 |
| `paper/` | CVPR LaTeX (2026 author kit, year set to 2027): `main.tex`, `sec/*.tex` (outline as orange [Plan] text), `OUTLINE.md` (claims, page budget, figures/tables, number sources), `build.sh` (tectonic); `tab/main_table.tex` and `fig/tradeoff.*` are generated by `code/p9_table.py`, `tab/reference_table.tex` and `supp/reference_table_full.tex` by `code/p10c_table.py` (do not edit by hand); `supp/moved_from_main.tex` holds text moved out of the 8 pages (amortisation, full negative list) for the supplement | — |

## Exported online material (`artifacts_export/`, Nov 3, 2026)

| Path | What it is |
|---|---|
| `artifacts_export/deck_initial_results/` | The 18-slide results deck (claude.ai artifact JcoKGKL4fA9avdYv5Fx2b6): `deck.json` + slide HTML; its images are `results/fig_qualitative/*.png` (matched by sha256) |
| `artifacts_export/deck_summary/` | The 6-slide summary deck (UB7gbAZ1omnnio4TwiKqZP); image `results/fig_qualitative/deck_videos.png` |
| `artifacts_export/study_page/` | The human-study page as published (EG8oUGHPCvmqrkJbkbKF8c, shared publicly): `index.html` (byte-identical to the live page), the 84 videos, `items.json`; mapping in `results/p11_study_design/` |

