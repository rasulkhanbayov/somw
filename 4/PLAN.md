# PLAN: Simulate Only What Matters

Plan for the strongest possible paper. Deadline pressure is ignored; the order is set so the cheap checks that can end the project run first. Phases 0–2 are fully scripted. Phases 3–6 are designed, but their scripts are stubs, because their design depends on what Phases 1–2 show.

Status: **nothing has been run.** Every number below is an estimate or a threshold, not a result.

Status update (Oct 10, 2026):
- CPU self-test passes 16/16.
- The synthetic dry run of Phases 0–2 passes end to end.
- `cluster.env` and the Slurm account/QOS values are still placeholders.
- Real deadlines (Nov 10/16/23) are now scheduled in **"Deadline schedule"** near the end of this file. That section overrides the "deadline pressure is ignored" stance above wherever the two conflict.

---

## What we learned from PhysCtrl's code before planning (Oct 2026)

These facts come from reading the repository (`notes/physctrl_ref_snapshot/`). They change the risk picture compared with the proposal.

| Fact | Consequence |
|---|---|
| The **base** generator samples with **4 DDIM steps** (8-layer DiT, latent 256). The **large** one uses 25 steps (12 layers, latent 512). | Generation is probably milliseconds to under a second. |
| The data simulator is Warp MPM with **substep 1e-4 s, 417 substeps per frame, 48 frames ≈ 20,000 substeps** on a 125³ grid. | At PhysCtrl's own resolution, simulation may be far slower than generation. That would be good for us, but Phase 1 must prove it, including against a cheaper simulator setting. |
| PhysCtrl's solver wraps each substep in `wp.ScopedTimer(synchronize=True)`, which forces 4 device syncs per substep. | Its as-shipped simulator is artificially slow. We must also time it without the syncs, and with CUDA graphs, or a reviewer will say we compared against a crippled simulator. |
| HDF5 data stores **x, v, F, C at all 49 frames**. | We can measure lifting error exactly against the true state (Phase 2a). |
| The generator predicts **24 frames = simulator frames 1, 3, …, 47**. Coordinates are normalised as (x − 5)/2. | Windows must start and end on odd frames. Velocity is lifted from samples 2 frames apart. |
| Elastic training data always has one drag force and **no gravity**. Its floor sits far below the object. | **Elastic data has essentially no contact events.** Contacts exist only in rigid / plasticine / sand data, and in our new `ood_gravity_drop` split. The contact-probe story needs that split, or the rigid track. |
| Only the **large** checkpoint takes gravity and floor conditioning. | `ood_gravity_drop` is evaluated with large. The primary model for the Phase 2 verdict is large. |
| Rigid data uses Genesis (49 frames, 20 substeps per frame), not Bullet. | RigidBench uses Blender's Bullet, so probing it needs a matching simulator or an explicit mismatch term (Phase 5b). |

Five silent-failure modes in reusing PhysCtrl's solver for restarts are guarded in `code/physctrl_bridge.py` (module docstring):
1. A checkpoint load with `strict=False` that matches no keys.
2. `F` versus `F_trial`.
3. Impulse masks re-selected from deformed positions.
4. Per-substep syncs.
5. The simulated frame length differing from the nominal one at coarse substeps.

---

## Phase overview

| Phase | Goal | Gate | Scripts | GPUs | Est. GPU-h* |
|---|---|---|---|---|---|
| 0 | Environment, data, replay, generator sanity | replay + keys | 00–02 | 1 A100 + 1 H200 | ~1 |
| 1 | **Is simulation expensive enough to save?** | **GO/NO-GO** | 03–06 | 4 A100 + 1 H200 | 5–15 |
| 2 | **Does a cheap window defect track true error?** | **GO/NO-GO** | 07–11 | 4 A100 + 2 H200 | 10–30 |
| 3 | Query stream, fine-tuning data, baselines at matched compute | Pareto | 20–21 (stubs) | 4 A100 (+ idle H200s) | 200–400 |
| 4 | Certificate (conformal) and repair (constrained re-sampling) | coverage, repair > replace | 30–31 (stubs) | A100 + 4 H200 | 100–250 |
| 5 | Online amortisation, RigidBench, real-world checks | probes/query falls | 40–42 (stubs) | mixed | 150–300 |
| 6 | Figures, videos, paper | — | — | H200 (rendering) | 30–60 |

\*Rough estimates written before any timing exists. Phase 1 replaces them with measured costs, and this table should be updated then.

---

## Phase 0: environment and reproduction

- **Goal.** Prove the stack works on both GPU types and that our wrappers reproduce PhysCtrl exactly.
- **Inputs.** PhysCtrl repo, `physctrl_{base,large}.safetensors`, `testset.zip` (all from HuggingFace `chenwang/physctrl`).
- **Outputs.**
  - `results/p0_check_{A100,H200}/p0_report.json`
  - `testset_manifest.json`, which every later phase reads
  - `physctrl_eval/` logs from PhysCtrl's own `eval.py`
- **Scripts.**
  - `00_setup_env.sh`: interactive, on a shared server.
  - `01_p0_check_A100.sh`, `02_p0_check_H200.sh`
- **GPUs.** 1 A100 (≤ 2 h wall), 1 H200 (≤ 1 h).
- **Checks.**
  - Re-simulate 8 stored elastic trajectories from frame 0 in all three timing modes. The error must stay below 0.01 simulator units at every frame.
  - Both checkpoints must load with **zero missing keys**.
  - Our bridge's model-coordinate MSE should match PhysCtrl's `eval.py` printout on the same items.
- **Falsifier.** Replay fails, or keys are missing.
- **If it fails.**
  - Replay failure means our solver setup differs from theirs: compare against `generate_mpm_data.py` line by line. Do not proceed. Phase 2 would then measure simulator mismatch, not model error.
  - Key mismatch: inspect `Generator` key prefixes.

### Phase 0 result (Oct 10, 2026, jobs 419225 A100 / 419226 H200): **PASS**

- **Replay:** stored trajectories re-simulate with a final-frame error of about 2e-6 (tolerance 0.01), on both GPU types and in all three timing modes.
- **Checkpoints:** both load with zero missing keys.
- **Bridge cross-check:** on the same 8 items, our generator's per-item MSE matches PhysCtrl's `eval.py` within ±3% (ratio 0.97–1.02, base and large). The remaining gap is `eval.py`'s bf16 autocast. `eval.py` per-item mean over 20 items: base 0.00197, large 0.00331 (model coordinates).
- **Test set:** 100 files, all elastic (no `mat_type` key; see `CLAUDE.md`).

## Phase 1: GO/NO-GO, the cost premise

- **Goal.** Decide whether "simulate only what matters" saves anything. This is the proposal's biggest risk, and our earlier review's main objection.
- **1a, timing** (`03_p1_cost_A100.sh`, `04_p1_cost_H200.sh`, 1 GPU each, ≤ 4 h).
  - Per-sample generation time: base and large, batch 1/8/32, warmed up, device-synchronised, model load excluded.
  - Simulator rollouts in 5 regimes × 3 modes:
    - Regimes: `physctrl` (2,048 particles); `dense10x` (20,480); `dense10x_fine` (20,480 on a 250³ grid); `multiobj2` and `multiobj3` (stacked objects colliding under gravity).
    - Modes: `as_is` / `nosync` / `graph`.
  - Probe-window cost for windows of 1 and 2 model frames.
- **1b, fidelity sweep** (`05_p1_fidelity_A100.sh`, 4 × A100 array, ≤ 8 h).
  - 5 substep sizes × 3 grids on 40 test queries, measuring error against ground truth, cost and stability.
  - Generator error on the same queries.
  - This finds the **cheapest simulator as accurate as the generator** (the matched-accuracy cost). It pre-empts "PhysCtrl's simulator setting is overkill".
- **First run (419227–419229) was invalid**, and is moved to `results/_superseded/`.
  - Generation at batch 32 exceeds the attention launch limit, so the batch sizes are now 1/8/31.
  - Coarse-substep rollouts (dt ≥ 8e-4) sometimes blow up and corrupt the CUDA context, which poisoned the rest of each shard. These are now handled with exit 75 and `--resume`.
  - Resubmit with `slurm/submit_phase1.sh`.
- **Decision** (`bash slurm/06_p1_summarize.sh` → `results/p1_summary/SUMMARY.md`).

### Phase 1 result (Oct 10, 2026, jobs 419234–419236): **GO by the rule, but narrow**

Full tables are in `results/p1_summary/SUMMARY.md`. Measured numbers:

**Only one regime passes:** `dense10x` with the base model, at 22.4× (A100) and 27.1× (H200). `dense10x_fine` would also pass with base (97×/110×), but it is not in the rule's realistic set.

**Caveats** (rule unchanged; these are threats a reviewer will raise):
1. **`dense10x` compares unlike outputs.**
   - The simulator runs 20,480 particles, but the generator outputs 2,048 points; the upsampling cost is not counted.
   - `dense10x` is charged at the reference simulator setting. The fidelity sweep was run only for `physctrl`, where the matched-accuracy setting (dt 8e-4, grid 64) costs ×0.13 of the reference.
   - If a similar factor holds for `dense10x`, its ratio would be about 3×. That is an extrapolation, not measured.
2. **In the paper's actual regime (`physctrl`, 2,048 points, matched accuracy):**
   - Simulation takes 0.547 s on A100 and 0.395 s on H200.
   - That is only 2.3×/2.8× base generation (0.234/0.140 s per sample).
   - It is **0.1×/0.2× of large generation** (4.52/2.53 s per sample).
   - So the proposal's "simulation is cheap anyway" risk is largely realised for PhysCtrl elastic.
3. **Simulation cost scales with substep count, not particle count.** `physctrl`, `multiobj2` and `multiobj3` all take about 4.3 s on A100 in graph mode.
4. **Accuracy:**
   - The large generator is less accurate than base on the elastic test set: median error 0.119 against 0.052.
   - Even the matched-accuracy simulator (0.016) is about 3× more accurate than base.
5. **Probe cost:** one probe window at the reference setting costs 0.307 s (1 frame) or 0.484 s (2 frames) on A100. That is 7–11% of a reference rollout, but **56–88% of a matched-accuracy rollout**.
6. **Stability:** coarse settings blow up on some objects. At dt 1.6e-3 / grid 125, only 55% of rollouts are stable; 12 rollouts crashed in total.

**Implication:** the efficiency claim (contribution 3) is weak in the PhysCtrl elastic regime. Phase 2 (the verifier) decides whether the certificate (contribution 1) can lead the paper. Choose the framing with the user after Phase 2, as scheduled for Oct 18.
  - Rule: **GO if, on both A100 and H200, some realistic regime has simulation ≥ 10× generation per sample.**
  - In the `physctrl` regime the simulator is charged at its matched-accuracy setting, with the speed-up measured inside the sweep.
  - Multi-object generation is charged as n × single-object, a lower bound that favours generation.
- **Falsifier.** No realistic regime reaches 10× on both GPUs.
- **If NO-GO.**
  1. If the ratio is ≥ 10× only for `dense*` or `multiobj*`, GO with that regime as the main benchmark. Multi-object then needs a generator that handles several objects (Phase 3 fine-tuning), and the proposal already lists multi-object as "where the method should help most".
  2. If no regime passes, the efficiency claim is dead. Re-centre the paper on contribution 1 alone: a residual-free certificate for generated physics, possibly scoped as a study of when a physics diffusion prior can be trusted. Then re-scope Phases 3–5 with the user before spending more compute.

## Phase 2: GO/NO-GO, the verifier

- **Prep: OOD queries** (`07_p2_build_ood_A100.sh`, 4 × A100 array, ≤ 4 h).
  - 100 released elastic test objects re-simulated with E = 10^3.5, E = 10^7.5, ×3 force, and a gravity drop onto a floor (the contact split).
  - Same Warp MPM as the training data. No meshes or Objaverse download needed.
- **2a, lift floor** (`08_p2a_lift_floor_A100.sh`, 4 × A100, ≤ 12 h).
  - Run the verifier on **ground-truth** trajectories with 7 variants:
    - oracle state;
    - lift from odd frames (what a generator gives);
    - lift from all frames;
    - no C; no F;
    - k = 6 and k = 24.
  - Record raw and chaos-normalised defects, plus lifted-state errors against the stored v, F, C. The `lift` floor is the smallest model error the verifier can see.
- **2b, does the defect track error?**
  - `09_p2b_generate_H200.sh` (2 × H200, ≤ 4 h): 8 samples per query from both models, for 100 in-distribution + 4 × 50 OOD queries.
  - `10_p2b_probe_A100.sh` (4 × A100, ≤ 16 h): computes per query
    - the true error;
    - baselines: seed spread across the 8 samples, and motion magnitude;
    - window defects under 6 placements (uniform, random, event, disagree, the proposal's anchor + events + random, and an oracle ceiling), with 3 windows of 2 simulator frames each.
  - Every signal carries its simulator cost.
- **Blow-up rule (pre-registered Oct 10, 2026, before any Phase 2 result; chosen by the user).**
  - A window whose rollout blows up (CUDA crash, non-finite positions, or particles outside the grid) is scored **defect = +∞**, an automatic reject.
  - A blown main rollout gives raw = norm = +∞. A blown perturbed rollout gives norm = +∞.
  - +∞ stays in the medians and in the Spearman ranks (it ranks last).
  - The gate verdict uses this scoring. A secondary column that excludes blow-ups is reported for comparison only, and never decides the gate.
  - Implemented in `p2_summarize.apply_blowup_rule` and `verifier.blew_up`.
- **Decision** (`bash slurm/11_p2_summarize.sh` → `results/p2_summary/SUMMARY.md`).
  - **G2a.** The lift floor (in-distribution) is ≤ 25% of the defect on generated in-distribution trajectories (model large).
  - **G2b.** Some non-oracle probe signal meets all three:
    - pooled Spearman ≥ 0.5 with true error;
    - a significant paired-bootstrap advantage over seed spread on OOD queries;
    - cost ≤ 20% of a full rollout.
  - **GO** = G2a and G2b.
  - "GO-CERTIFICATE-ONLY" means the signal works but costs more than 20%.
- **Falsifiers and what to do.**
  - **G2a fails** (the verifier is blind below its own lifting error).
    - Try longer windows, which grow model error faster than lift error.
    - Try a lift from a short simulated warm-up: restart 1 frame earlier and discard it.
    - If nothing helps, the window verifier is not viable on positions-only output. The fallback is the proposal's "calibration study" paper.
  - **G2b fails on rank correlation.**
    - Report it. A clean negative ("simulator windows do not beat free seed spread") is useful.
    - Then check whether the event and contact placements help on `ood_gravity_drop` specifically, which is where the proposal's thesis lives.
  - **G2b passes only above the cost cap.**
    - Drop chaos normalisation (n_perturb = 0) if raw defects are nearly as good. The summary reports raw signals charged without perturbation runs.
    - Otherwise lead with the certificate.
  - **Seed spread wins outright.** Then the paper's novelty must come from repair and amortisation, which are untested. Discuss with the user before Phase 3.

### Phase 2 result (Oct 10, 2026, jobs 419290–419296): **NO-GO under the pre-registered rule; verifier defect found**

Full tables are in `results/p2_summary/SUMMARY.md`.

**Data built:**
- OOD queries: 398 of 400. The 2 that failed were `ood_E_low` rollouts that went unstable during the build.
- Generations: 600 (300 queries × 2 models). Probe rows: 600.

**Verdict:**
- **G2a passes only trivially:** the median defect on generated trajectories is +∞.
- **G2b fails:** the best window signal reaches Spearman 0.21 overall, against the 0.5 needed, and every window signal loses to seed spread on OOD.

**Cause: the F-lifting step makes the simulator blow up, even on ground truth.**
- Every lift variant that estimates F blew up (NaN) on 36–58% of in-distribution **ground-truth** windows: `lift` 220/480, `lift_k6` 277/480, `lift_k24` 172/480.
- `lift_noF` (F = I) and `oracle_state` never blew up: 0 of 840 rows each.
- In the probes, 85–97% of rows per split had a blown main rollout.
- On ground truth the median lifted-F error is small (0.03), but the tail is severe: a few particles come out inverted (det F down to −0.69) or near-singular (singular value 0.03). Five particles had F error above 1 in one window at frame 25.
- Blow-ups grow with start frame (24 at frame 1, 42 at frames 25–41). The elastic model is PhysCtrl's fixed-corotated (FCR) model.
- By the pre-registered rule these windows score +∞, so this **run measures the lift defect, not the verifier idea**.

**Findings that do not depend on the defect:**
- **Seed spread of the large model is a strong, free error predictor:** Spearman 0.82 overall, 0.85 OOD, 0.72 ID.
- **Base seed spread only works in distribution:** 0.82 ID, 0.05 OOD.
- **True error:** base is better in distribution (0.052 vs 0.114), but fails on `ood_gravity_drop` (0.401 vs large 0.120).
- **Lift floor on ground truth, in-distribution, m = 1:** `lift_noF` raw 0.0014, `lift` 0.0024, `lift_k24` 0.0020.
- **Probe cost:** with chaos normalisation it is 0.31–0.375 of a reference rollout, above the 0.2 cap. Raw-only cost is 0.10–0.125.

**Caution:** the secondary column (blow-ups excluded) has been seen. It covers a heavily selected subset (about 10–15% of rows), so it must not drive any choice.

**Next step (approved by the user, Oct 10; pre-registered before any v2 result): Phase 2b-v2.**
- A probe rerun with `lift_noF` (F = identity on restart), chosen on ground-truth evidence only.
- Config: `configs/p2b_v2.yaml`; script: `slurm/12_p2b_probe_v2_A100.sh`, which reuses the v1 generations.
- Verdict: `slurm/13_p2_summarize_v2.sh` with `configs/p2_summary_v2.yaml`.
- G2a now compares against the `lift_noF` floor, the lift the probes actually use.
- Thresholds (0.25 / 0.5 / 0.2), the primary model (large) and the +∞ blow-up rule are unchanged.
- v1 stays reported as run, and the paper must disclose the v1 → v2 change.

### Phase 2-v2 result (Oct 11, 2026, job 419387, `lift_noF`): **GO under the pre-registered rules**

Full tables are in `results/p2_summary_v2/SUMMARY.md`. All 600 probe rows completed with 0 CUDA restarts, and only 1–7 of 600 rows per strategy had a blow-up. The blow-up rule no longer matters: the secondary column equals the primary.

**Gates:**
- **G2a passes:** the `lift_noF` floor is 0.0014.
  - Large generated, in-distribution: 0.0151 (ratio 0.095).
  - Base: 0.0114 in-distribution (0.12) and 0.0191 OOD (0.07).
- **G2b passes through the base model only.** All 10 raw window signals pass: Spearman 0.80–0.86 overall, cost 0.10–0.125.
  - In-distribution rho is 0.67–0.73, OOD 0.77–0.86.
  - They beat base seed spread on OOD by +0.72 to +0.81 (95% CI lower bounds above +0.55). Base seed spread collapses OOD (0.05).

**What the data say for the paper:**
1. **Residual-free verifier (contribution 1): supported for base.** Short exact-simulator windows catch out-of-distribution failures that the model's own sample spread misses entirely.
2. **Large model: windows do not beat seed spread.** Window Spearman is 0.45–0.61, against seed spread 0.82, and windows lose on OOD by −0.21 to −0.47. Large's own uncertainty is already informative. Combining the two signals is untested.
3. **Placement does not matter (bad for contribution 2).** For base, uniform ≈ random ≈ event ≈ disagree ≈ proposal ≈ even oracle placement (0.79–0.86). The defect looks like a global property of a trajectory that any window sees, so event-aware placement adds nothing here.
4. **Chaos normalisation adds nothing.** Norm signals equal raw ones, or are worse for proposal (0.56–0.64), at 3× the cost (0.31–0.375). Drop it.
5. **Cost vs matched-accuracy simulation (bad for contribution 3).** One probe set (3 windows of 2 frames) costs 0.125 of a reference rollout. A full rollout at the matched-accuracy setting (Phase 1: dt 8e-4, grid 64) costs 0.127, and is about 3× more accurate than base (error 0.016 vs 0.052). So at reference fidelity, probing costs about as much as simulating everything coarsely.
   - Untested fix: run the probes at the coarse setting too (about 0.016 of a reference rollout). This needs a new lift floor, because the defect would then include discretisation error.

### Diagnosis and improvement plan (Oct 11, 2026)

The full analysis is in `notes/improvement_analysis_2026-10-11.md`; the diagnostics are in `results/diag_p2/DIAG.md` (exploratory).

In short:
- **Placement:** in drag-only splits the error is global, so placement cannot matter there by construction. In the contact split the error *is* localised, but the event detector read events from the generation, which misses them (0% coverage for base). The v1 oracle was mis-specified.
- **Cross-split scale:** the pooled base advantage comes from the defect's absolute scale across conditions. That is the certificate's real strength, and it should be evaluated by single-threshold AUROC and conformal coverage.
- **Large model:** its errors drift smoothly and affinely, which favours longer windows and fusion with seed spread.
- **Cost:** it is set by substep count. Probes have not yet been run at matched accuracy, and windows are not yet packed or run in parallel.

### Phase 2c: exhaustive window scan (pre-registered Oct 11, 2026, before any scan result)

- **Run:** `code/p2c_scan.py` (`configs/p2c_scan.yaml`, `slurm/14_p2c_scan_A100.sh`).
  - Every candidate window plus the anchor, for base, large and ground truth (the lift floor), at `ref` and `coarse` (dt 8e-4, grid 64) fidelity, for m = 1, 2, 4.
  - Restarts use `lift_noF`; the queries and generations are Phase 2-v2's.
- **Verdict:** `code/p2c_summarize.py` (`configs/p2c_summary.yaml`, `slurm/15_p2c_summarize.sh`). Blow-ups score +∞.
- **A. Fidelity (contribution 3).** At coarse, m = 1, uniform-3:
  - floor / generated defect (in-distribution) ≤ 0.25 for both models;
  - base pooled Spearman ≥ 0.5;
  - and Spearman ≥ the reference value − 0.05.
- **B. Window length and fusion (contribution 1, large).** Reference fidelity, uniform-3, primary m = 2. Fusion = mean rank of seed spread and defect. PASS if, for large, fusion beats seed spread in pooled Spearman and on OOD (paired bootstrap, 95% lower bound > 0).
- **C. Placement on contacts (contribution 2).** `ood_gravity_drop`, reference, m = 1, k = 3. capture = strategy defect / exhaustive defect. PASS if `event_union` beats uniform (paired bootstrap, lower bound > 0) for each model. Drag-only splits are reported, not judged.
  - `event_union` adds the physics-prior contact frame, predicted from gravity, floor and the initial points only (`verifier.prior_contact_frame`), weighted +3 over a ±1-frame band.
- **D. Single-threshold certificate (contribution 1).** Pooled over all splits, reference, m = 1, uniform-3. AUROC for err_mean > ε, ε ∈ {0.05, 0.10, 0.20}, primary 0.10. PASS if AUROC(defect) − AUROC(seed spread) has 95% lower bound > 0 for base.
- **Cost (estimate):** about 10 A100-h, about 3 h wall on 4 A100.

**Phase 2c run 1 (job 419393, Oct 11): incomplete.**
- **What happened:** all 4 shards stopped at the 50-restart cap, crash-looping on `ood_E_high` at the fixed coarse setting.
- **Why:** dt 8e-4 on grid 64 violates the explicit-MPM CFL condition for E = 10^7.5.
  - Every crashing query had CFL 1.18–1.76 (11 queries, 204 windows).
  - Every query with no blow-up had CFL ≤ 1.03.
  - The reference setting has CFL ≤ 0.43 on every query.
- **Completed before the stop:** in-distribution (ref and coarse), plus part of `ood_E_high`, 2,607 of 5,400 rows. No verdict has been computed.
- **This is itself a result for contribution 3:** a fixed matched-accuracy setting chosen in distribution is not safe for stiffer materials.

**Fixes (Oct 11), made before any Phase 2c verdict was computed:**
1. **CFL gate.** `cfl_gate: 1.15` (between 1.03 and 1.18). Windows above it are recorded as blown and scored +∞ without running, which is what the pre-registered +∞ rule gives a blow-up. The gated windows are counted in the summary. Question A is unchanged.
2. **Question A′ (new; pre-registered before any of its results).** The same criteria as A, for fidelity `coarse_cfl`: grid 64 with per-query dt = min(8e-4, 0.5·dx/c), where c is the dilatational wave speed. This is the standard explicit-MPM choice, made from stability theory.
3. **Resume jobs:**
   - `slurm/16_p2c_scan_resume_A100.sh` keeps every row (`--resume` from the first start), has the restart cap raised to 300, and must use the same 4 shards.
   - `slurm/35_p4_scan_calib_cfl_H200.sh` adds `coarse_cfl` to the calibration scan.
   - Estimate: about 7 GPU-h of simulation in total (measured per-unit time: 11.2 s ref, 2.0 s coarse).

### Phase 4a data: held-out calibration set (runs in parallel with Phase 2c, on H200)

**Status (Oct 11): complete.** Jobs 419394–419396: 400 queries, 800 generations, 800 error rows, the full scan (4,800 rows; 13 crash windows, all coarse).

- **Queries:** the 100 test objects × 4 fresh in-distribution physics draws (`id_new_0..3`, PhysCtrl's training ranges) = 400 queries.
- **Jobs:**
  - `slurm/32_p4_build_calib_H200.sh` builds the queries;
  - `slurm/33_*` generates base and large samples;
  - `slurm/34_*` computes true error and seed spread, and runs the same scan.
- **Split:** calibration and test halves are split by object.
- **Protocol:** the signal is chosen on Phase 2c, then confirmed and calibrated (split-conformal) here. That analysis code is written after Phase 2c, because only the choice of signal depends on 2c.
- **Submit both:** `bash slurm/submit_phase2c_4a.sh`.

### Phase 2c result (Oct 11, 2026; jobs 419393 + resume 419566): verdicts under the pre-registered rules

Full tables are in `results/p2c_summary/SUMMARY.md`. Scan complete: 8,100 rows. The CFL gate skipped 1,017 fixed-coarse `ood_E_high` windows, scored +∞.

| Q | Verdict | Key numbers |
|---|---|---|
| A (fixed coarse dt) | **FAIL** | Base Spearman 0.62 vs 0.82 at reference. The stiff split is gated to +∞ (CFL violation). |
| A′ (CFL-adaptive coarse) | **PASS** | Base 0.81 vs 0.82. Floor ratio 0.23 base, 0.18 large. Cost 1/8 of reference. |
| B (large, primary m = 2) | **FAIL** | Fusion 0.80 vs seed spread 0.82. At m = 4 (secondary), fusion 0.86, OOD +0.04 [+0.02, +0.07]. The defect improves with m: large 0.54/0.64/0.82, base 0.82/0.86/0.91. |
| C (placement on contacts) | **PASS** | `event_union` − uniform capture: base +0.01 [+0.00, +0.01], large +0.17 [+0.06, +0.27]. Plain generation `event` for large: 0.96 vs uniform 0.61. Drag-only: uniform is best (0.98–0.99). |
| D (single threshold, base) | **PASS** | AUROC 0.911 vs seed spread 0.616 at ε = 0.10 (+0.30 [+0.21, +0.38]). Large: seed spread wins (0.924 vs 0.733). |

**Correction to the v2 G2a margin.** Measured like for like (max over the same 3 windows on ground truth and on the generation), base's reference floor ratio is 0.27, above the 0.25 limit. v2's 0.095 compared a per-window median against a max over 3 windows. The CFL-adaptive coarse setting is under the limit for both models.

**Selection-set table:** `results/p2c_select/SELECT.md` (`code/p2c_select.py`; exploratory). The `spread_event` placement there is defined in that file: one window at the strongest generation event, plus two at the 1/3 and 2/3 points of the rest.

### Phase 4a: conformal certificate on the held-out calibration set (pre-registered Oct 11, 2026, before any analysis of the calibration data)

**Signal choice**, by a rule fixed now: per model, the cheapest configuration whose Phase 2c pooled AUROC (err > 0.10) is within 0.01 of the best.
- **Base:** `coarse_cfl`, m = 4, `spread_event`, max raw defect. Selection-set AUROC 0.958 (best 0.968); cost 0.062 of a reference rollout.
- **Large:** seed spread alone (0.924; best 0.934). Secondary: fused, `ref`, m = 4, `spread_event`, where fusion = mean of the calibration-half empirical CDFs of seed spread and defect (a fixed function once calibrated).

**Procedure:** class-conditional split conformal on bad trajectories.
- Bad means err_mean > ε. The calibration set is split 50/50 **by object**, seed 0.
- λ̂ = the k-th smallest score among bad calibration trajectories, with k = ⌊α(n_bad + 1)⌋. Accept if score < λ̂; if k = 0, accept nothing.
- Guarantee: P(accept | bad) ≤ α, under exchangeability (in distribution).
- Primary: ε = 0.10, α = 0.10. Also reported: ε ∈ {0.05, 0.20}, α = 0.05.

**Checks:**
- **P1 (validity, in distribution):** on the test half, the false-accept rate among bad trajectories. PASS if its Clopper–Pearson 95% interval includes values ≤ α. The point estimate is reported.
- **P2 (power, base):** acceptance rate among good trajectories on the test half, against a seed-spread certificate built the same way. PASS if the defect certificate accepts more good trajectories (paired bootstrap over objects, 95% lower bound > 0).
- **P3 (shift, empirical; no guarantee claimed):** apply the in-distribution λ̂ to the Phase 2c splits. Report the false-accept rate among bad trajectories and the acceptance rate among good ones, per split.

### Phase 4a result (Oct 11, 2026): primary PASS (P1 weakly, P2 clearly); a strong shift result

Full tables are in `results/p4a_certificate/SUMMARY.md` (`code/p4a_certificate.py`, `configs/p4a_certificate.yaml`). Self-test 22/22, including a finite-sample check of the conformal threshold.

**Primary: base defect certificate, ε = 0.10, α = 0.10.**
- **P1 validity: PASS by the pre-registered rule, but weak.** The false-accept rate is 0.194 (7/36), Clopper–Pearson [0.082, 0.360]. The point estimate is about 2× α, with only 25 bad calibration trajectories.
- **P2 power: PASS.** Good trajectories accepted: 0.689 vs 0.488 for a seed-spread certificate, +0.20 [+0.13, +0.27]. At α = 0.05: +0.12 [+0.07, +0.16].
- **P3 shift, using the in-distribution λ̂ (empirical):**

  | Split | Failures accepted: window certificate | Failures accepted: base seed-spread certificate (comparator, added after the primary result) |
  |---|---|---|
  | `ood_gravity_drop` | 0/50 | **50/50** |
  | `ood_E_low` | 0/32 | 0/32 |
  | `ood_force_x3` | 0/36 | 0/36 |
  | `ood_E_high` | 1/5 | 1/5 |

  Good trajectories accepted in distribution (`test_id`): 0.64 vs 0.45.

**Large:**
- The seed-spread certificate is valid and conservative (false-accept rate 0.027 at α = 0.1) and stays ≤ 0.05 on every shifted split.
- The fused certificate (secondary) is valid. It gives more power at α = 0.05 (+0.13 [+0.05, +0.21]) and at ε = 0.2 (+0.26), but not at the primary setting (+0.03 [−0.03, +0.11]).

**Limitations to fix next:**
1. **Exchangeability is at the object level.** The 4 physics draws of one object are correlated. Calibrate per object instead: one random draw per object, or a cluster-conformal score.
2. **Few bad calibration trajectories at strict ε.** For base at ε = 0.2 there are 8, so k = 0 and nothing is accepted. More calibration data is needed (cheap: more `id_new_j` draws).

### Phase 4a extension: object-level certificate (pre-registered Oct 11, 2026, before its data existed)

- **Data:** 8 more in-distribution draws per held-out object (`id_new_4..11`), giving 12 per object.
  - Configs: `configs/p4_calib2_*.yaml`. Jobs: `slurm/36` → `37` → `38` (H200), submitted together by `slurm/submit_phase4a2.sh`.
  - The scan is restricted to m = 4 at `ref` and `coarse_cfl`, the only configurations the certificates use.
  - The PhysCtrl `sample/` set (100 more objects, no overlap with the test set) is **not** used: it is drawn from PhysCtrl's training data.
- **Analysis:** `code/p4b_certificate_objects.py` (`configs/p4b_certificate_objects.yaml`). The guarantee is P(accept | a bad draw of a NEW object) ≤ α.
  - **Primary:** single-draw subsampling, one random bad draw per calibration object (exact).
  - **Secondary:** object-weighted pooling, weight 1/n_j per draw (approximately valid).
  - Repeated over 500 random 50/50 object splits; test rates are object-weighted.
  - Signals are frozen as in Phase 4a.
- **Checks** (primary: base defect, ε = α = 0.10):
  - **P1′:** mean false-accept rate over replicates ≤ α + 0.01.
  - **P2′:** good-draw acceptance minus that of the seed-spread certificate; 2.5th percentile > 0.
  - **P3′:** the replicate-0 λ̂ on the Phase 2c splits (empirical).

### Phase 4b: training-free repair (pre-registered Oct 11, 2026, before any repair result)

- **Code:** `code/p4b_repair.py` (`configs/p4b_repair.yaml`, `slurm/39_p4b_repair_A100.sh`); verdict from `code/p4b_summarize.py`.
- **Constraints:** the anchor window (true initial state) plus the certificate's `spread_event` windows (m = 4), simulated at `ref`.
- **Methods:**
  - R0: original generation.
  - R0p: re-sample alone.
  - R1: replace the verified frames.
  - **R2: constrained re-sample** (x₀ replacement inside PhysCtrl's own DDIM loop: `Generator.sample_constrained`).
  - R3: anchor constraints only.
- **Equivalence check:** before any repair, the job checks that our unconstrained loop reproduces PhysCtrl's pipeline for the same seed (tolerance 1e-3).
- **Primary (falsifier):** base trajectories rejected by the 4a primary certificate (R0 defect ≥ λ̂ = 0.03007). PASS if mean error R1 − R2 has a 95% paired-bootstrap lower bound > 0.
- **Secondary:**
  - R2 vs R0p (constraints vs re-sampling alone);
  - R3 vs R2;
  - backward propagation (error on frames between the anchor and the first non-anchor constraint, where R1 = R0);
  - unconstrained frames only;
  - acceptance after repair;
  - all base and all large trajectories.

### Phase 4a extension result (Oct 11, 2026; jobs 419953–419955)

Full tables are in `results/p4b_certificate_objects/SUMMARY.md`. The data: 2,400 trajectories over 100 held-out objects, 500 object splits.

**Primary: base defect, ε = α = 0.10, single-draw subsampling.**
- **P1′ validity: PASS.** Mean false-accept rate 0.084, target ≤ 0.11. This fixes the Phase 4a exchangeability concern.
- **P2′ power: FAIL.** Mean good-draw acceptance is 0.644 vs 0.531 for seed spread (+0.114), but the 2.5th percentile over splits is −0.104. Pooled: +0.097 [−0.024, +0.228]. In distribution, the defect certificate is not reliably more powerful than seed spread, consistent with Phase 2c's within-split results.
- **P3′ shift** (λ̂ = 0.0261), failures accepted:

  | Split | Accepted |
  |---|---|
  | `ood_gravity_drop` | 0/50 |
  | `ood_E_low` | 0/32 |
  | `ood_force_x3` | 0/36 |
  | `ood_E_high` | 1/5 |
  | in-distribution (`test_id`) | 3/22 |

**Large:** the seed-spread certificate is valid (mean false-accept rate 0.103 at α = 0.10). Under shift it accepts 3/26 gravity-drop failures; the fused certificate accepts 0/26.

**Supported claim:** a valid object-level certificate whose advantage is robustness under shift, not in-distribution power.

### Phase 4b result (Oct 11, 2026; job 419956): primary PASS, small effect

Full tables are in `results/p4b_summary/SUMMARY.md`. The equivalence check (our loop vs PhysCtrl's pipeline) gave a difference of exactly 0 on all 4 shards. There were 600 rows and no blown constraint windows.

**Rejected base trajectories (n = 198), mean error:**

| Method | Error |
|---|---|
| R0 original | 0.2671 |
| R0p re-sample alone | 0.2674 |
| R1 replace | 0.2300 |
| **R2 constrained** | **0.2276** |
| R3 anchor only | 0.2591 |

- **Primary: R1 − R2 = +0.0024 [+0.0017, +0.0031]. PASS, but the effect is about 1%.**
- **Propagation:** unconstrained frames 0.3139 → 0.3072 (2%); frames before the windows 0.1394 → 0.1378 (1%).
- **Accepted after repair:** R1 0.28, R2 0.17.
- **Large:** R1 − R2 = −0.0008 [−0.0027, +0.0012]; frames before the windows are slightly worse.
- **Interpretation:** x₀ replacement barely moves the generator's global response error. Most of the repair gain is the verified frames themselves.
- **Candidate stronger repairs** (each to be pre-registered before it runs):
  - RePaint resampling jumps;
  - reconstruction guidance (the gradient of the constraint loss through the denoiser);
  - **conditioning repair:** search the generator's physical inputs (E or force scale) so that its output matches the verified windows, then re-generate. This targets the diagnosed global-response error directly.

### Phase 4c: conditioning repair (pre-registered Oct 11–12, 2026, before any result; user-approved)

- **Code:** `code/p4c_repair_conditioning.py` (`configs/p4c_repair.yaml`, `slurm/43_p4c_repair_cond_H200.sh`); verdict from `code/p4c_summarize.py`.
- **Idea:** the generator's error is a global wrong response (`results/diag_p2`), so change its *conditioning*, not its frames.
- **Candidates:** for each Phase 2c query and model, 30 candidates from **shared noise** (the Phase 4b repair seed): Δlog₁₀E ∈ {−1, −0.5, 0, +0.5, +1} × force scale ∈ {0.33, 0.5, 0.75, 1, 1.5, 2}, applied to the model's log₁₀E and force inputs. Implemented as `Generator.sample_constrained(adjust=…, shared_noise=True)`.
- **Selection:** argmin of max(certificate defect [`coarse_cfl`, m = 4, `spread_event`], anchor mismatch [model frames 0–3 vs the anchor window simulated from the true initial state at `ref`]). The identity comes first, so ties go to it. Ground truth is never used.
- **Equivalence check:** identity conditioning must reproduce PhysCtrl's pipeline (tolerance 1e-3).
- **Primary (falsifier):** rejected base trajectories, the same population as 4b. PASS if mean error R1 − R4 has a 95% paired-bootstrap lower bound > 0.
- **Secondary:**
  - R2 − R4, R0 − R4;
  - R4id − R4 (the value of selection beyond re-sampling);
  - chosen adjustments per split;
  - **held-out defect** (uniform windows, `ref`, m = 4; never used in selection);
  - certificate-signal acceptance after repair (selection-biased, reported as such);
  - all base and all large trajectories.
- **Cost (estimate):** about 8 H200-h, dominated by large.

### Phase 4c result (Oct 12, 2026; jobs 420304 and 420307): **FAIL**, a Goodhart effect

Full tables are in `results/p4c_summary/SUMMARY.md`. The identity check gave a difference of exactly 0 on every shard. The script was submitted twice: each shard of 420307 started after the same shard of 420304 had finished and rebuilt it from scratch (the computation is deterministic). The result has 600 rows and no duplicates. One unit (large, `128221_000__ood_force_x3`) crashed in both runs and is excluded.

**Rejected base trajectories (n = 198), mean error:**

| Method | Error |
|---|---|
| R0 original | 0.267 |
| R1 replace | 0.230 |
| **R4 conditioning** | **0.511** |

- R1 − R4 = −0.280 [−0.321, −0.244]. All base: 0.190 → 0.363. Large: 0.193 → 0.411.
- **Yet the held-out defect fell:** base median 0.131 → 0.091, lower for R4 in 73% of trajectories.

**Diagnosis:**
1. **The selection is noise-dominated.** In 93% of trajectories a candidate scores below the identity's score; in 84% of those, its true error is higher.
2. **The window defect is nearly flat across stiffness candidates** (median 0.053–0.061 over Δlog₁₀E from −1 to +1), so the argmin over 30 candidates is a winner's curse.
3. **The anchor mismatch has the correct minimum** (force scale ×1: 0.0114) but is about 5× smaller than the defect, so it decided only 1% of selections.

**Conclusion for the paper:** the window defect is a valid *between-trajectory* certificate (Phases 2c, 4a), but it must not be used as an *objective* to optimise the generator against. That breaks the exchangeability the certificate relies on, and it rewards locally plausible but globally wrong outputs. Report this as an ablation.

**Next (needs user choice):**
- **Certify-or-simulate:** the proposal's fallback. Rejected trajectories are replaced by one full simulation at the CFL-adaptive coarse setting (about 1/8 of a reference rollout, error about 0.016). This gives a compute-accounted pipeline whose accepted outputs carry the certificate.
- **Anchored repair:** a truth-tied objective, for example a longer anchor simulated from the true initial state.

### Phase 4d: certify-or-simulate (pre-registered Oct 12, 2026, before the fallback rollouts existed; user-approved)

- **Code:** data from `code/p4d_fullsim.py` (`configs/p4d_fullsim.yaml`, `slurm/44_p4d_fullsim_A100.sh`); analysis in `code/p4d_pipeline.py` (`configs/p4d_pipeline.yaml`).
- **Fallback:** one full rollout from the true initial state at `coarse_cfl`, for the Phase 2c queries and the calibration set, timed on A100.
- **Evaluation set:** the 300 Phase 2c queries per model.
- **Costs (A100 seconds):**
  - generation: the Phase 1 best batch (batch 1 reported); ×8 for a seed-spread certificate;
  - probes: the measured 2c scan seconds of the certificate's windows;
  - fallback: measured by the new job;
  - reference: the Phase 1 `physctrl` nosync rollout.
- **Pipelines:** gen; sim_ref; sim_fb; **cert** (accept if score < λ̂, else fallback); random (the same fallback rate); oracle; ss_cert.
- **λ̂:**
  - base defect: 0.02614, the object-level value;
  - base seed spread: 0.01577, from Phase 4a;
  - large seed spread: 0.02979.
- **Tests (primary model base):**
  - **T1:** mean error random − cert, 95% lower bound > 0 (the value of verification).
  - **T2:** risk (fraction with err > 0.10), reported.
  - **T3:** cost against sim_fb and sim_ref, and the break-even simulator cost c* = (c_gen + c_probe) / (1 − fallback rate).
  - **T4:** error and risk against cost as λ sweeps the score quantiles.
- **Tested:** the analysis ran in a scratch sandbox with fake fallback rows (outputs discarded; nothing written to `results/`).

### Phase 4e: anchored conditioning repair (pre-registered Oct 12, 2026, before any result; user-approved)

- **Code:** `code/p4e_repair_anchored.py` (`configs/p4e_repair.yaml`, `slurm/45_p4e_repair_anch_H200.sh`); verdict from `code/p4e_summarize.py`.
- **Candidates:** the same 30 as Phase 4c (same grid, shared noise and seed).
- **Selection:** J_L = mean L2 to a rollout from the **true initial state** over model frames 0..L−1, at `coarse_cfl`, for L ∈ {8 (primary), 12}. The window defect is not used, and ground truth never enters the selection path.
- **Outputs:**
  - R5_L = argmin J_L (identity first);
  - R6_L = R5_L with frames 0..L−1 replaced by the anchor.
- **Primary (falsifier):** rejected base trajectories (the 4b/4c population). PASS if mean error R1 − R6_8 has a 95% lower bound > 0.
- **Secondary:**
  - R0 − R5_L on frames ≥ L: does an exact start fix the generated remainder?
  - L = 12;
  - large;
  - the chosen adjustments.
- **Non-finite repaired trajectories** score +∞ and are counted.

### Phase 4d result (Oct 12, 2026; job 421205): T1 PASS; the cheap fallback is not accurate OOD

Full tables are in `results/p4d_pipeline/SUMMARY.md`. Data: 1,500 fallback rollouts; 2 crashed and 3 unstable (all `ood_force_x3`), handled as reference reruns. The script was fixed to read the cost of crashed rows safely (`.get`); the treatment is unchanged.

**Fallback (`coarse_cfl` full rollout) error vs reference ground truth:**

| Split | Median error | Fraction > 0.10 |
|---|---|---|
| in-distribution | 0.015–0.017 | 0.04–0.05 |
| `ood_E_high` | 0.015 | 0.04 |
| `ood_E_low` | 0.045 | 0.26 |
| `ood_force_x3` | 0.062 | 0.30 |
| `ood_gravity_drop` | **0.103** | **0.52** |

The Phase 1 "matched accuracy" setting was measured in distribution only. It fails in the large-deformation and contact cases where the generator also fails.

**Base defect certificate** (λ̂ = 0.02614, fallback rate 0.70):

| Pipeline | Mean error | Risk | Cost (s/query) |
|---|---|---|---|
| gen | 0.190 | 0.483 | 0.236 |
| random | 0.101 | 0.284 | 0.738 |
| **cert** | **0.069** | **0.207** | **1.135** |
| sim_fb | 0.063 | 0.200 | 0.714 |
| sim_ref | 0 | 0 | 4.666 |

- **T1 PASS:** random − cert = +0.032 [+0.026, +0.038].
- **T3 break-even:** a rollout cost of 2.21 s (generation 0.236 + probes 0.419). Against sim_fb (0.71 s) cert is not cheaper; against sim_ref (4.67 s) it is.

**Other certificates:**
- **Base seed-spread certificate: T1 FAIL** (−0.012 [−0.024, +0.000]). It needs 8 samples, which costs 1.89 s.
- **Large seed-spread certificate: T1 PASS** (+0.024 [+0.018, +0.031]), but generation dominates: 36 s per query for 8 samples.

**Exploratory, not pre-registered:** cert with the **reference** simulator as fallback.
- Base: mean error 0.011, risk 0.013, cost 3.94 s vs 4.67 s (0.84×).
- In distribution: 3.04 s (0.65×); `ood_E_high`: 2.87 s (0.62×).
- Under shift it falls back in 88–100% of queries.
- This should be confirmed on the calibration test half before any claim.

### Phase 4e result (Oct 12, 2026; job 421206): **FAIL**; anchored conditioning does not propagate

Full tables are in `results/p4e_summary/SUMMARY.md`. 600 rows, no crashes. The anchor's own error is 0.035 on rejected base trajectories.

**Primary (L = 8), rejected base trajectories:**
- R6_8 0.2416 vs R1 0.2300: R1 − R6 = −0.0116 [−0.016, −0.007]. **FAIL.**
- At L = 12, R6 = 0.2162 (+0.0138 vs R1), but only because half of the trajectory is replaced by simulated frames.

**Extrapolation (frames ≥ L, R0 − R5):**
- base: −0.0014 [−0.006, +0.002] at L = 8 and +0.0013 at L = 12. There is no improvement from matching an exact start.
- large: −0.020 (worse).

**Conclusion across 4b, 4c and 4e:** PhysCtrl's later frames do not follow from constraints on earlier frames or on its conditioning. Every repair gain came from the simulated frames themselves. Training-free generative repair is not supported. A generator trained to condition on frames, such as a LoRA fine-tune with random frame-conditioning masks (anticipated in the original plan, Phase 4b), is the remaining generative route.

### Phase 4f: confirm certify-or-simulate with a reference fallback (pre-registered Oct 12, 2026, before any analysis of it)

- **Why:** the exploratory Phase 4d number (reference fallback, 0.84× cost) was not pre-registered and was computed on the selection splits.
- **Code:** `code/p4f_confirm_pipeline.py` (`configs/p4f_confirm.yaml`). CPU only.
- **Data:** base calibration pool (id_new_0..11: 2,400 trajectories, 100 held-out objects).
- **Signal:** the frozen Phase 4a signal (`coarse_cfl`, m = 4, `spread_event`).
- **Procedure:** 500 random 50/50 object splits.
  - λ̂ comes from object-level single-draw subsampling on the calibration half, at ε = α = 0.10.
  - The test half is evaluated object-weighted.
- **Pipelines on the test half:**
  - gen;
  - sim_ref (error 0, cost c_ref);
  - **cert_ref** (accept if score < λ̂, else reference rollout);
  - random_ref (the same fallback rate, random).
- **Costs (A100 s/query):**
  - generation: Phase 1 best batch (batch 1 reported);
  - probes: substeps × the A100 per-substep rate, with a per-window overhead fitted by least squares on the Phase 2c `coarse_cfl` m = 4 windows (A100-timed), because the calibration scans ran on H200;
  - fallback: c_ref, the Phase 1 `physctrl` nosync rollout.
- **Tests:**
  - **F1 (risk):** mean over replicates of P(output err > ε) for cert_ref ≤ α + 0.01. This is guaranteed when P(accept | bad) ≤ α.
  - **F2 (compute):** cost(cert_ref) / cost(sim_ref) has a mean < 1 and a 97.5th percentile < 1 over replicates.
  - **F3 (value of verification):** risk(random_ref) − risk(cert_ref) has a 2.5th percentile > 0 over replicates.
- **Reported:** mean error, fallback rate, and the cost ratio with batch-1 generation.
- **Caveat:** in distribution only; the calibration objects are the Phase 2c objects with new physics draws.

**Result (Oct 12, 2026): F1, F2 and F3 all PASS.** Full tables are in `results/p4f_confirm/SUMMARY.md`. Base model, 1,200 trajectories, 100 objects, 500 splits.
- **F1 risk:** 0.023 [0.000, 0.055] ≤ 0.11.
- **F2 cost ratio** vs always-reference: 0.559 [0.352, 0.849]; with batch-1 generation 0.573.
- **F3** risk(random) − risk(cert): +0.053 [+0.029, +0.072].
- **Other numbers:** mean error 0.025 vs 0.071 for generation alone; fallback rate 0.42; cost 2.61 s vs 4.67 s.
- **Probe cost model:** 0.047 s/window + 0.202 ms/substep (18,898 A100 windows).

### Phase 4g: frame-conditioned repair model (pre-registered Oct 12, 2026, before any data or training)

**Hypothesis.** Training-free repair failed (4b/4c/4e) because PhysCtrl was never trained to use clean frames as context: it ignores constraints. Fine-tuned with random frame-conditioning masks, the same DiT should *propagate* verified windows to the frames that were not simulated.

**What would be new:** a generator that conditions on simulator-verified windows, inside a certify → repair → re-certify loop. PhysCtrl, PSIVG and PhysAgent condition on images, forces or a full simulation, never on sparse verified windows chosen by a certificate.

**Model (`code/fc_model.py`).**
- PhysCtrl base, unchanged, plus one learned "known frame" embedding, zero-initialised and added to the `input_encoder` output of masked frames through a forward hook.
- At initialisation it is exactly PhysCtrl base; the job checks this first.
- Inference: known frames are written clean into x_t before every denoising step and into x₀ after it.

**Training data (`configs/p4g_train_build.yaml`).**
- Objects: the 100 objects of PhysCtrl's released `sample/` set (training-distribution objects, no overlap with the 100 evaluation objects).
- Physics: 30 fresh **in-distribution** draws each (the `id_new_*` generator) = 3,000 trajectories, simulated at the reference setting.
- In-distribution physics only, so that OOD gains cannot come from training data.

**Training (`code/p4g_train_fc.py`, frozen config `configs/p4g_train.yaml`).**
- Full fine-tune of base, 20,000 steps, learning rate 5e-5, batch 8, seed 0.
- Loss: PhysCtrl's x₀ MSE plus velocity loss (λ_vel = 1).
- Masks per sample: with p = 0.1 no frames known; otherwise the anchor (frames 0..3) with p = 0.7, plus 0–3 random 4-frame windows.
  - Clarification (Oct 12, before any training): the resulting P(empty mask) is 0.1 + 0.9 × 0.3 × ¼ = 0.1675 (measured 0.1669); the recipe is unchanged.
- Known frames are clean in the noisy input.
- **The final checkpoint is used.** No checkpoint is selected on evaluation data; a training-object validation loss is logged only.

**Evaluation (`code/p4g_repair_fc.py`).**
- Queries: the Phase 2c queries and base generations.
- Constraints: identical to Phase 4b (the anchor plus the certificate's `spread_event` windows, simulated at `ref`).
- Methods, compared with R0 (original) and R1 (replace):
  - **R7**: fine-tuned model, constrained;
  - R7u: fine-tuned model, unconstrained;
  - R8: fine-tuned model, anchor only.
- Seeds: as in Phase 4b.

**Primary (falsifier):** on rejected base trajectories (the 4b population, n = 198), PASS only if **both** hold:
1. mean error on **unconstrained frames**, R1 − R7, has a 95% paired-bootstrap lower bound > 0 (propagation);
2. mean error on all frames, R1 − R7, has a lower bound > 0.

**Secondary:**
- R7u vs R0 (fine-tuning alone);
- R8 on frames ≥ 4 vs R0 (extrapolation from the anchor);
- certificate acceptance of R7 at λ̂ = 0.02614;
- all base trajectories; in-distribution only.

**Cost (estimate):** about 4 A100-h for data, 2–4 H200-h for training, about 1 H200-h for evaluation. Under the 50 GPU-h threshold.

### Phase 4g result (Oct 13, 2026; jobs 421441–421443): **FAIL**; the fine-tune caused forgetting, and conditioning was learned

Full tables are in `results/p4g_summary/SUMMARY.md`.

**Data and training:**
- Training data: 3,000 trajectories, 0 failures.
- Training: identity check at initialisation exact (0); held-out-training-object validation loss 7.0e-4 → 3.1e-4, plateau from about 10k steps; known_emb norm 0.04 → 0.50.
- Evaluation: 300 rows.

**Primary, rejected base (n = 198):**

| Method | All frames | Unconstrained frames |
|---|---|---|
| R0 original | 0.267 | — |
| R1 replace | 0.230 | 0.314 |
| R2 (4b) | 0.228 | — |
| **R7 (fine-tuned, constrained)** | **0.243** | **0.345** |

- R1 − R7: all −0.013 [−0.020, −0.007]; unconstrained −0.031 [−0.046, −0.018]. **FAIL.**

**Diagnosis:**
1. **Forgetting.** The fine-tuned model *without* constraints (R7u) is far worse than PhysCtrl base: 0.313 vs 0.267 (rejected), and 0.103 vs 0.070 on `test_id` (+47%). Twenty thousand steps at lr 5e-5 over 100 objects (about 53 epochs) degraded the prior. A possible extra cause: our training physics draws use a radial proxy for the drag direction (PhysCtrl uses the mesh normal).
2. **Conditioning was learned.** Relative to its own unconstrained output, constraints help: R7u 0.313 → R7 0.243 overall; unconstrained frames 0.373 → 0.345. Propagation exists, but it is swamped by the degraded prior.
3. **Anchor-only extrapolation is worse than base** (R8 0.310 → 0.365 on frames ≥ 4), consistent with (1).

**Next:** the same idea with forgetting prevented (Phase 4g-v2, below).

### Phase 4g-v2: frame-conditioned repair without forgetting (pre-registered Oct 13, 2026, before any v2 code ran on GPU)

**Why:** in v1, conditioning was learned, but the full fine-tune degraded the prior (unconstrained +47% in distribution).

**Changes (principled; the evaluation is unchanged):**
1. **Base frozen.** Only the known-frame embedding and **LoRA adapters on the temporal attention** (`attn_temp`: `to_q`, `to_k`, `to_v`, `to_out.0`) of every block are trained: rank 8, α = 8, B zero-initialised. Constraints can reach other frames only through temporal attention, so that is where the change goes.
2. **Distillation to the frozen base:** loss += λ_d · MSE(student(x_t, no known frames), base(x_t)) on the same noisy batch, λ_d = 1. "Base" is the same network with the adapters switched off.

**Training (frozen `configs/p4g_train_v2.yaml`):**
- Same data as v1. 10,000 steps, learning rate 1e-4, batch 8, seed 0.
- Same masks and task loss (x₀ + velocity). The final checkpoint is used.

**Evaluation:** identical to Phase 4g (`p4g_repair_fc.py`), with run names `p4g_repair_v2` / `p4g_summary_v2` and checkpoint `fc_base_v2`.

**Primary:** the same two-part test as 4g (R1 − R7 > 0 on unconstrained frames AND on all frames, rejected base trajectories).

**New pre-registered secondary, non-degradation:** |mean err R7u − R0| ≤ 0.005 on `test_id`, reported as PASS/FAIL.

**Cost (estimate):** about 3 H200-h for training and 1 H200-h for evaluation.

### Phase 4g-v2 result (Oct 13, 2026; jobs 422192–422193): forgetting fixed, no propagation, **FAIL**

Full tables are in `results/p4g_summary_v2/SUMMARY.md`.

**Training:** identity check exact (0); 131k trainable parameters; validation loss 6.4e-4 → 2.2e-4 (lower than v1's 3.1e-4).

**Non-degradation: PASS**, just inside the limit: R7u − R0 = +0.0046 on `test_id` (v1: +0.033).

**Primary (rejected base): FAIL.**
- R7 0.2303 vs R1 0.2300: R1 − R7 = −0.0002 [−0.0018, +0.0013].
- Unconstrained frames: −0.0004 [−0.0041, +0.0030].
- Anchor-only extrapolation is slightly worse than base (−0.008).

**Diagnosis: the windows carry little corrective information** (from the 4b rows, rejected base, n = 198):

| Verified frames | Generated error | Simulated error | Reduction |
|---|---|---|---|
| Anchor, from the TRUE initial state | 0.052 | 0.003 | 94% |
| Windows restarted from the GENERATED state | 0.299 | 0.238 | 20% |

Windows restarted from the generated state inherit its accumulated drift. They detect inconsistency (the certificate works) but do not supply the truth needed to correct it. That bounds every window-constrained repair (4b, 4c, 4e, 4g, 4g-v2), and more training data would not lift the bound.

**Lesson for the paper:** verification is not correction. Correction needs truth-anchored simulation, so make it *targeted*.

### Phase 4h: certified-prefix repair, re-simulating only the uncertified suffix (pre-registered Oct 13, 2026, before any data)

**Hypothesis.** The certificate's windows can *localise in time* where a trajectory first becomes inconsistent. Keep the certified prefix, and re-simulate from the state at the start of the first failing window to the end. This costs a suffix rollout instead of a full one, while keeping the error near certify-or-simulate's.

**What would be new:** post-hoc, residual-free temporal localisation of the first failure on a one-pass generated trajectory, with a restart from a lifted state.
- ANCHOR corrects online, inside an autoregressive rollout, triggered by a PDE residual.
- Hybrid Neural World Models defer the whole trajectory.

**Procedure** (`code/p4h_suffix_repair.py`, `configs/p4h_suffix.yaml`, `slurm/51_p4h_suffix_A100.sh`):
- Population: the Phase 2c base queries.
- Temporal chain of windows, all at `coarse_cfl`, m = 4, defects read from the Phase 2c exhaustive scan:
  - the anchor (true initial state, model frames 0..3);
  - windows starting at model frames k ∈ {3, 7, 11, 15, 19}.
- A window passes if its defect < τ = 0.02614 (the Phase 4a object-level λ̂). j is the first failing window.
- **Repair:**
  - anchor fails: a full reference rollout;
  - no window fails: accept the generation;
  - otherwise: keep generated frames 0..k_j, lift the state at model frame k_j from the generation (`lift_noF`), and simulate the rest at the reference setting.
- **Suffix lift floor:** the same suffix restart from the ground-truth positions at k_j, which measures the error of restarting with F = I.

**Costs (A100):**
- generation: Phase 1 best batch;
- probes: the scan's A100 seconds of all chain windows evaluated up to and including j;
- suffix: the measured rollout seconds.

**Amendment (Oct 13, before any 4h result):** the A100 queue was about 1.5 days, so the job may run on H200 (`slurm/52_p4h_suffix_H200.sh`). The suffix is then costed as its **share of the measured A100 reference rollout**: c_ref × suffix substeps / 20,016. That is the same scale as a full fallback (c_ref). A linear window fit was rejected because it extrapolated a full rollout to 5.75 s against the measured 4.67 s. Generation, probe and reference costs are unchanged (A100).

**Primary**, against certify-or-simulate with reference fallback on the same queries and λ̂ (the 4d/4f pipeline), PASS if both hold:
1. mean cost per query is lower (paired bootstrap, 95% lower bound of the cost saving > 0);
2. risk P(err > 0.10) ≤ the certify-or-simulate risk + 0.02.

**Secondary:** mean error; the distribution of the first-failure position; the suffix lift floor; per split.

### Phase 4h result (Oct 13, 2026; H200 job 423151): cost PASS, risk FAIL, **FAIL**

Run on H200 under the Oct 13 amendment; 300/300 rows, no crashes, no suffix or floor blow-ups. Full table: `results/p4h_summary/SUMMARY.md`.

| pipeline | mean error | risk P(err > 0.10) | cost / always-reference |
|---|---|---|---|
| certify-or-simulate (reference fallback) | 0.0114 | 0.013 | 0.84 |
| certified-prefix repair | 0.0403 | 0.070 | 0.71 |

- (1) cost saving +0.636 s/query [+0.480, +0.789]: PASS. (2) risk 0.070 vs 0.013 + 0.02: **FAIL**.
- Modes: 85 full (anchor fails; all 50 gravity drops), 80 accepted, 135 suffix. First failure: k = 3: 52, 7: 26, 11: 16, 15: 12, 19: 29.
- On the 135 suffix queries the suffix lowers error versus keeping the generation in 90% of cases (mean 0.110 → 0.066), but the reference fallback gets 0.

**Diagnosis (exploratory, CPU, after the verdict):** two separate error sources, by restart position.

| first failure k_j | n | generation error at k_j | suffix error | suffix floor (true positions, F = I) |
|---|---|---|---|---|
| 3 | 52 | 0.021 | 0.073 | 0.070 |
| 7 | 26 | 0.046 | 0.069 | 0.026 |
| 11 | 16 | 0.066 | 0.074 | 0.008 |
| 15 | 12 | 0.094 | 0.071 | 0.005 |
| 19 | 29 | 0.079 | 0.045 | 0.001 |

1. **Early restarts lose the stress state.** At k = 3 the floor alone (true positions, F = I) is 0.070; 10 of the 52 floor restarts already exceed ε, 9 of them in `ood_E_low` (floor 0.089 there). Positions are not a state for a soft, still-deforming body.
2. **Late restarts start from a state that is already wrong.** The floor is tiny (≤ 0.008), but the generated positions at k_j are 0.07–0.09 off. Passing windows before k_j certify *local physical consistency*, not *accuracy*: a trajectory can drift smoothly away from the truth while every window from its own states stays consistent. The "certified prefix" is consistent, not correct.

Both point the same way: a repair needs a full, accurate state at the restart, and neither the generator (positions only) nor the windows (consistency only) provides one. This is the sixth repair attempt that fails (4b, 4c, 4e, 4g, 4g-v2, 4h).

### Phase 4i: full-state restarts for the suffix repair (pre-registered Oct 13, 2026, before any 4i code ran; user-approved: "do both")

**Why.** Phase 4h failed for two diagnosed reasons: (a) early restarts lose the stress state (F = I), and (b) late restarts begin from generated positions that have drifted. Each variant below targets one cause. Everything else is the 4h protocol, unchanged: same 300 queries, same chain, same τ, same first-failure decision j / k_j, same output (generated frames 0..k_j, simulated frames after k_j), same comparator and same costing.

**Variants** (`code/p4i_state_repair.py`, `configs/p4i_state.yaml`, `slurm/53_p4i_state_H200.sh`), run only on the 4h `suffix` queries; `accept` and `full` queries keep their 4h rows:
- **V1, stable F-lift** (targets cause a): lift (x, v, F, C) from the generated positions at k_j with F estimated by least squares (k = 12) and then **projected**: SVD F = U Σ Vᵀ; if det(U Vᵀ) < 0, negate the last column of U and the smallest singular value (closest proper rotation); clamp every singular value to [s_lo, s_hi].
  - **Bounds rule (fixed now, computed once by `code/p4i_f_bounds.py` on CPU):** s_lo, s_hi = the 0.1% and 99.9% quantiles of the singular values of the stored ground-truth F, over all particles and the odd frames 1..47, of the first file (sorted) of each 4g training object in `data/fc_train/`. These objects are disjoint from every evaluation object. The bounds are written to `results/p4i_f_bounds/bounds.json` and not changed afterwards.
  - V1 floor: the same restart from the ground-truth positions at k_j.
- **V2, coarse-shadow state** (targets causes a and b): simulate from the true initial state at `coarse_cfl` (the Phase 4d setting: dt = min(8e-4, 0.5·dx/c), grid 64) up to sim frame 2k_j + 1, take its full particle state (x, v, F, C), and continue at the reference setting from there to sim frame 47.

**Blow-ups (observable without ground truth, so the pipeline may react):** if the V2 shadow or any suffix rollout blows up, that query falls back to a full reference rollout: error 0, cost = what was already spent + c_ref. Counts are reported.

**Costs (A100 seconds, as in 4h):** generation and probes as 4h; suffix = c_ref × substeps / 20,016 (the 4h amendment); V2 shadow = the query's measured A100 `coarse_cfl` full rollout from Phase 4d × shadow substeps / that rollout's substeps (if the query has no stable 4d row: the split's median A100 seconds per substep). The lift itself (CPU, milliseconds) is not charged, as in 4h.

**Primary, per variant** (same two conditions as 4h, against certify-or-simulate on the same queries), with a Bonferroni correction for two variants: PASS if
1. the cost saving's paired-bootstrap lower bound at α = 0.025 (two-sided 97.5% interval) is > 0, and
2. risk P(err > 0.10) ≤ certify-or-simulate's risk + 0.02.

**Secondary:** each variant vs 4h on the 135 suffix queries (paired mean-error difference with 95% CI); per split and per first-failure position; V1 floor vs 4h floor; the V2 shadow's position error at k_j vs the generated one; the fraction of clamped singular values; blow-up counts.

**Falsifiers.** If both variants fail (2), restarting from a better state is not enough at this τ, and contribution 2's repair is reported as the exact fallback (4f) with 4b–4i as negative results; this needs the user's approval before the paper says so. If V2 passes and V1 fails, the drift (cause b) is the binding problem, and the generator supplies only the certified prefix.

### Phase 4i result (Oct 13, 2026; H200 job 423187): **V2 PASS**, V1 FAIL

135/135 suffix rows, no crashes, no blow-ups (so no fallbacks were used). Full table: `results/p4i_summary/SUMMARY.md`.

| pipeline (300 queries) | mean error | risk P(err > 0.10) | cost / always-reference |
|---|---|---|---|
| certify-or-simulate | 0.0114 | 0.013 | 0.84 |
| 4h suffix (F = I, generated positions) | 0.0403 | 0.070 | 0.71 |
| V1 stable F-lift | 0.0360 | 0.057 | 0.71 |
| **V2 coarse shadow** | **0.0235** | **0.017** | **0.74** |

- **V2: PASS.** Cost saving +0.505 s/query, 97.5% CI [+0.335, +0.661]; risk 0.017 ≤ 0.013 + 0.02. On the suffix queries it lowers error vs 4h by 0.0375 [0.032, 0.044]; risk 0.00 on `test_id`, `ood_E_low` and `ood_force_x3`, 0.05 on `ood_E_high`.
- **V1: FAIL** on risk (0.057). Projected F fixes the restart itself (ground-truth floor 0.033 → 0.007, no blow-ups) but not the generated positions: F estimated from them is noisy (22.5% of singular values clamped), and `test_id` risk rose 0.07 → 0.13.
- So cause (b), the drift of the generated state, is the binding problem, as the falsifier anticipated: the generator supplies the certified prefix and the accept decisions, and the suffix needs a state from simulation.

**Exploratory (after the verdict):**
- V2's 5 failures: 4 are accepted queries (certificate false accepts, the same mechanism as certify-or-simulate's residual risk), 1 is a suffix (`ood_E_high`, k = 11; shadow error at k_j 0.08).
- V2 error splits into the kept generated prefix (mean 0.025) and the simulated suffix (0.032); the shadow's position error at k_j is 0.023 vs 0.050 for the generation.
- **Caveat:** V2 was designed from the 4h diagnosis on these same 300 queries. Its settings were not tuned (the shadow is the Phase 4d setting, the restart point is 4h's, and both variants are reported with Bonferroni), but no held-out confirmation exists yet.

### Phase 4j: held-out confirmation of V2 + ablation of the temporal localisation (pre-registered Oct 13, 2026, before any 4j code ran; user-approved)

**Why.** 4i V2 was designed from the 4h diagnosis on the same 300 queries. This confirms it on the calibration pool with the Phase 4f protocol, and tests whether the certificate's *timing* matters.

**Data:** the base calibration pool (id_new_0..11: 1,200 queries, 100 objects; in distribution only, as in 4f). Generated sample 0, as in the scans and errors.

**GPU job** (`code/p4j_allk.py`, `configs/p4j_confirm.yaml`, `slurm/54_p4j_allk_H200.sh`): for every query, one `coarse_cfl` shadow from the true initial state to sim frame 39, exporting its full state at sim frames 2k+1 for k ∈ {3, 7, 11, 15, 19}; then, for each k, a reference rollout from that state to sim frame 47. Output frames: generated 0..k, simulated after k. Recorded per k: err_t, substeps, blow-up flags, shadow position error.

**Pipelines** (`code/p4j_confirm.py`, CPU), on each of the same 500 object splits as 4f (same seeds; λ̂ from the calibration half exactly as 4f, so cert_ref reproduces 4f):
- **cert_ref:** the 4f pipeline (3 `spread_event` windows; reference fallback).
- **V2:** the 4h chain (anchor + windows at k ∈ {3, 7, 11, 15, 19}) with τ = λ̂: all pass → accept; anchor fails → reference rollout; otherwise V2 at the first failure k_j.
- **shuffled (ablation):** the same as V2, but within each test half the k_j of the suffix queries are randomly permuted, so the restart times have the same distribution but carry no information.
- Secondary only: **earliest** (V2 with k = 3 for every suffix query), **coarse-only** (Phase 4d full `coarse_cfl` rollout), generate-only.

**Costs (A100 s/query):** generation and c_ref as 4f; window probes with the 4f fit (a + b × substeps, from A100-timed 2c windows), for every chain window evaluated up to and including j; suffix = c_ref × substeps / 20,016; shadow = the query's Phase 4d A100 `coarse_cfl` rate × shadow substeps; a blow-up falls back to a reference rollout (error 0, + c_ref); a crashed unit is charged the planned costs + c_ref. All rates are object-weighted, as in 4f.

**Tests** (over the 500 splits):
- **J1 (risk, primary):** mean risk(V2) ≤ α + 0.01 = 0.11, and mean [risk(V2) − risk(cert_ref)] ≤ 0.02.
- **J2 (compute, primary):** cost(V2) / cost(cert_ref): mean < 1 and 97.5th percentile < 1.
- **J3 (localisation, ablation):** mean error(shuffled) − mean error(V2): 2.5th percentile > 0.
- **Confirmed** if J1 and J2 pass. The paper may claim that *the certificate's timing* matters only if J3 also passes.

**Falsifiers.** J1 or J2 fails: V2 does not generalise and 4i is reported as a development-set result only. J3 fails: the gain comes from the coarse shadow, not from temporal localisation, and the paper must say so.

**Expected GPU cost:** ≈ 2.5 reference-rollout equivalents per query, ≈ 7 GPU-hours (4 × H200, about 2 h wall-clock).

### Phase 4j result (Oct 14, 2026; H200 job 423214): **NOT CONFIRMED** (J1 PASS, J2 FAIL at the margin, J3 PASS)

1,200/1,200 queries, 6,000 restarts, no crashes or blow-ups; cert_ref reproduces 4f exactly. Full table: `results/p4j_confirm/SUMMARY.md`. Values: mean [2.5%, 97.5%] over 500 object splits.

| pipeline | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only | 0.071 | 0.131 | 0.05 |
| coarse_cfl only (no generation) | 0.029 | 0.042 | 0.12 |
| certify-or-simulate (4f) | 0.025 | 0.023 | 0.56 |
| **V2** | 0.034 | 0.032 | 0.49 |
| V2, shuffled restarts | 0.037 | 0.041 | 0.49 |
| V2, earliest restart k = 3 | 0.027 | 0.027 | 0.59 |

- **J1 risk: PASS.** V2 0.032 [0.002, 0.068] ≤ 0.11; V2 − cert_ref +0.009 [0.000, 0.020], mean ≤ 0.02.
- **J2 compute: FAIL.** cost(V2)/cost(cert_ref) 0.888 [0.810, **1.034**]: the mean is below 1, the 97.5th percentile is not.
- **J3 localisation: PASS.** error(shuffled) − error(V2) +0.003 [+0.001, +0.006].
- By the pre-registered rule, V2 is **not confirmed**, and the timing claim is **not supported** (it requires J1 and J2).

**Exploratory diagnosis (after the verdict; sandbox outside `results/`):**
- V2 is cheaper than cert_ref in 94.4% of splits. Its cost ratio rises with λ̂ (Spearman 0.96): in the 5.6% of splits where it exceeds 1, λ̂ ≈ 0.046 (vs 0.029 otherwise) and cert_ref accepts most queries cheaply (cost 1.61 s vs V2 1.66 s).
- Structural cause: V2 accepts only if all 6 chain windows pass, so its accept path costs about twice cert_ref's 3 windows. When most queries are accepted, that probe overhead outweighs V2's savings on the suffixes.
- Shuffled restarts also raise the risk: +0.009 [−0.002, +0.022] (secondary, not significant).
- A candidate fix: gate acceptance by cert_ref's own 3 windows and run the chain only on rejected queries, so the accept path costs the same as cert_ref. It was designed after seeing these results, so a clean test needs fresh queries. It is not pre-registered yet.

### Phase 4k: cert-gated V2 on fresh queries (pre-registered Oct 14, 2026, before any 4k code ran or data existed; user-approved)

**Why.** 4j: V2 was cheaper than certify-or-simulate on average but not in every calibration (J2 FAIL), because its accept path needs 6 chain windows against cert_ref's 3. The gated variant below was designed *after* seeing 4j, so it is tested only on fresh queries that no analysis has touched.

**Fresh data** (H200 chain, `slurm/submit_phase4k.sh`): the 100 test objects × 8 new in-distribution physics draws `id_new_12..19` (800 queries), from the same generator of physics as the calibration pool (`p2_build_ood.py`, seeded by qid); base generations (8 samples); true error; the `coarse_cfl`, m = 4 exhaustive window scan; and V2 at every restart point (`p4j_allk.py`), with the coarse shadow continued to sim frame 47 so that coarse-only comes from the same run.

**Threshold, fixed:** τ = 0.02614, the object-level λ̂ already used in 4h/4i (replicate 0 of the calibration pool). No recalibration.

**Pipelines** (`code/p4k_gated.py`, CPU):
- **cert_ref** (4f): accept if the 3 `spread_event` windows pass τ, else a reference rollout.
- **gated V2 (new):** the cert_ref decision first. Accepted → keep the generation (same cost and error as cert_ref). Rejected → evaluate the chain (anchor + windows at k ∈ {3, 7, 11, 15, 19}; a window already run for the certificate is not charged twice): anchor fails or no chain window fails → reference rollout; otherwise V2 at the first failure k_j.
- Secondary: **ungated V2** (4j), **gated V2 with shuffled restarts** (the k_j of the rejected suffix queries permuted), **coarse-only**, **generate-only**.

**Costs (A100 s/query):** generation and c_ref as 4f; windows with the 4f fit; suffix = c_ref × substeps / 20,016; shadow and coarse-only = pooled median A100 `coarse_cfl` seconds per substep over the stable Phase 4d calibration rows × substeps (no fresh query has a 4d row); a blow-up falls back to a reference rollout (+ c_ref, error 0); a crashed unit is charged the planned costs + c_ref.

**Tests** (object-cluster bootstrap over the 100 objects, 2,000 resamples; rates object-weighted):
- **K1 (risk):** risk(gated) ≤ risk(cert_ref) + 0.02 and risk(gated) ≤ α + 0.01 = 0.11 (point estimates).
- **K2 (compute):** the cost saving cert_ref − gated has a 95% lower bound > 0 at τ = 0.02614 **and** at τ = 0.04501 (the 97.5th percentile of λ̂ in 4j, the regime where V2 lost).
- **K3 (timing):** error(shuffled) − error(gated), averaged over 50 permutations per resample, has a 95% lower bound > 0 at τ = 0.02614.
  - *Amendment (Oct 14, before any 4k data existed):* instead of 50 random permutations, use the exact expectation over permutations: each suffix query's shuffled error is the average of its errors at the other restart points, weighted by the empirical distribution of k_j among the suffix queries (recomputed in every resample). Same quantity, no Monte Carlo noise.
- **Confirmed** if K1 and K2 pass; the timing claim needs K3 too.
- Secondary: everything at τ ∈ {0.01625, 0.02614, 0.02992, 0.04501} (the 4j λ̂ 2.5%, replicate-0, mean, 97.5%); and restricted to the 50 replicate-0 test-half objects (unseen geometry relative to the τ calibration).

**Falsifiers.** K1 fails: gating does not keep V2's risk. K2 fails: the suffix savings on rejected queries do not pay for the extra chain windows. Either way the paper reports certify-or-simulate (4f) as the confirmed pipeline, with V2/gated V2 as negative or partial results.

**Expected GPU cost:** build ≈ 1 h, generate ≈ 0.5 h, scan + all-k ≈ 6 h on 4 × H200 in total: about 8–10 GPU-hours.

**Phase 4k code check on development data (Oct 14, after the protocol above was frozen; exploratory, NOT the test):** run in a sandbox outside `results/`, with the 4k inputs pointed at the already-analysed calibration draws id_new_4..11 (800 queries). At τ = 0.02614: gated V2 risk 0.026 vs cert_ref 0.014; cost 0.50× vs 0.60× always-reference; saving +0.50 s/query at τ and +0.21 at τ_hi; shuffled restarts +0.002 error. These queries informed the design, so only the fresh run counts.

### Phase 4k result (Oct 14, 2026; H200 jobs 424063–424065): **CONFIRMED** (K1, K2, K3 all PASS)

800/800 fresh queries (id_new_12..19), no build failures, crashes or blown restarts. Full tables: `results/p4k_gated/SUMMARY.md`. At the fixed τ = 0.02614, object-weighted:

| pipeline | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only | 0.069 | 0.139 | 0.05 |
| coarse_cfl only | 0.030 | 0.046 | 0.12 |
| certify-or-simulate | 0.018 | 0.011 | 0.65 |
| **gated V2** | 0.031 | 0.025 | **0.53** |
| gated V2, shuffled restarts | 0.033 | 0.028 | 0.53 |

- **K1 risk: PASS.** 0.025 ≤ 0.011 + 0.02 and ≤ 0.11.
- **K2 compute: PASS.** Saving +0.535 s/query [+0.479, +0.592] at τ, and +0.256 [+0.208, +0.305] at τ_hi = 0.04501, the regime where ungated V2 lost.
- **K3 timing: PASS.** Shuffled restarts +0.002 error [+0.001, +0.003].
- So the claim holds on fresh queries: **certificate-gated suffix re-simulation is cheaper than certify-or-simulate (0.53× vs 0.65× always-reference) at a risk within +0.02, and the certificate's timing carries information.**
- The secondary results agree: gated V2 is cheaper than cert_ref at all four τ, and on the 50 replicate-0 test-half objects (cost 2.55 vs 3.07 s, risk 0.028 vs 0.015).
- **Limits:** in distribution only. The timing effect is small (+0.002 error). Coarse-only is cheaper still (0.12×) at risk 0.046 here.

**Exploratory, after the verdict: gated V2 on the Phase 2c queries, including OOD** (`code/explore_gated_2c.py`, from the 4h/4i rows; these are development queries for V2, so not a test). Error / risk / cost vs always-reference:

| split | certify-or-simulate | gated V2 | coarse-only | generate-only risk |
|---|---|---|---|---|
| test_id | 0.019 / 0.03 / 0.65 | 0.034 / 0.03 / 0.51 | 0.030 / 0.05 / 0.12 | 0.22 |
| ood_E_high | 0.022 / 0.02 / 0.62 | 0.035 / 0.04 / 0.54 | 0.023 / 0.04 / 0.26 | 0.10 |
| ood_E_low | 0.004 / 0.00 / 1.01 | 0.020 / 0.00 / 0.94 | 0.075 / 0.26 / 0.12 | 0.64 |
| ood_force_x3 | 0.005 / 0.00 / 1.01 | 0.017 / 0.00 / 0.93 | 0.683 / 0.34 / 0.12 | 0.72 |
| ood_gravity_drop | 0.000 / 0.00 / 1.12 | 0.000 / 0.00 / 1.14 | 0.139 / 0.52 / 0.11 | 1.00 |
| all 300 | 0.011 / 0.013 / 0.84 | 0.023 / 0.017 / 0.76 | 0.163 / 0.210 / 0.14 | 0.48 |

- This is the paper's central contrast: the cheap baselines (generation alone, coarse simulation alone) are fine in distribution but fail under shift (coarse-only risk 0.26–0.52 on E_low, force ×3 and gravity drop). The certified pipelines hold the risk at ≤ 0.04 in every split.
- **The honest cost of safety:** under a hard shift, almost everything is rejected, so cost approaches or exceeds always-reference (gravity drop 1.12–1.14× because of probe overhead).

### Phase 4m: fresh shifted queries, the strongest baselines, and placement at matched budget (pre-registered Oct 14, 2026, before any 4m code ran or data existed)

**Why** (`notes/audit_2026-10-14.md`, objections 1–3):
- The shift results are exploratory; they used the Phase 2c splits, which also served signal selection.
- Reviewers will ask why we do not just run a coarse simulator, and whether adaptive windows beat uniform ones at a matched budget.
- This phase settles all three on queries no analysis has touched.

**Fresh data** (`slurm/submit_phase4m.sh`, H200), from the 100 test objects. Each draw is seeded by qid (which contains the split name), so all draws are new:
- `ood2_E_low`: log10 E ~ U(3.0, 3.9) (training range 4–7); ν ~ U(0.05, 0.45); in-distribution drag (as `id_new_`).
- `ood2_E_high`: log10 E ~ U(7.1, 7.8); otherwise as above.
- `ood2_force`: in-distribution E and ν; drag coefficient ~ U(0.3, 0.8) × PhysCtrl's base coefficient (training range 0.02–0.2, so 1.5–4× the training maximum).
- `ood2_drop`: in-distribution E and ν; no drag; gravity on; floor U(0.1, 0.5) below the object's lowest point (contact).
- `id_new_20`, `id_new_21`: in distribution (200 queries), for the in-distribution side of the agreement baseline.
- Then: base generations (8 samples), true error and seed spread; the `coarse_cfl` m = 4 exhaustive scan; V2 at every restart point with the coarse shadow continued to the end; **coarse model-frame positions saved** (also for the 1,200 calibration queries, for calibration only).
- These shift families differ from Phase 2c's fixed transformations: they test the shift claims on a broader, freshly drawn family.

**Fixed thresholds, all calibrated before any fresh data exists,** by the 4a object-level procedure: replicate-0 object split of the calibration pool, single-draw subsampling, ε = α = 0.10.
- Window defect (`spread_event`): τ = 0.02614. The summarizer recomputes it as a reproduction check.
- Base seed spread: its replicate-0 λ.
- Uniform placement (3 evenly spaced m = 4 `coarse_cfl` windows from the exhaustive scan): its own replicate-0 λ.
- Cross-fidelity agreement: its own replicate-0 λ, calibrated on the error of the coarse trajectory, because that is what it outputs.

**Pipelines** (costs as in 4k):
- **gen:** generation only.
- **coarse:** the full `coarse_cfl` rollout.
- **cert_ref:** 4f.
- **gated V2:** 4k.
- **XF (cross-fidelity agreement cascade, the strongest simple baseline):** run the generation and the coarse rollout. a = max over model frames of the mean point distance between generated sample 0 and the coarse rollout. If a < λ_a, output the coarse trajectory; else run a reference rollout. Cost = generation + coarse + rejection × c_ref.
- **cert_ref_uniform:** cert_ref with uniform windows and their own λ.

**Tests.** The unit of resampling is the object (cluster bootstrap, 2,000 resamples). Shift tests pool the 400 `ood2_*` queries; the per-split values are reported.
- **M1, certificate under shift (confirmatory):**
  - false-accept rate among failing queries (generation error > ε) ≤ 0.10 for the window certificate, and
  - FAR(seed spread) − FAR(window) has a 95% lower bound > 0.
- **M2, pipelines under shift (confirmatory):**
  - risk(gated V2) ≤ 0.10, and
  - risk(coarse) − risk(gated V2) has a 95% lower bound > 0.
- **M3, the agreement baseline (decision rule, reported either way):** XF *dominates* gated V2 if, on BOTH the in-distribution and the shift sets, risk(XF) ≤ risk(gated V2) + 0.02 AND the cost saving gated V2 − XF has a 95% lower bound > 0.
  - If XF dominates, the paper must present XF as a baseline that wins on cost, and repositioning the contributions goes to the user.
  - Otherwise XF is reported as a baseline.
- **M4, placement at matched budget (the compute-value audit bar):** cost(cert_ref_uniform) − cost(cert_ref) and risk(cert_ref_uniform) − risk(cert_ref), with 95% CIs, on the shift set and on the in-distribution set. **Descriptive; no pass/fail.** If the intervals cover 0, the paper says placement does not matter for these splits.
- **Secondary:** per-split tables; gated V2 vs cert_ref cost under shift; blow-ups and crashes.

**Amendment (Oct 14, before any 4m data existed; found by the sandbox test on stand-in data):**
1. **XF calibration.** In distribution the coarse rollout rarely fails, so the replicate-0 subsample rule can have fewer than the 9 failing calibration objects it needs at α = 0.10. It would then return −∞, and XF would reject everything: a strawman. XF's λ_a therefore uses the pooled object-weighted rule (`pooled_threshold`, the Phase 4a-extension secondary method) on the **whole** calibration pool, which gives the baseline its best chance. The fresh queries are disjoint, so using both halves is valid.
2. **One more secondary pipeline** (from the literature check: conformalised model-uncertainty deferral, as in Hybrid Neural World Models): **cert_ss**, which accepts the generation if the base seed spread is below its replicate-0 λ and otherwise runs a reference rollout. It is charged for the 8 generations that seed spread needs.

**Falsifiers.**
- M1 fails: the certificate's shift robustness does not generalise beyond the Phase 2c splits; the paper drops the claim.
- M2 fails: the "cheap shortcuts break, certified pipelines hold" contrast is not supported.
- M3 dominance: the main pipeline is not the best cheap option.

**Expected GPU cost:** build ≈ 1.5 h, generate + errors ≈ 0.5 h, scan + all-k ≈ 5 h, calibration coarse positions ≈ 1.5 h: **≈ 9–11 GPU-hours on H200.**

### Phase 4m result (Oct 15, 2026; H200 jobs 424571–424574): **M1 PASS, M2 PASS, XF does not dominate (M3)**

598 of 600 fresh queries. Two ground-truth builds were unstable at the reference setting (`153956_000__ood2_E_low`, `28968_000__ood2_force`), so those queries do not exist. No crashes; 7 blown restarts fell back to a reference rollout. All thresholds came from the calibration pool, and the window τ reproduced 0.02614. Every job logged that `$BIG_ROOT` is mounted on its GPU node. Full tables: `results/p4m_confirm/SUMMARY.md`.

| pipeline | shift (398): error / risk / cost | in distribution (200): error / risk / cost |
|---|---|---|
| generate only | 0.347 / 0.703 / 0.05 | 0.073 / 0.115 / 0.05 |
| coarse_cfl only | 0.192 / **0.374** / 0.15 | 0.031 / 0.055 / 0.12 |
| certify or simulate | 0.007 / 0.010 / 0.97 | 0.021 / 0.005 / 0.56 |
| certify or simulate, uniform windows | 0.003 / 0.005 / 1.08 | 0.009 / 0.000 / 0.82 |
| **gated V2** | 0.016 / **0.015** / **0.94** | 0.036 / 0.025 / **0.44** |
| XF (accept coarse if it agrees with the generation) | 0.004 / 0.007 / 1.04 | 0.007 / 0.005 / 0.66 |
| seed-spread deferral (8 samples) | 0.131 / 0.242 / 1.06 | 0.009 / 0.000 / 1.08 |

- **M1 certificate under shift: PASS.** FAR(window) 0.012 [0.003, 0.024]. Seed spread's FAR is 0.341 higher [0.315, 0.363]; on `ood2_drop` it accepts 99% of the failures, the window certificate 2%.
- **M2 pipelines under shift: PASS.** risk(gated V2) 0.015 [0.005, 0.028]; coarse-only is 0.359 riskier [0.310, 0.405] (0.31–0.66 on the hard shifts).
- **M3: XF does not dominate gated V2.** XF has lower risk (shift 0.007 vs 0.015; in distribution 0.005 vs 0.025) but costs more: gated V2 is 0.475 s/query cheaper under shift [0.337, 0.607] and 1.009 s cheaper in distribution [0.716, 1.312].
  - Certify-or-simulate also beats XF in distribution (0.56× vs 0.66× at the same risk).
  - So XF is a strong, low-risk baseline at a higher cost, not a dominating one.
- **M4 placement at matched budget (descriptive).** Uniform windows reject more: cost +0.51 s under shift [0.39, 0.63] and +1.24 s in distribution [0.96, 1.54], at risk −0.005 [−0.013 to −0.015, 0.000].
  - `spread_event` sits at a cheaper operating point with risk not significantly different, which answers the compute-value-audit objection for these splits.
  - It is a different operating point, not dominance at equal risk; say so in the paper.
- **Per split:**
  - `ood2_E_high` is the weak spot: FAR(window) 0.33, but only about 6 failures in the split (generation risk 0.06).
  - Under hard shift (E_low, force, drop) all certified pipelines cost 1.0–1.14× always-simulate: the expected price of safety.

**Consequence for the paper:** objections 1–3 of the audit are answered on fresh pre-registered data:
- the shift contrast is now confirmatory;
- the coarse simulator alone fails under shift (risk 0.37);
- the agreement cascade is a costlier operating point;
- placement gives a cheaper operating point.

### Phase 4l: multi-object contact benchmark (item 3; pre-registered Oct 15, 2026, before any 4l code ran or data existed)

**Why.** On drag-only scenes the error is global, so window placement and repair timing cannot matter much (2c, 4k K3: +0.002). With two objects that collide, the error of a per-object generator is local in time (the collision it ignores). This is where event-aware probing (contribution 2) should show, if anywhere. It also answers "single object". Design decisions are in "Ranked plan" below; the design memo is `notes/item3_contact_benchmark_design_2026-10-14.md`. Change from the memo: **no separate pilot**; validity rules are fixed here and H0 uses the main run.

**Construction** (all physics in-distribution, elastic, no gravity, floor 0.2; same Warp MPM at PhysCtrl's settings):
- **Pairing.** For draw j ∈ {0..7}: permute the 100 test objects with `default_rng(crc32("mo_" + j))` and pair consecutive ones: pairs 0–39 are *contact* scenes, 40–49 *controls*.
  - Each scene draws one shared (E, ν) (log10 E ~ U(4, 7), ν ~ U(0.05, 0.45)) from `crc32(scene id)`.
  - Each object draws its own in-distribution drag (as `id_new_`) from `crc32(component qid)`.
  - Scene qid: `<objA>__<objB>__mo_<j>`; the first object is the bootstrap cluster.
- **Components** (`mo_comp_<j>`): each object simulated alone with its scene's (E, ν) and its own drag (800 single-object ground truths). Base generations (8 samples) per component.
- **Placement** (CPU, inside the scene-build job, from component ground truth only, never from generations):
  - A stays in place; B is translated by t_B, a multiple of dx = 0.08 in each axis, so its isolated dynamics are unchanged.
  - Distances use 256 seeded points per object.
  - *Contact:* draw a target frame k* ~ U{5..18} and a random unit direction u; base offset b = c_A(k*) − c_B(k*). By bisection on s ∈ [0, 6], find the smallest s with no approach closer than d_c = 0.16 (2 dx) before k*. Set t_B = round((b + s·u)/dx)·dx.
  - Valid if: the first approach frame k_c (the first frame the superposed isolated motions come within d_c) lies in [3, 21]; initial distance ≥ d_c; every point of both objects in all 49 frames within [0.5, 9.5]. Otherwise redraw u (≤ 20 tries), then reject the pair.
  - *Control:* s grows in steps of 0.2 from 0 until the minimum distance over all frames is ≥ 0.5 and bounds hold (s ≤ 8, else rejected).
  - Rejections are counted and reported.
- **Ground truth:** one joint rollout per scene: 4,096 particles in one grid, both objects' stored volumes, two drag impulses with per-object masks. Unstable builds are recorded and dropped.
  - Sanity rule (fixed now): a scene is **invalid** if, before frame k_c − 2, the joint ground truth deviates from the superposed isolated ground truths by more than 0.02 (mean point distance). That would indicate a setup error, not contact. Invalid scenes are counted and excluded. *(Clarification, Oct 15, before any data: for controls, which never approach, the rule applies to all frames.)*
  - Single-grid MPM gives sticky, no-slip contact starting about 1–1.5 dx early; the paper says so.
- **Two "generated" sources:**
  - **base:** sample s of A plus sample s of B translated by t_B, concatenated (neither generation knows about the other object; 8 samples).
  - **gt_super:** the superposed isolated ground truths (a perfect single-object generator; its only error is the ignored interaction).
- **Verifier changes:** neighbour lists are computed per object (never across the gap). New placements:
  - `contact`: a window starting at k_g − 1, where k_g is the first model frame at which the two objects in generated sample 0 come within d_c; this is available at test time.
  - `contact_spread`: that window plus two windows at the 1/3 and 2/3 points of the remaining candidates. Falls back to `spread_event` if the generated objects never come within d_c.
- **Runs:**
  - exhaustive `coarse_cfl` m = 4 scan for sources base, gt_super and gt (the lift floor);
  - true error and seed spread;
  - V2 restarts at the chain points k ∈ {3, 7, 11, 15, 19} **and** at the event point k_e = max(k_g − 1, 1) (no extra point when k_g is undefined or ≤ 1, in which case the event pipeline falls back to a reference rollout).

**Tests** (contact scenes unless stated; cluster bootstrap over the first object, 2,000 resamples; ε = 0.10; fixed τ = 0.02614):
- **H0, gate:** the median lift floor (gt source) over contact scenes at the `contact` window ≤ 0.25 × the median base defect at the same window.
  - If H0 fails: H1–H3 are still reported but flagged, and the pre-registered follow-up is the same scan at the reference setting (≈ 9 GPU-h, a separate job).
- **H1, placement (primary for contribution 2):** capture = max defect over the chosen 3 windows ÷ max over all windows (exhaustive). Both differences must have a 95% lower bound > 0, on the base source:
  - capture(`contact_spread`) − capture(`uniform`), and
  - capture(`contact_spread`) − capture(`spread_event`).
  - gt_super is reported as secondary.
- **H2, certificate:** among bad contact scenes (base error > ε), FAR(`contact_spread`) ≤ 0.15, and FAR(`uniform`) − FAR(`contact_spread`) has a 95% lower bound > 0. Controls report false rejects.
- **H3, event-timed repair:** the event-gated pipeline accepts if the `contact_spread` windows pass τ; otherwise it restarts V2 at k_e (from the coarse shadow).
  - (a) Against cert_ref (same certificate, reference fallback): risk ≤ cert_ref + 0.02 and cost saving with 95% lower bound > 0.
  - (b) Timing: error(chain-timed, as 4k: the first failing chain window among the rejected) − error(event-timed) has a 95% lower bound > 0, and so does error(shuffled event timing) − error(event-timed).
- **Costs:** as 4k; generation is charged twice (two objects). The reference and window costs use the single-object A100 measurements, because simulation cost tracks substeps, not particles (Phase 1: multiobj2 ≈ physctrl). Stated as an assumption.

**Amendment (Oct 15, before any 4l data existed; found while writing the summarizer):**
- H3(b)'s shuffled event timing needs each scene's error at other scenes' event points, which the planned restarts (chain points + own k_e) do not provide.
- V2 restarts are therefore run at **every model frame k = 1..22** (`chain_k` = 1..22). This contains the chain points and every k_e. Added cost ≈ 8 GPU-h (total ≈ 20, still under the 50 GPU-h limit).
- The tests are unchanged: chain-timed decisions still use only the chain windows at k ∈ {3, 7, 11, 15, 19}.
- The full restart curve (error vs restart frame relative to k_g) is reported as an **exploratory** figure.

**Falsifiers.**
- H1 fails: event-aware placement stays limited to the gravity-drop split; contribution 2 rests on the certificate and repair.
- H3(b) fails: the repair-timing claim stays at 4k's small effect.

**Storage:** components, scenes and composed generations go under `$BIG_ROOT` (≈ 25 GB).

**Expected GPU cost:** component builds ≈ 2.5, generation ≈ 0.5, scene builds ≈ 1.5, scans (3 sources) ≈ 3, errors + all-k ≈ 12 (after the amendment): **≈ 20 GPU-hours (H200)**.

### Phase 6a: qualitative videos (fixed Oct 15, 2026, before any render; qualitative, no hypothesis test)

**Goal.** Videos for the paper and supplement: PhysCtrl's own image-to-video pipeline, with our certificate and repair inserted between its trajectory and its renderer.

**Cases** (all are rendered; none is dropped after seeing it): the 5 elastic PhysCtrl examples (`blanket`, `chair`, `earphone`, `giraffeX`, `penguin`), each **as given** and with **3× the drag force**, the same transform as `ood_force_x3`. That makes 10 cases. `blanket` as given has E = 10³ Pa, already below the training range.

**Per case** (`code/p6_video.py`, `slurm/66_p6_video_H200.sh`):
1. PhysCtrl preprocessing, unchanged: SAM → SV3D → LGM (the point cloud).
2. A PhysCtrl-format query from the reconstruction: 2,048 farthest-point samples (seeded), normalised as in PhysCtrl, the drag box and force exactly as `inference.py` builds them (MPM force = direction × coeff × base_drag_coeff, the data generator's rule).
3. Base generation: 8 samples, seed 0, through our `Generator` on that query.
4. Ground truth: one reference MPM rollout.
5. The 4k gated V2 at τ = 0.02614: the `spread_event` certificate; on reject, the chain and a restart from the coarse shadow.
6. Three trajectories: generation (sample 0), certified/repaired, ground truth. Each is passed through PhysCtrl's tracking step and DAS renderer with the example's prompt and seed.

**Reported for every case:** generation error, repaired error, the certificate decision and restart frame, cost.

**Selection for the main-paper figure (rule fixed now):** among the cases the certificate rejects, the one with the median generation error. All 10 go in the supplement.

**GPU:** ≈ 20 min per case, ≈ 3.5 GPU-h on H200 (DAS is GPU-heavy). Outputs under `$BIG_ROOT/renders/`.

### Phase 6a result (Oct 15, 2026; H200 job 425599): 10/10 cases, 30 renders (qualitative)

| case | generation error | after certify + repair | certificate |
|---|---|---|---|
| blanket ×1 (E = 10³, below training) | 0.038 | 0.038 | accept (score 0.017) |
| chair ×1 | 0.018 | 0.018 | accept (0.013) |
| earphone ×1 | 0.092 | 0.092 | accept (0.024, close to τ) |
| giraffeX ×1 | 0.043 | 0.043 | accept (0.013) |
| penguin ×1 | 0.027 | 0.027 | accept (0.015) |
| blanket ×3 | 0.089 | 0.029 | reject → repair from k = 7 |
| chair ×3 | 0.061 | 0.038 | reject → repair from k = 15 |
| earphone ×3 | 0.222 | 0.010 | reject → repair from k = 3 |
| giraffeX ×3 | 0.086 | 0.027 | reject → repair from k = 7 |
| penguin ×3 | 0.073 | 0.017 | reject → repair from k = 3 |

- All as-given cases are accepted; all 3×-force cases are rejected and repaired (error ↓ 1.6–22×). This is qualitative: 10 cases, no test.
- The pre-stated figure rule (median generation error among rejected cases) selects **giraffeX ×3**. Outputs:
  - side-by-side video: `$BIG_ROOT/renders/_figures/giraffeX__force_x3_side_by_side.mp4`;
  - contact sheet: `results/fig_qualitative/video_side_by_side.png`;
  - selection record: `results/fig_qualitative/video_selection.json`.
  - Under 3× force the object leaves the camera view in the last third of the clip, so figure frames should come from the visible span.
- GPU usage: 44–55% mean, 33–36 GB peak.

**Phase 4l scene build (Oct 15, job 425531; verdict pending):**

| | designed | no valid placement | failed the sanity rule | valid |
|---|---|---|---|---|
| contact | 320 | 18 | 47 | **255** |
| control | 80 | 1 | 0 | **79** |

- **Controls:** match the superposed isolated truths to ≈ 3e-5 over all frames, so the construction (translation, drags, volumes) is exact.
- **Valid contact scenes:** median pre-contact deviation 4e-5; median contact effect by the last frame 0.53.
- **The 47 failures** started interacting before k_c − 2 (median deviation 0.06; k_c mostly 6–12). This is consistent with the 256-point design underestimating the first approach and with MPM's ~3-cell grid coupling.
- Excluded by the pre-registered rule; report that the exclusion removes the earliest contacts.

### Phase 4l result (Oct 15, 2026; H200 jobs 425529–425532): **H0–H3 FAIL** (event-aware placement and timing not supported); certificate catches every contact failure

334 valid scenes (255 contact, 79 control), no crashes; GPU usage 60–72% in the last step. Full tables: `results/p4l_summary/SUMMARY.md`.

**Verdicts:**
- **H0 lift floor: FAIL.** The median floor at the contact window is 0.047, against 0.25 × the median base defect = 0.026. The check is noisy exactly at contact. By the pre-registered rule, H1–H3 are flagged, and the follow-up is the reference-setting scan.
- **H1 placement: FAIL.** Capture of `contact_spread` is −0.079 [−0.118, −0.042] vs uniform and −0.097 [−0.126, −0.069] vs `spread_event` (gt_super: −0.064). Contact-placed windows capture *less*.
- **H2 certificate: FAIL as stated, but for a strong reason.** FAR(`contact_spread`) = 0.000 over 188 bad contact scenes, and uniform is also 0, so the difference cannot be > 0. **Every placement rejects every contact failure.** Controls falsely rejected: 0.49 (the in-distribution base rate, as in 4k).
- **H3a: FAIL.** Event-gated V2 risk is 0.071 vs cert_ref 0.000 (limit +0.02), although it saves 1.39 s/query [1.26, 1.52].
- **H3b: FAIL.** Chain timing beats event timing (chain − event = −0.016 [−0.021, −0.011]). Event timing beats shuffled timing (+0.006 [+0.003, +0.010]).

| pipeline (contact scenes) | error | risk | cost / always-reference |
|---|---|---|---|
| generate only (two objects) | 0.202 | 0.731 | 0.10 |
| certify or simulate | 0.000 | 0.000 | 1.19 |
| event-gated V2 | 0.041 | 0.071 | 0.89 |
| chain-gated V2 (4k timing) | 0.025 | 0.020 | 1.01 |

**Exploratory diagnosis (after the verdict):**
- **The generator's failure is exactly the ignored contact.** Generation error is 0.022 before the approach and 0.305 after it.
- **Defects peak after contact, not at it.** The median base defect rises from 0.02 (8 frames before k_g) to 0.11 at k_g − 1 and 0.15 at k_g + 3…+6. The largest-defect window starts 2–10 frames after k_g in most scenes, so `contact_spread` (one window at k_g − 1) captures less than placements that sample the post-contact tail. Capture penalises it although the accept decision is the same: all placements reject.
- **The check and the shadow are coarse-grid (dx = 0.156; grid coupling ≈ 3 cells ≈ 0.47) while contact happens at 0.16.**
  - The coarse simulator couples the objects far earlier than the reference does, so the coarse lift floor rises at contact (0.011 → 0.05).
  - The coarse-shadow state just before contact is already wrong. The restart curve rises monotonically: 0.015 at k_g − 10, 0.052 at k_g − 1, 0.094 at k_g + 6. Restarting just before contact is therefore worse than restarting earlier, which is why chain timing (earlier) beats event timing.
- **Implication:** in contact scenes the binding problem is the *fidelity* of the cheap simulator near the event, not *where* the windows go. Candidate next step: **event-aware fidelity** (coarse away from contact, reference resolution around it, for both the check and the shadow). It is not pre-registered yet: it needs the pre-registered H0 follow-up (the reference-setting scan) and fresh scenes for any confirmatory claim.

### Phase 4l-ref: the pre-registered H0 follow-up (specified Oct 15, 2026, before any reference-setting contact data)

**User decision (Oct 15):** contribution 2 may be reframed from event-aware *placement* to **event-aware fidelity** (cheap simulation away from predicted events, reference resolution around them) if the evidence supports it. First run the follow-up that 4l pre-registered for an H0 failure.

**Run:** the same exhaustive m = 4 scan as 4l, at the **reference setting** (grid 125, dt 1e-4), for sources base, gt_super and gt, on the same 334 valid scenes (`configs/p4l_scan_ref.yaml`, `slurm/67_p4l_scan_ref_H200.sh`). Two shards share each GPU to raise utilisation.

**Analysis** (`code/p4l_summarize.py` with `configs/p4l_summary_ref.yaml`, unchanged code):
- H0, H1 and H2 are re-evaluated with reference windows.
- The certificate threshold for reference windows, **τ_ref = 0.02894**, was computed before this run by the same replicate-0 procedure on the calibration pool. The same procedure reproduces 0.02614 for `coarse_cfl`.
- H3 is reported but unchanged in meaning (its restarts still use the coarse shadow).

**Reading:**
- If H0 passes at the reference setting, the coarse grid was the cause; this supports event-aware fidelity, which is then pre-registered on fresh scenes.
- If H0 still fails, the check itself is unreliable at contact (lifting with F = I discards contact stress), and event-aware fidelity will not fix it; contribution 2 then stays at the confirmed 4k repair and the reframing is dropped.

**Expected GPU cost:** ≈ 9 GPU-h (21k reference windows on H200), ≈ 2–3 h wall-clock.

### Phase 4l-ref result (Oct 16, 2026; H200 job 425725): **H0 PASS** at the reference setting; H1–H3 still FAIL

1,002/1,002 rows (334 scenes × 3 sources), no crashes, no blown windows of 21,042. Two shards per GPU raised GPU utilisation to **95–97%** (1.4 GB). Tables: `results/p4l_summary_ref/SUMMARY.md`.

- **H0 lift floor: PASS.** The floor at the contact window drops from 0.047 (coarse) to **0.010**, against an allowance of 0.25 × 0.065 = 0.016. The coarse grid was what made the check unreliable at contact. Per the pre-registered reading, this supports event-aware *fidelity*.
- **H1: FAIL.** `contact_spread` − uniform = −0.111 [−0.153, −0.070]; vs `spread_event` −0.117. Defects peak after contact, so contact placement captures less even when the check is reliable.
- **H2: FAIL.** FAR(`contact_spread`) 0.024 at τ_ref; uniform is slightly lower (difference −0.021 [−0.050, −0.003]).
- **H3: FAIL**, unchanged in substance: the restarts still use the coarse shadow, so the restart curve is the same.
- **Conclusion:** event-aware *placement* is not supported, for either check fidelity. The case for event-aware *fidelity* rests on H0 and on the repair diagnosis below.

**Exploratory, Oct 16 (4l scenes are seen data; this only decides whether a fresh test is worth running): event-aware fidelity repair (EAF).**
- Certify with the coarse `spread_event` windows (τ = 0.02614).
- On reject, restart at the reference setting from the coarse shadow at k_s − 1, where k_s is the first frame the *generated* objects come within d_s = 3 coarse cells = 0.469: the support of MPM's quadratic kernel on the 64-grid, i.e. where the coarse simulator starts coupling the objects.
- This restart falls a median of 4 frames before the generated approach (IQR −6…−3).

| pipeline (255 contact scenes) | error | risk | cost / always-reference |
|---|---|---|---|
| certify or simulate | 0.000 | 0.000 | 1.19 |
| **EAF repair** | 0.011 | 0.012 | 1.04 |
| chain-gated V2 (4k timing) | 0.025 | 0.024 | 0.94 |

- EAF meets the 4l risk rule against cert_ref (+0.012 ≤ +0.02) and is cheaper (1.04× vs 1.19×). It halves chain timing's error at slightly higher cost.
- **Honest limit:** on contact scenes, almost every query is rejected (generation risk 0.73), so every certified pipeline costs about as much as simulating everything (≥ 0.94×). The saving is relative to certify-or-simulate, not to always-simulate.
- **Next:** pre-register EAF on fresh scenes (Phase 4n).

### Phase 4n: event-aware fidelity (EAF) repair on fresh contact scenes (pre-registered Oct 16, 2026, before any 4n code ran or data existed; contribution 2 reframing approved by the user Oct 15)

**Claim under test.** In scenes with contact, repair should switch from the cheap simulator to the reference simulator *before the cheap simulator's resolution fails*, i.e. before its grid couples the objects, rather than at the first failing window (4k) or at the contact itself (4l). Motivated by the exploratory 4l analysis (seen data), so it is tested only on fresh scenes.

**Fresh data:**
- The same construction as 4l (`mo_scenes.py`, `p4l_mo_build.py`, unchanged) with **new draws j = 8..15**: new pairings, new shared materials and new drags. That gives 320 contact + 80 control scenes before validity rules.
- The 4l validity rules apply unchanged: placement, bounds, and the 0.02 pre-contact sanity rule.
- Components and base generations (8 samples) as in 4l.

**Runs (GPU):**
- The coarse `coarse_cfl` m = 4 exhaustive scan, base source only.
- V2 restarts from the coarse shadow at the chain points {3, 7, 11, 15, 19}, at the event point k_e = max(k_g − 1, 1), and at the **fidelity point** k_s = max(k_f − 1, 1). Here k_f is the first model frame at which the two objects in generated sample 0 come within **d_s = 3 × (10/64) = 0.46875**: the support of the quadratic MPM kernel on the 64-grid, fixed from the grid alone.
- No restart is run when k_f is undefined or ≤ 1; such scenes fall back to a reference rollout.

**Pipelines** (contact scenes; costs as 4l: generation ×2, coarse probe fit, suffix = c_ref share, shadow = 4d rate; a blow-up falls back to a reference rollout):
- **cert_ref:** coarse `spread_event` certificate at τ = 0.02614; reject → reference rollout.
- **EAF:** the same certificate; reject → V2 restart at k_s.
- **chain** (4k timing): the same certificate; reject → the first failing chain window j; anchor fails or none fails → reference; else V2 at chain_k[j − 1]. Chain windows the certificate did not run are charged.
- Secondary: **event** (V2 at k_e), always-reference, generate-only; controls (false rejects, cost).

**Tests** (cluster bootstrap over the first object of each scene, 2,000 resamples; ε = 0.10):
- **N1 (risk):** risk(EAF) ≤ risk(cert_ref) + 0.02.
- **N2 (cost):** the saving cost(cert_ref) − cost(EAF) has a 95% lower bound > 0.
- **N3 (timing):** error(chain) − error(EAF) has a 95% lower bound > 0.
- **Confirmed** if N1 and N2 pass. The claim that "fidelity timing beats the 4k timing on contact" needs N3 too.

**Reported regardless:** the cost relative to always-simulate (expected ≈ 1× because most contact queries are rejected), and the event-timed pipeline.

**Falsifiers.** N1 or N2 fails: EAF does not generalise beyond the scenes it was found on, and contribution 2 rests on the 4k repair alone (with 4l/4l-ref reported as negative). N3 fails: fidelity timing is no better than the 4k chain on contact.

**Expected GPU cost:** components ≈ 2.2, generation ≈ 0.5, scenes ≈ 1.3, scan ≈ 0.8, restarts ≈ 4: **≈ 9–10 GPU-h**. Simulation steps run two shards per GPU. The verdict (`code/p4n_summarize.py`) runs on a dev server and computes the true errors itself, so no CPU-only step occupies a GPU node.

**Code check (Oct 16, after pre-registration; NOT the test):** `p4n_summarize.py` run in a sandbox on the already-analysed 4l scenes reproduces the exploratory numbers: EAF error 0.0118, risk 0.014, cost 1.04× vs cert_ref 1.19×; chain 0.0248 / 0.020 / 1.02×. Those scenes informed the design, so only the fresh draws 8–15 count.

### Phase 4n result (Oct 16, 2026; H200 jobs 426353–426356): **NOT CONFIRMED** (N1 FAIL by 0.004; N2, N3 PASS)

347/400 fresh scenes valid (271 contact, 76 control; 17 without placement, 36 failed the sanity rule). No crashes; GPU 84–96% on the simulation steps with two shards per GPU. Tables: `results/p4n_summary/SUMMARY.md`.

| pipeline (271 contact scenes) | error | risk | cost / always-simulate |
|---|---|---|---|
| generate only | 0.186 | 0.748 | 0.10 |
| certify or simulate | 0.003 | 0.010 | 1.16 |
| **EAF repair** | 0.017 | 0.034 | 1.01 |
| chain-gated V2 (4k timing) | 0.028 | 0.026 | 0.99 |
| event-timed V2 | 0.045 | 0.103 | 0.87 |

- **N1 risk: FAIL.** 0.034 vs 0.010 + 0.02 (exceeded by 0.004).
- **N2 cost: PASS.** Saving +0.694 s/query [0.585, 0.808].
- **N3 timing: PASS.** chain − EAF error +0.010 [0.007, 0.014].
- By the rule, EAF is **not confirmed**: lower mean error than the 4k timing, but a heavier tail.
- **Secondary (pre-registered comparator, not a test):** the 4k chain-gated V2 stays within the +0.02 risk margin on these contact scenes (0.026 vs 0.010) at 0.99× vs 1.16×. The confirmed 4k repair carries over to contact.

**Exploratory diagnosis (after the verdict).** EAF has 9 failures, 7 of them restarts.
- In those, the kept generated prefix is already wrong (error up to 0.20; median over all EAF restarts 0.019), or the generated objects never approach although the true ones collide (2 cases).
- The switch time is predicted from the *generation*, which is wrong exactly when it matters. This is the same lesson as "windows certify consistency, not accuracy".
- **Candidate next step:** predict the switch from the *coarse shadow itself* (it runs from the true initial state): switch when the shadow's objects come within 3 coarse cells. Not pre-registered; it needs fresh scenes.

### Phase 4o: shadow-timed fidelity switch (SF) on fresh contact scenes (pre-registered Oct 16, 2026, before any 4o code ran or data existed; user chose this "one more test" option)

**Why.** 4n's EAF failed N1 by 0.004. Its extra failures came from a switch time predicted from the *generation*, which is wrong exactly when it matters. SF takes the switch time from the coarse shadow, which the repair already runs from the true initial state: the cheap simulator decides when it can no longer be trusted. **Agreed stopping rule:** if 4o fails, contribution 2 becomes certificate-gated repair (4k, confirmed in distribution, under shift (4m) and within the margin on contact (4n secondary)), and the event-aware results are reported as negative/partial (this framing was offered to the user on Oct 16).

**Fresh data:** the unchanged 4l construction and validity rules with **new draws j = 16..23** (components, base generations, scenes); the coarse `coarse_cfl` m = 4 scan of the base source.

**Restarts** (`p4j_allk.py`, new option `shadow_switch_dist`):
- One coarse shadow from the true initial state records its state at every model frame 1..22.
- k_sh = the first model frame at which the *shadow's* two objects come within **d_s = 0.46875** (3 coarse cells, as 4n).
- SF restarts the reference simulation at **k_sw = max(k_sh − 1, 1)** from the shadow state there. No restart when k_sh is undefined; if k_sh ≤ 1 the scene falls back to a reference rollout.
- Also run, for comparison: the chain points {3, 7, 11, 15, 19}, the 4n generation-timed point k_s and the event point k_e.
- SF is charged the shadow's substeps only up to 2·k_sw + 1. That is all it needs, although the experiment runs the shadow further.

**Pipelines and tests:** identical to 4n with SF in place of EAF (`code/p4n_summarize.py` with `primary: sf`; contact scenes; cluster bootstrap over the first object; ε = 0.10; τ = 0.02614):
- **O1:** risk(SF) ≤ risk(cert_ref) + 0.02.
- **O2:** the saving cost(cert_ref) − cost(SF) has a 95% lower bound > 0.
- **O3:** error(chain) − error(SF) has a 95% lower bound > 0.
- **Confirmed** if O1 and O2 pass; the "fidelity timing beats the 4k timing" claim also needs O3.
- Secondary: EAF (generation-timed), event-timed, controls; and the 4k chain against the +0.02 margin.

**Expected GPU cost:** ≈ 10 GPU-h (as 4n, plus recording the shadow state per frame), two shards per GPU.

### Phase 4o result (Oct 16, 2026; H200 jobs 426465–426468): **CONFIRMED** (O1, O2, O3 all PASS)

346/400 fresh scenes valid (270 contact, 76 control); no crashes; GPU 77–95% on the simulation steps. The verdict script change was checked: re-running 4n gives an identical SUMMARY. Tables: `results/p4o_summary/SUMMARY.md`.

| pipeline (270 contact scenes) | error | risk | cost / always-simulate |
|---|---|---|---|
| generate only | 0.190 | 0.753 | 0.10 |
| certify or simulate | 0.000 | 0.000 | 1.18 |
| EAF (generation-timed, 4n rule) | 0.011 | 0.014 | 1.04 |
| **SF (shadow-timed switch)** | **0.011** | **0.012** | **1.05** |
| chain-gated V2 (4k timing) | 0.026 | 0.015 | 1.01 |
| event-timed V2 | 0.047 | 0.131 | 0.88 |

- **O1 risk: PASS.** 0.012 vs 0.000 + 0.02.
- **O2 cost: PASS.** Saving +0.649 s/query [0.519, 0.777].
- **O3 timing: PASS.** chain − SF error +0.016 [0.011, 0.020].
- **Contribution 2 (reframed, user-approved Oct 15) is confirmed:** on contact scenes, switching to the reference simulator where the cheap simulator's own state says its grid starts coupling the objects keeps risk within the margin. It is cheaper than certify-or-simulate and more accurate than the 4k timing.

**Honest limits, all to be stated in the paper:**
1. **Cost vs always-simulate.** Contact scenes reject most queries (generation risk 0.75), and generation is charged twice, so every certified pipeline costs ≥ 1× always-simulate here (SF 1.05×). The saving is relative to certify-or-simulate, not to simulating everything.
2. **Most contact scenes fall back to a full rollout.** In 158/270 contact scenes the objects start within 3 coarse cells, so SF falls back to a full reference rollout from the start; the switch acts mid-trajectory in the other 112.
3. **The 4n/4o difference.** EAF (generation-timed) also stays within the margin in this draw (0.014). Its 4n failure was narrow (by 0.004), so the 4n → 4o difference between the two timing rules is small. SF's advantage is principled (it uses the simulator's own state), not large.

### Phase 5a: audited amortisation of the certificate (pre-registered Oct 16, 2026, before any 5a code ran; CPU only, existing data)

**Why this mechanism.** The proposal's amortisation (replay verified windows into the generator) is ruled out: fine-tuning PhysCtrl on verified frames does not propagate (4g, 4g-v2). What can be amortised is the **verifier**. Over a stream of queries, a cheap learned predictor of the certificate score is trained on past probes. When it is confident a query would be accepted, the probes are skipped, so simulator calls per query fall as the stream grows. The contribution's claim (simulator calls per query fall over a stream at controlled risk) is kept; the mechanism changes, and the paper says so.

**Guarantee.** A random **audit** fraction a = 0.10 of queries is always probed. Audited queries are exchangeable with the stream, so they calibrate the skip rule (split conformal on the residuals of the predictor). The predictor is trained only on non-audited probed queries, so the audit residuals are held out. The skip rule is: skip probes iff ĝ(x) + q̂ < log10 τ, where ĝ predicts log10 of the certificate score and q̂ is the conformal (1 − δ) quantile of the audit residuals, δ = 0.05. Then P(the certificate would have rejected | skipped) ≲ δ.

**Data** (all single-object pools with coarse scans, true errors and generations, *excluding* the calibration pool used for τ):
- Phase 4k fresh (800, in distribution);
- Phase 4m fresh (598: 200 in distribution + 398 shifted);
- Phase 2c (300: 100 test_id + 200 OOD).

Total 1,698 queries. The certificate score is the max coarse_cfl m = 4 `spread_event` defect; τ = 0.02614.

**Features (fixed now; no simulation):**
- physics inputs: log10 E, ν, the force coefficient (|drag force| / base drag coefficient), drag-mask fraction, gravity flag, height above the floor;
- generated-sample-0 statistics: centroid displacement at the last frame, max per-point displacement, mean and max per-point speed, max centroid acceleration, bounding-box volume ratio (end / start).

**Learner (fixed):** `sklearn` HistGradientBoostingRegressor(max_iter = 200, random_state = 0) on log10(clip(score, 1e-4, 1)). It is retrained every 100 stream steps. No skipping until ≥ 50 audits and one trained model exist.

**Pipelines on each stream** (costs as 4k: generation, probes by the 4f fit, reference c_ref):
- **cert_ref:** always probe; reject → reference rollout.
- **amortised:** as above; a skipped query is accepted without probes (cost = generation only).

**Streams:** R = 50 random orders of the 1,698 queries (seeds 0..49). Secondary: a drift order (all in-distribution queries first, then all shifted) with seed 0.

**Tests** (over the 50 orders):
- **A1 (amortisation):** probe rate (probed queries / queries) in the first third − in the last third; the mean is > 0 and the 2.5th percentile is > 0.
- **A2 (risk):** the mean of risk(amortised) − risk(cert_ref) is ≤ 0.02, where risk = P(output error > 0.10) over the stream.
- **A3 (cost):** cost(cert_ref) − cost(amortised); the mean is > 0 and the 2.5th percentile is > 0.
- **Confirmed** if A1–A3 all pass.
- **Secondary:** the probe-rate curve over the stream; the risk among skipped queries; the drift-order run (does the rule stop skipping when the shift arrives?); the skip rate per split.

**Falsifier:** A1 fails (no learning) or A2 fails (skips cost risk). Then amortisation is reported as negative, and contribution 3 rests on the efficiency results alone; dropping amortisation from the paper's claims needs the user's approval.

### Phase 5a result (Oct 16, 2026; CPU): **CONFIRMED, small effect** (A1, A2, A3 PASS)

1,698 stream queries (1,100 in distribution, 598 shifted), 50 random orders. Tables: `results/p5a_amortise/SUMMARY.md`.

**Note on the run:** the first run was stopped before computing anything; it was I/O-bound because it read full HDF5 files over NFS. Only the loader changed, to read frame 0 and the scalar inputs; the analysis is unchanged.

- **A1 amortisation: PASS.** The probe rate falls from 0.994 (first third) to 0.848 (last third); difference 0.146 [0.093, 0.195]. By decile: 1.00, 1.00, 0.99, 0.93, 0.89, 0.88, 0.86, 0.85, 0.84, 0.84. Skipping starts after ≥ 50 audits.
- **A2 risk: PASS.** 0.011 vs always-probe 0.011 (difference 0.000 [0.000, 0.001]); risk among skipped queries 0.012.
- **A3 cost: PASS but tiny.** Saving 0.044 s/query [0.027, 0.068], i.e. 1.3% of 3.49 s. Probes are already cheap (≈ 8% of a reference rollout), and the dominant cost is the reference fallback for rejected queries, which cannot be skipped.
- **The rule learned where skipping is safe.** It never skips the hard shifts (E_low, drop, force ×3/ood2_force: skip rate 0.00–0.02). It skips 7–20% of in-distribution and E_high queries.

**Drift order (secondary).** With all in-distribution queries first, then the shifted ones, the shifted part has skip rate 0.13 and risk **0.048** vs 0.008 always-probe. When the stream shifts, the audits calibrated on the old distribution lag behind, and the skip rule's guarantee (which assumes exchangeability) no longer holds. **The paper must state this.** The guarantee is for stationary streams; under drift it needs adaptive/online conformal recalibration. That is a natural extension, not done here.

**Contribution 3 status:** efficiency confirmed (4f, 4k); amortisation confirmed for stationary streams, with a small compute effect (simulator calls −15%, compute −1.3%) and a stated drift limitation. The mechanism is amortising the verifier, not the generator: the generator route failed (4g, 4g-v2).

### Phase 7a: reviewer metrics and literature baselines (pre-registered Oct 17, 2026, before any 7a code ran; CPU only, existing data)

**Why.** The paper needs independent metrics beyond risk and cost, and the baselines reviewers will ask for: a physics-residual trigger (ANCHOR-style) and best-of-N sampling. Everything uses data that already exists.

**Data.** The Phase 4m fresh pool (598 queries: 200 in distribution, 398 shifted), the only pool where every baseline was run. New thresholds are calibrated on the Phase 4m calibration pool (1,200 queries) with the replicate-0 rule used for the window and seed-spread thresholds (ε = α = 0.10). Nothing is recalibrated on the fresh pool.

**New baselines (fixed now):**
- **B1 momentum residual (ANCHOR-style, no simulator).** From generated sample 0 alone. Centre of mass c(t) = Σ vᵢxᵢ(t) / Σ vᵢ (vᵢ = particle volume). Momentum balance predicts ĉ(t) = c₀ + u₀t + ½at², with a = Σⱼ nⱼFⱼ / (ρ Σ vᵢ) + g·[gravity] (nⱼ particles in drag mask j, Fⱼ its per-particle force, ρ = 1000, g = (0, −9.8, 0)); c₀, u₀ from the stored initial state (frame 0 x, v); model frame j is at t = (2j + 1)·417·10⁻⁴ s. Residual R = maxⱼ ‖c_gen(tⱼ) − ĉ(tⱼ)‖. Floor and wall contact forces are not modelled (a residual check cannot know them); internal forces cancel. Gate **resid_ref**: accept sample 0 if R < λ_R, else a reference rollout; cost = generation + rejections × c_ref.
- **B2 best-of-8.** The 8 stored samples per query; output = the selected sample, no simulation; cost = 8 generations. Selection by (i) **medoid** (least mean distance to the other 7) and (ii) **least residual R**. Also (iii) **oracle** (least true error): an upper bound for any selection rule, not a method.

**Metrics.**
- **Selective prediction** for the scores window S (ours), seed spread, residual R and agreement (generation vs a full coarse rollout; secondary, since it costs a coarse rollout). Bad = sample-0 error > 0.10. AUROC, and the risk–coverage curve: selective risk at k = the bad fraction among the k lowest-score queries; **AURC** = mean over k = 1..n. Ties are broken by a fixed random permutation (seed 0); +∞ scores rank last. Query-level.
- **Risk at ε ∈ {0.05, 0.10, 0.20}** for every pipeline, with thresholds unchanged (calibrated at 0.10).
- **Tails:** 90th, 99th percentile and maximum of the per-query mean error; mean **last-frame** error (model frame 23).
- Object-cluster bootstrap, 2,000 resamples, seed 1.

**Tests** (Bonferroni over four: 98.75% two-sided CIs):
- **Q1** (pooled fresh): AURC(window) − AURC(residual), upper bound < 0.
- **Q2** (pooled fresh): AURC(window) − AURC(seed spread), upper bound < 0.
- **Q3** (shift): risk(resid_ref) − risk(gated V2), lower bound > 0. Costs reported either way.
- **Q4** (shift): risk(best-of-8 by residual) − risk(gated V2), lower bound > 0. The oracle best-of-8 risk is reported: if it exceeds 0.10, no selection rule over PhysCtrl's samples can reach our risk.

**Falsifiers.** Q1/Q2 fail: our window score is not the better ranker of generation error, and the paper says so. Q3/Q4 fail: a cheaper simulator-free baseline is as safe; we then report the cost comparison and narrow the claim.

### Phase 7a result (Oct 17, 2026; CPU): Q1, Q2, Q4 PASS; **Q3 FAIL** (the momentum-residual gate is as safe as ours, but costlier in distribution)

598 fresh queries (4m pool). Tables: `results/p7a_metrics/SUMMARY.md`; curves: `risk_coverage.json`. The 4m verdict was reproduced byte-identically after moving its assembly into `p4m_confirm.assemble()` (used here).

**Unit check of the momentum model:** ground-truth residual median 1.0e-4 over 40 gravity-free calibration queries. The one outlier (0.157, `126836_000__id_new_3`) touches the box wall (z = 0.19 vs the collider at 0.2): a contact, which the residual does not model by design. λ_R = 0.01257.

- **Q1 PASS:** AURC(window) − AURC(residual) = −0.063 [−0.086, −0.040] (pooled).
- **Q2 PASS:** AURC(window) − AURC(seed spread) = −0.339 [−0.360, −0.312].
- **Q3 FAIL:** risk(residual gate) − risk(gated V2) under shift = +0.003 [−0.020, +0.025]. The residual gate reaches risk 0.018 under shift (ours 0.015) and 0.000 in distribution (ours 0.025).
  - **Cost decides it:** the residual gate costs 0.79× always-simulate in distribution (ours 0.44×) and 0.97× under shift (ours 0.94×). It is safe because it rejects most queries; true trajectories that touch a floor or wall also violate free momentum balance.
  - **Claim narrowed as pre-registered:** "a simulator-free residual gate is as safe as ours; ours is cheaper in distribution (0.44× vs 0.79×) and similar under shift". It is not "ours is safer".
- **Q4 PASS:** risk(best-of-8 by residual) − risk(gated V2) under shift = +0.681 [+0.644, +0.714]. **Even the oracle best-of-8 has risk 0.676 under shift**, so no selection rule over PhysCtrl's samples can reach our risk; in distribution best-of-8 does not help either (0.110–0.125 vs 0.115 for sample 0).

**Selective prediction (AUROC, bad = error > 0.10):**

| set | window (ours) | agreement (needs a coarse rollout) | momentum residual | seed spread |
|---|---|---|---|---|
| pooled | **0.977** | 0.973 | 0.886 | 0.671 |
| in distribution | 0.895 | **0.930** | 0.737 | 0.873 |
| shift | **0.980** | 0.975 | 0.881 | 0.657 |

**Tails (descriptive):** ours has a heavier tail than certify-or-simulate. In distribution its 99th-percentile error is 0.232 (certify-or-simulate 0.090; maximum 0.442 vs 0.232). The repaired suffix starts from a coarse state, and a few late restarts keep the drift. Under shift the 99th percentiles are 0.140 vs 0.098.

**Other thresholds (thresholds kept as calibrated at 0.10):**
- At ε = 0.20, the order of pipelines is as at 0.10.
- At the stricter **ε = 0.05**, in distribution ours has risk **0.170**, worse than coarse only (0.105) and certify-or-simulate (0.115). Under shift ours is 0.053 (coarse only 0.517).
- The certificate controls the error level it was calibrated for. A stricter target needs its own τ; the paper must say so.

### Phase 7b: PhysCtrl's own metrics on PhysCtrl's test set (pre-registered Oct 17, 2026; descriptive, no test)

**Queries.** The 100 queries of PhysCtrl's released test set (`test_id`). They were part of the Phase 2c development data (signal selection), so 7b is **descriptive**; τ = 0.02614 comes from the calibration pool.

**Pipelines.** Generate only (sample 0); certify-or-simulate; gated V2 with the exact 4k rule (chain windows from the existing 2c coarse_cfl m = 4 scan; restarts from the coarse shadow at the first failing chain point); always simulate.

**Metrics.**
- PhysCtrl's `eval.py`: position MSE in its normalised coordinates ((x − 5)/2), and its `DeformLoss` (`utils/physics.py`, frame_interval 2, norm_fac 5, the ground-truth F and C), also on the ground truth itself as the reference value. Means over queries.
- Ours: mean point error, risk (ε = 0.10), cost.

**GPU job** (`slurm/77_p7b_testset_H200.sh`): for each query, the coarse shadow and the five chain restarts (as 4j/4k), saving the spliced trajectories; DeformLoss on GPU for sample 0, the ground truth and each restart variant. The CPU summary then assembles each pipeline's output by its decision.

### Phase 7b result (Oct 17, 2026; H200 job 426931; descriptive)

100/100 test-set queries; no crashes, no missing restarts; GPU use 80–87% (two shards per GPU). The CPU summary's internal cross-checks passed (saved sample 0 = generation file; assembled per-query errors = the 4k pipeline rule within 2e-3). Table: `results/p7b_testset/SUMMARY.md`.

| pipeline | position MSE (eval.py) | DeformLoss | mean error | risk | cost |
|---|---|---|---|---|---|
| PhysCtrl (generate only) | 1.49e-3 | 0.196 | 0.070 | 0.22 | 0.05× |
| certify or simulate | 3.15e-4 | 0.103 | 0.019 | 0.03 | 0.66× |
| **ours** | 4.23e-4 | 0.135 | 0.034 | 0.03 | 0.52× |
| always simulate | 0 | 0.074 (the ground truth's own value) | 0 | 0 | 1.00× |

- **Reading:** on PhysCtrl's own metrics and test set, ours cuts the position MSE 3.5× and halves the DeformLoss excess over the ground truth (0.121 → 0.061), at 0.52× the cost of simulating everything.
- Certify-or-simulate is more accurate at 0.66×; both have risk 0.03.
- **Sanity check:** the generate-only MSE (1.49e-3 over 100 items) is in line with Phase 0's `eval.py` reproduction (1.97e-3 over 20 items).
- **Status:** descriptive only, because test_id was development data (2c).

### Phase 7c: stricter and looser targets, everything recalibrated (pre-registered Oct 17, 2026, before any 7c code ran; CPU only, existing data)

**Why.** 7a found that at ε = 0.05, with thresholds calibrated for ε = 0.10, ours is worse than coarse only in distribution (risk 0.170 vs 0.105). That compares a system tuned for one target against a stricter one. The fair question is whether ours stays the cheapest safe pipeline when every method is calibrated for the target it is judged on.

**Data.** As 7a: the Phase 4m fresh pool (200 in distribution + 398 shifted), the 4m calibration pool (1,200), the existing chain restarts (all five chain points per query), coarse rollouts, 8 samples and residuals. No new simulation.

**Grid.** ε ∈ {0.05, 0.10, 0.20} × α ∈ {0.05, 0.10, 0.20}. For each cell, every gate is recalibrated on the calibration pool with bad = error > ε:
- the window certificate τ, seed spread and the momentum residual by the replicate-0 rule;
- the agreement (XF) threshold by the pooled rule, as in 4m.

Pipelines: generate only, coarse only, certify or simulate, gated V2 (the 4k rule at the new τ, chain decisions included), XF, seed-spread deferral, residual gate. Risk is measured at the same ε. ε = 0.10 / α = 0.10 must reproduce 4m/7a exactly (code check).

**Tests** (in distribution, at ε = 0.05, α = 0.10; Bonferroni over three: 98.33% two-sided object-cluster bootstrap CIs):
- **S1 (risk):** risk(gated) ≤ risk(certify or simulate) + 0.02 (point estimates, as K1).
- **S2 (cost):** saving certify-or-simulate − gated, lower bound > 0.
- **S3 (vs coarse):** risk(gated) − risk(coarse only), upper bound < 0.

**Descriptive.** For each grid cell and set (in distribution, shift): risk and cost of every pipeline, and the cheapest pipeline among those with risk ≤ α.

**Falsifiers.**
- S1 or S2 fails: the repair's advantage is specific to ε = 0.10, and the paper says so.
- S3 fails: at the strict target, the coarse simulator alone is as safe as ours in distribution.

### Phase 7c result (Oct 17, 2026; CPU): S2 PASS; **S1 FAIL, S3 FAIL** (ours does not meet the strict ε = 0.05 target in distribution)

Table: `results/p7c_targets/SUMMARY.md`. The ε = α = 0.10 cell reproduced the 4m/7a thresholds (checked in code). The 4m verdict again reproduced byte-identically after splitting `calibration_pool()` out of `p4m_confirm.calibration()`.

**Tests (in distribution, ε = 0.05, α = 0.10, all gates recalibrated):**
- **S1 FAIL:** risk(gated V2) 0.125 vs certify-or-simulate 0.085 (+0.02 allowed).
- **S2 PASS:** saving +0.589 s/query [0.440, 0.741].
- **S3 FAIL:** risk(gated) − risk(coarse only) = +0.020 [−0.050, +0.095]. Not safer than the coarse simulator at this target.

**Cheapest pipeline with risk ≤ α (descriptive, 9 cells per set):**
- **Under shift, ours is the cheapest such pipeline in all 6 cells with ε ∈ {0.05, 0.10}** (risk 0.005–0.081 at 0.84–0.99×).
  - At ε = 0.20 the window certificate cannot be calibrated for α ≤ 0.10: too few bad calibration objects, τ = −∞, so it rejects everything. Coarse only qualifies at (0.20, 0.20).
- **In distribution, ours is the cheapest such pipeline only at (0.10, 0.05)** (0.015 at 0.56×).
  - Where α is looser than the generator's or the coarse simulator's own risk, those win trivially, e.g. coarse only at (0.10, 0.10): 0.055 at 0.12×.
  - At **ε = 0.05** ours never reaches risk ≤ α (0.120–0.230). Others do: XF, the residual gate and seed spread in every cell; certify-or-simulate at α ≥ 0.10 (the cheapest at α = 0.10: 0.085 at 0.62×); coarse only at α = 0.20.

**Interpretation (exploratory, not a test):**
- The repaired suffix restarts from a coarse-shadow state, so its own error floor sits near the strict target. Certify-or-simulate, which falls back to a full exact rollout, does not have this floor.
- So ours is designed for, and confirmed at, the ε = 0.10 target in distribution and for both targets under shift. For a strict in-distribution target, the safe choice is a full exact fallback (certify-or-simulate or XF), at higher cost.
- Not checked yet: whether the failing repaired outputs are the late restarts. That needs the per-restart errors; recorded as a candidate diagnostic only.

**Paper consequence:** state the ε dependence plainly; report the grid in the supplement and the shift row in the main text.

### Phase 7d: the missing baselines on the contact benchmark (pre-registered Oct 17, 2026, before any 7d code ran)

**Why.** The contact column of the main table lacks four baselines that 4m has: coarse only, accept-coarse-if-it-agrees (XF), seed-spread deferral, and the momentum-residual gate.

**Data.** The Phase 4o fresh scenes (270 contact + 76 controls), their composed generations (8 samples), the existing 4o verdict rows, and **one new GPU job**: a coarse rollout of each joint scene from its true initial state to the end (`slurm/78_p7d_contact_coarse_H200.sh`, the 4m coarse-only setting, no restarts).

**Thresholds.** Exactly as each gate was used under shift: from the single-object calibration pool, nothing refit on contact scenes.
- window τ = 0.02614;
- seed spread λ (4m);
- agreement λ (4m, pooled rule);
- residual λ_R (7a).

**Pipelines on contact scenes.** Costs follow 4o: two generations, reference c_ref, coarse charged at the 4d rate.
- coarse only;
- XF: accept the coarse rollout if its agreement with the composed sample 0 is below λ, else a reference rollout;
- seed-spread deferral: 8 composed samples; accept sample 0 if the spread is below λ, else reference;
- residual gate: momentum residual of the composed sample 0 (all particles, total centre of mass).

**Prediction from physics (stated before data):** the collision is an internal force, so the total centre of mass of a composed generation that ignores the contact still follows the drag forces. The residual gate is therefore expected to miss contact failures. The test decides.

**Tests** (contact scenes; cluster bootstrap over each scene's first object, as 4o; Bonferroni over three: 98.33% CIs):
- **D1:** risk(coarse only) − risk(SF), lower bound > 0.
- **D2:** risk(seed-spread deferral) − risk(SF), lower bound > 0.
- **D3:** risk(residual gate) − risk(SF), lower bound > 0.
- **XF (decision rule, as M3):** XF dominates SF iff risk(XF) ≤ risk(SF) + 0.02 and cost(SF) − cost(XF) has lower bound > 0. Reported either way.

**Falsifiers.** A failed D test means that baseline is as safe as ours on contact. If XF dominates, the contact claim narrows to "safe", not "cheapest safe".

### Pilots 8a and 8b (defined Oct 17, 2026, before any pilot code ran; EXPLORATORY, development data only)

Context: the second review pass (`notes/audit_literature_2026-10-17.md`); the user said go. Pilots run only on development data that earlier phases already used (Phase 2c: test_id + 4 OOD splits, 300 queries; Phase 4l: 334 contact/control scenes; the calibration pool). Every "go" is re-tested on fresh data under a new pre-registration before it enters the paper.

**Pilot 8a: the fidelity ladder (P1).**
- **Question:** is a medium simulator setting accurate enough to replace the reference (dt 1e-4, grid 125) as the "always simulate" baseline and as our repair tier?
- **GPU job 79, part A:** full rollouts from the true start on the 2c pool and the 4l scenes at three CFL-adaptive settings: M1 (dt ≤ 8e-4, grid 96), M2 (dt ≤ 4e-4, grid 96), M3 (dt ≤ 2e-4, grid 125).
- **Part B:** on the 2c pool, gated-V2 restarts at all five chain points with the suffix at M2, from (B1) the usual coarse shadow and (B2) an M1 shadow.
- **Costs:** substeps × the reference per-substep cost. Phase 1 shows the grid size barely changes the per-substep time.
- **Promising if:**
  - **(a)** some medium setting has risk(> 0.10) ≤ 0.02 in every single-object 2c split at ≤ 0.30× cost. This defines the "always simulate (medium)" baseline.
  - **(b)** gated V2 with a medium suffix (B1 or B2) has risk ≤ that baseline's risk + 0.02, and costs ≥ 20% less in distribution (test_id) and ≥ 10% less on the pooled 2c shift splits.
  - **Contact:** a medium setting's full rollout has risk ≤ 0.02 on the 4l contact scenes.
- **Decision:**
  - (a) and (b): go. Re-test the ladder method on fresh data.
  - (a) without (b): the efficiency claim is in trouble against a tuned simulator. Reposition the paper (certificate / safety layer) and tell the user.
  - Not (a): the reference fallback is justified, and the ladder table goes into the paper as evidence.

**Pilot 8b: a fused certificate score (P2), CPU only.**
- **Question:** does a learned combination of signals accept more good trajectories at the same false-accept rate?
- **Signals:** the three placed window defects (sorted) and their max, the momentum residual, seed spread, and the 5a generation statistics. The coarse rollout is excluded, because it costs a simulation.
- **Data:** the calibration pool split by object into thirds (seed 0): fit (logistic regression, bad = error > 0.10), calibrate (the single-draw object rule, α = 0.10), evaluate. Also evaluated on the 4 OOD 2c splits for false-accepts under shift. The window-only certificate is recalibrated on the same calibration third for a fair comparison.
- **Promising if:**
  - on the evaluation third, acceptance of good trajectories rises by ≥ 0.10 over window-only;
  - FAR stays ≤ 0.10;
  - FAR under 2c shift rises by ≤ 0.02.
- **Go:** confirm on fresh data.

**Pilot 8c (video-level evidence, P3)** is defined after 8a, so its GPU time goes to the more decisive pilot first.

### Pilot 8b result (Oct 17, 2026; CPU; exploratory): **NO-GO**

Table: `results/p8b_fused/SUMMARY.md`.
- The fused score (logistic regression on windows + residual + seed spread + generation statistics) accepts 0.555 of good trajectories on the evaluation third, against 0.503 for window-only: +0.052, below the +0.10 rule.
- Its false-accept rate on the evaluation third is **0.346** (window-only 0.115). Under 2c shift both are 0.008.
- **Why:** the calibration third has only 16 bad objects, so the single-draw threshold is the smallest of 16 scores. That is very noisy, and a learned probability score with a heavy right tail falls through it.
- **Decision:** no further tuning. The window-only certificate stays. The residual and the windows carry the most weight (largest coefficients), consistent with 7a. Kept as an exploratory negative.

### Phase 7d result (Oct 18, 2026; H200 job 427021 + CPU): **D1, D2, D3 PASS; XF does not dominate**

346 fresh 4o scenes (270 contact), no crashed coarse rollouts. Table: `results/p7d_contact/SUMMARY.md`.
- **D1:** risk(coarse only) − risk(SF) = +0.306 [0.240, 0.381]. Coarse only: risk 0.318 at 0.12×.
- **D2:** risk(seed spread) − risk(SF) = +0.074 [0.026, 0.128]. Seed spread: risk 0.086 at 1.66×.
- **D3:** risk(residual gate) − risk(SF) = +0.278 [0.209, 0.352]. Residual gate: risk 0.290 at 0.69×. **As predicted from physics before the data:** a collision is internal, so the momentum residual cannot see an ignored contact. Under shift the same gate was as safe as ours (7a); on contact it is not.
- **XF:** rejects every contact scene, so it is safe (0.000) but costs 1.22×. Saving SF − XF −0.84 s: XF does not dominate.
- **Job note:** GPU use was only 26–32% (coarse-only rollouts are light). Future light jobs should pack more shards per GPU.

### Pilot 8a result (Oct 18, 2026; H200 job 427074 + CPU; EXPLORATORY, development data): rule (a) NOT met; contact rule met by M1 and M3; **a threat to the efficiency claim**

1,902 full rollouts + 600 restart rows, no crashes, GPU 74–85%. Table: `results/p8a_pilot/SUMMARY.md`.

**Part A (full rollouts from the true start; risk > 0.10 / cost):**
- **In distribution (test_id):**
  - M1 (dt ≤ 8e-4, grid 96): **0.000 at 0.13×**
  - M2: 0.010 at 0.25×
  - M3: 0.010 at 0.50×
  - coarse: 0.050 at 0.13×
- **Shift, per split (M1 / M2 / M3):**
  - E_low 0.04 / 0.04 / 0.16
  - force ×3 0.06 / 0.06 / 0.08
  - drop 0.22 / 0.10 / 0.16
  - E_high 0.02 / 0.02 / 0.00 (CFL caps the step, cost 0.40–0.55×)
  - Not monotone in fidelity (M3 is worse than M2 on E_low and drop); cause not investigated.
- **Contact (4l development scenes):** M1 0.020 at 0.13×, M2 0.031, M3 0.004 at 0.50×.

**Rule (a) fails:** no medium setting is safe (≤ 0.02) in every shift split, so by the pre-stated rule the reference fallback is justified under shift.

**But the threat:**
- In distribution, "always simulate M1" has risk 0.000 at **0.13×**. That beats our confirmed 0.44–0.53× (risk 0.025) and coarse only (0.05 at 0.12×).
- On contact scenes, M1 has risk 0.020 at 0.13× (development scenes), against SF's 0.012 at 1.05× (fresh scenes).
- So, against a tuned cheap simulator, **our efficiency claim does not hold in distribution or on contact**. It holds only where medium settings fail silently: under parameter shift, where M1/M2 reach risk 0.055–0.085 pooled and up to 0.22 on drops.

**Part B (gated V2 with an M2 suffix; risk / cost):**
- In distribution: 0.030 at 0.30×, for both the B1 coarse shadow and the B2 M1 shadow. Certify-or-simulate (reference): 0.030 at 0.66×.
- Under shift: B1 0.015 / B2 0.010 at 0.72×. Certify-or-simulate: 0.005 at 0.95×. Always M2: 0.055 at 0.29×.
- So a medium suffix cuts our own cost by 25–55% at about the same risk.

**Interpretation (exploratory):**
- Neither the learned generator nor a tuned cheap simulator is trustworthy under shift: both fail silently.
- The certificate plus suffix repair is what makes either one safe.
- A defensible framing: certify any fast proposal (generator or cheap simulator), repair at the fidelity needed. This changes contribution 3, so **it needs the user's decision** (to be raised now).

### Pilot 8d: certify the cheap simulator itself (defined Oct 18, 2026, before any 8d code ran; EXPLORATORY, development data; the user said "start pilot 8d")

**Why.** 8a showed a tuned cheap simulator, M1 (dt ≤ 8e-4, grid 96), is nearly exact in distribution (risk 0.000 at 0.13×) and on contact development scenes (0.020), but fails silently under shift (pooled 0.085; drop 0.22). If our probes can certify M1's own trajectory and repair it when it fails, the result is a pipeline that is safe everywhere at well below our current cost.

**Pipeline CM1 (certified M1):**
1. **Proposal:** M1 rollout from the true initial state (no generation).
2. **Check:** three short windows at M3 fidelity (dt ≤ 2e-4, grid 125) of m = 4 model frames, restarted from M1's own full state (positions, velocities, deformation gradients, affine velocities; no lifting) at model frames k ∈ {3, 11, 19}. Defect d_k = mean point distance between the window's end and M1's frame k + 4. Score = max d_k. A blown window scores +∞.
3. **Threshold:** M1 has almost no bad trajectories in distribution, so a false-accept threshold cannot be calibrated. Instead, a **novelty threshold**: τ_M1 = the ⌈0.95 (n + 1)⌉-th smallest score over 400 in-distribution calibration queries (id_new_0..3). This bounds false rejections in distribution at about 5%. It does **not** bound false accepts. This is weaker than the generator certificate, and is stated as such.
4. **Repair:** on rejection, keep M1's frames up to the first k whose d_k ≥ τ_M1, and re-simulate the rest at the reference setting from M1's full state at k.

**GPU job 80:**
- the 2c pool (300) and the 4l scenes (334): M1 rollout with states at k ∈ {3, 11, 19}, the three M3 windows, and the three reference suffixes;
- the calibration subset (400): M1 rollout + the three windows only.

Costs are substeps as a share of a reference rollout.

**Promising if** (development data):
- on the pooled 2c shift splits, CM1 has risk ≤ 0.02 at ≤ 0.50× (8a's B2, gated generation with an M2 suffix: 0.010 at 0.72×);
- in distribution (test_id), CM1 has risk ≤ 0.01 at ≤ 0.45×. **Amended Oct 18, before job 80 ran or any 8d data existed.** The original ≤ 0.30× was infeasible by construction: M1 alone costs 0.13× and three M3 windows of 4 frames cost 3 × 0.083×, a floor of about 0.38×. The arithmetic error was found by the sandbox test of the analysis on synthetic rows. 0.45× still beats our confirmed in-distribution pipelines (4k: 0.53×; certify-or-simulate: 0.65×), but not 8a's B2 (0.30×), which is reported next to it;
- on 4l contact scenes, CM1 has risk ≤ 0.02 at ≤ 0.60× (SF on fresh scenes: 0.012 at 1.05×).

**Also reported:** AUROC of the CM1 score for M1 failures; CM1 next to always-M1, always-M2, B2, certify-or-simulate and always-reference.

**Decision.**
- Promising: re-test CM1, and a cascade that tries the generation first, on fresh data under a new pre-registration. Reframe contribution 3 as "certified fast physics", with the user's approval.
- Not promising: keep the current framing and report the tuned-simulator baseline openly.

### Pilot 8d result (Oct 18, 2026; H200 job 80 + CPU; EXPLORATORY, development data): **NOT PROMISING** (shift rule fails)

1,034 rows (300 dev + 334 scenes + 400 threshold queries), no crashes, GPU use 80–90%. τ_M1 = 0.01764. Table: `results/p8d_pilot/SUMMARY.md`.

| set | M1 alone (risk / cost) | CM1 (risk / cost) | rule |
|---|---|---|---|
| in distribution (test_id) | 0.000 / 0.13× | 0.000 / 0.39× | met (≤ 0.01 at ≤ 0.45×) |
| shift (2c pooled) | 0.085 / 0.20× | **0.065 / 0.70×** | **failed** (≤ 0.02 at ≤ 0.50×) |
| contact (4l) | 0.020 / 0.13× | 0.008 / 0.56× | met (≤ 0.02 at ≤ 0.60×) |

**Exploratory diagnosis:**
- False accepts are rare: 1 of 200 shifted queries is a bad M1 trajectory that was accepted. AUROC is 0.84 under shift and 0.89 on contact.
- The leak is in the **repairs**:
  1. **force ×3:** the first failing window is often k = 19, when M1 is already 0.11–0.43 off. The windows pass slow drift, the same lesson as 4h ("windows certify consistency, not accuracy").
  2. **drop:** the window at k = 3 fails correctly, but the reference suffix from M1's state at k = 3 still ends 0.12–0.17 off. M1's grid 96 has already mishandled the floor contact.
- A fix (an earlier or true-start restart) would be tuning after seeing the data, so it is not adopted here.

**Where this leaves the method question (exploratory, development data):**
- The cheapest pipelines with risk ≤ 0.02 per setting are:
  - in distribution: always M1 (0.000 at 0.13×);
  - shift: gated generation with an M1 shadow and M2 suffix, 8a B2 (0.010 at 0.72×);
  - contact: always M1 (0.020 at 0.13×) or CM1 (0.008 at 0.56×).
- No single pipeline is cheapest-and-safe everywhere. B2 is the only one with risk ≤ 0.02 in every development setting below 0.95× (0.30× in distribution, 0.72× under shift). It is not yet run on contact with a medium suffix.

### Pilots E1–E3: making contribution 3 defensible (defined Oct 18, 2026, before any E-pilot code ran; EXPLORATORY; the user said "yes")

Diagnosis and literature: `notes/efficiency_c3_2026-10-18.md`.
- On soft materials we cannot beat a tuned simulator by construction: generation + windows ≈ M1's whole cost.
- Explicit-MPM cost grows with stiffness (CFL), while generation does not. E1 tests that regime.
- Efficiency claims must follow McGreivy & Hakim: compare at matched accuracy against a tuned baseline.

**E1: the stiff-material regime (GPU).**
- **Data:** a new pilot set, `stiff_0` + `stiff_1`, the first 75 test objects each, so 150 queries. E = 10^U(7.5, 9.0), above PhysCtrl's training range of 10^4–10^7; ν and drag in distribution; seeded by qid.
- **Reference:** the 125 grid with dt = min(1e-4, 0.3 dx / c), a CFL-safe extension of PhysCtrl's setting (the backend's new `reference_cfl`, off by default).
- **Ladder:** full rollouts at M1, M2, M3 (CFL 0.5).
- **Generation:** PhysCtrl base, 8 samples.
- **Ours:** the gated pipeline at τ = 0.02614: coarse_cfl spread_event windows; repair = 8a B2 (M1 shadow from the true start, M2 suffix at the first failing chain point).
- **Costs:** A100 seconds, using the 4k cost model (generation 0.236 s; probes by the fit; shadow by the coarse rate; simulator runs as substeps × c_ref / 20,016).
- **Bands:** log10 E ∈ [7.5, 8), [8, 8.5), [8.5, 9]. S* = the cheapest of M1/M2/M3 with band risk ≤ 0.02, else the reference.
- **Promising if,** on the pooled E ≥ 10^8 queries:
  - (i) S* costs ≥ 0.5× the stiff reference (simulation is genuinely expensive here), **and**
  - (ii) ours has risk ≤ 0.02 at ≤ 0.5× S*'s cost.
- **Also reported:** PhysCtrl's generation risk per band; ours vs always-S* per band; idealised latency (E3).
- **Decision:**
  - Promising: confirm on fresh stiff draws with a pre-registration, and add an implicit-MPM check of S* (GeoWarp) if feasible in time.
  - Not promising: contribution 3 becomes the honest cost-of-safety accounting.

**E2: cheaper verification (CPU, existing data).**
- **Variants:** (a) one window, the event window; (b) two windows, event + the middle of the rest; (c) baseline, three m = 4 windows (spread_event); (d) three m = 2 windows (spread_event at m = 2; calibration rows exist for id_new_0..3 only).
- **Calibration:** each variant's threshold by the replicate-0 object rule (α = ε = 0.10) on the calibration pool.
- **Evaluation:** on the 2c development pool — AUROC (pooled), false-accept rate under shift (4 OOD splits), good acceptance in distribution (test_id), probe cost.
- **Promising if:** some variant cuts the probe cost by ≥ 50% against (c), with pooled AUROC ≥ (c)'s − 0.02 and shift FAR ≤ 0.10.

**E3: latency with parallel windows (CPU, idealised, from E1 and 8a rows).**
- **Model:** latency = generation + the longest single window (windows run concurrently) + any serial repair (shadow up to k, then suffix). Compared with the serial wall time of the cheapest accurate simulator S*. Times are substeps × measured per-substep rates.
- **Promising if:** on the E ≥ 10^8 queries, the median latency of ours ≤ 0.5× S*'s.
- **Caveat:** idealised. Concurrency is not implemented or measured; the paper may only state it as an estimate unless it is measured.

### Pilot E2 result (Oct 18, 2026; CPU; EXPLORATORY, development data): **PROMISING** — one window suffices

Table: `results/p8f_cheapcheck/SUMMARY.md`.

| variant | AUROC (2c pooled) | false accept (2c shift) | good accepted (test_id) | probe cost (A100 s) |
|---|---|---|---|---|
| (a) event window only | 0.955 | 0.008 | 0.667 | **0.148** |
| (b) event + middle | 0.959 | 0.008 | 0.667 | 0.295 |
| (c) baseline, 3 × m = 4 | 0.958 | 0.008 | 0.641 | 0.443 |
| (d) 3 × m = 2 | 0.944 | 0.033 | 0.705 | 0.277 |

- **Variant (a)** cuts the probe cost by two-thirds (−0.30 s/query, about 6% of a reference rollout), with AUROC −0.003 and the same false accepts under shift. It passes the rule.
- **Thresholds coincide** for (a), (b), (c) at 0.03007. The same draws are picked, and the decisive low-scoring bad draws take their maximum at the event window. Over all bad calibration draws, the event window is the maximum in only 36%.
- **Caveat:** the baseline threshold here (0.03007) differs from the 4m τ (0.02614) only because the rows are ordered differently, and the single-draw rule's random pick depends on row order. The paper must report the replicate spread (4a: 500 replicates) rather than one draw.
- **Consequence:** confirming the efficiency claim should use the one-window certificate. It matters most in the stiff regime, where windows scale with the CFL limit. E1 itself stays as pre-registered (three windows); a one-window version of the E1 numbers is reported as a secondary.

### Pilot E1 result (Oct 19, 2026; H200 jobs 81–83 + CPU; EXPLORATORY): **NOT PROMISING by the rule (narrow miss); E3 latency promising**

150 stiff pilot queries (log10 E 7.52–9.00; 109 with E ≥ 10^8). All jobs complete, no crashes. GPU use: build 79–89%, sims 85–87%, generation 0–50% (a short job; pack it next time). Table: `results/p8e_stiff/SUMMARY.md`.

**Pooled E ≥ 10^8 (risk / cost in A100 s):**
- generate only: 0.073 / 0.24
- stiff reference: 0 / 14.4
- always M1 / M2: 0.028 / 6.64
- **always M3: 0 / 8.65 (= S*)**
- **ours (3 windows, M2 repair): 0.037 / 4.63** (0.54× S*)
- ours, one-window certificate (secondary): 0.037 / 3.26 (0.38× S*)

**Rules:**
- (i) S* costs ≥ 0.5× the stiff reference: yes (8.65 vs 14.4).
- (ii) ours ≤ 0.02 at ≤ 0.5× S*: **no** (0.037; 4.63 vs 4.32).
- **E1 not promising.**
- **E3:** idealised median latency 0.98 s vs 0.5 × S* = 4.32 s: promising (an estimate; concurrency not measured).

**What the regime shows (as predicted from physics):**
- simulation is genuinely expensive (the stiff reference costs 2.5–3.9× PhysCtrl's own reference);
- PhysCtrl's generations stay accurate (risk 0.05–0.09) and 76% are accepted.

**Failure decomposition (exploratory):** all 4 bad outputs above 10^8 are **repairs** (4 of 27 repaired; restarts at k = 7, 11, 19, 19; errors 0.11–0.90). The certificate accepted **0 of the 8 bad generations**. The M2 suffix is not accurate enough for stiff materials: M2 alone has risk 0.028, and two repairs ended far off. A repair at the accurate setting (M3, which had risk 0 here) would remove that, at higher suffix cost. Using it would be data-informed, so it needs a fresh round-2 test, not a re-analysis of these queries.

### Pilot E1b: stiff regime, round 2 (defined Oct 19, 2026, before any E1b code ran or data existed; EXPLORATORY; the user said "yes"; second and last pilot round)

**Changes from E1, both fixed now from the E1/E2 lessons:**
1. **The one-window certificate (E2):** the event window only (coarse_cfl, m = 4), accepted if its defect < τ = 0.02614. That is the deployed threshold, slightly stricter than E2's calibrated one-window value of 0.03007.
2. **Repair at M3** (dt ≤ 2e-4, grid 125, CFL 0.5), the cheapest setting with risk 0 in E1, instead of M2. The shadow stays M1 from the true start. The chain windows and the first-failure rule are as in 4k. Chain windows already run by the certificate are not charged twice.

**Everything else as E1.** Fresh pilot draws `stiff_2` + `stiff_3`: the first 75 test objects, E = 10^U(7.5, 9.0), new seeds through the split name. The CFL-safe reference (`reference_cfl` 0.3); the M1/M2/M3 ladder; PhysCtrl base generations; the 4k cost model.

**Promising if,** on pooled E ≥ 10^8, with S* = the cheapest of M1/M2/M3 with risk ≤ 0.02 on this set (else the reference):
- S* costs ≥ 0.5× the stiff reference, **and**
- ours has risk ≤ 0.02 at ≤ 0.5× S*'s cost.

**Also reported:** the 3-window variant with M3 repair; the E3 latency estimate; the failure decomposition (false accepts vs repairs).

**Decision:**
- **Promising:** a pre-registered confirmation on fresh stiff draws (with an implicit-MPM check of S* if feasible). Contribution 3 becomes "about 2× cheaper than the best tuned explicit simulator for stiff materials at matched accuracy".
- **Not promising:** stop piloting (two rounds used). Contribution 3 becomes the honest cost-of-safety accounting, with the E1/E1b tables as evidence about where the cost goes.

### Pilot E1b result (Oct 19, 2026; H200 jobs 84–86 + CPU; EXPLORATORY): **NOT PROMISING** — piloting for contribution 3 stops (two rounds used)

150 fresh stiff pilot queries (101 with E ≥ 10^8). All jobs complete, no crashes. GPU use 80–92% (generation 12–33%). Table: `results/p8g_stiff/SUMMARY.md`.

**Pooled E ≥ 10^8 (risk / cost in A100 s):**
- S* = always M3: 0 / 8.65
- always M1/M2: 0.040 / 6.64
- stiff reference: 0 / 14.4
- generate only: 0.079 / 0.24
- **ours (one window, M3 repair): 0.059 / 3.23 (0.37× S*)**
- three windows, M3 repair: 0.050 / 4.90

**Rules:**
- (i) yes;
- (ii) **no** (risk 0.059 > 0.02; cost 0.37× passes).
- E3 latency estimate: median 1.04 s vs 4.32 s (promising, idealised).

**Failure decomposition (exploratory):**
- The M3 repair fixed E1's problem: **0 of 21 repairs bad**.
- All 6 bad outputs are **certificate false accepts**: 6 of 8 bad generations were accepted, five of them at log10 E 8.85–8.97, with errors up to 0.98 and window defects of only 0.017–0.025. The three-window score accepted the same five.
- E1 + E1b pooled: **6 of 16 bad generations at E ≥ 10^8 accepted**, against the 4m shift false-accept rate of 0.012.
- **Reading:** at very high stiffness a generated trajectory can be locally consistent (near-rigid, small drift per window) while globally wrong, and the threshold calibrated on soft materials does not transfer.
- **New limitation of contribution 1, to be stated in the paper:** the certificate's guarantee is in distribution, and it leaks under this stiffness shift.

**Decision (pre-agreed stopping rule):**
- No more efficiency pilots.
- Contribution 3 becomes the honest accounting of what safety costs:
  - confirmed savings against exact simulation (4k, 4f);
  - "cheapest safe pipeline" against untuned shortcuts (4m, 7a, 7c, 7d);
  - exploratory gains from cheaper verification (E2: −67% probe cost) and medium-fidelity repair (8a);
  - an explicit statement that a tuned simulator is cheaper for soft materials (8a);
  - and that the stiff regime, where generation is structurally cheaper, needs a stiffness-aware certificate (E1/E1b).
- The E1/E1b tables go to the supplement as the analysis of where the cost goes.

### Phase 9: confirmation of the improved pipeline on fresh data (pre-registered Oct 19, 2026, before any Phase 9 code ran or data existed; the user asked for it)

**What is confirmed: "ours-v2", the pilot-proven cheaper pipeline.**
- **Certificate:** the one-window certificate (E2): the event window, coarse_cfl, m = 4, accept iff defect < τ = 0.02614.
- **Single objects:** on rejection, the gated chain of 4k (anchor + windows at k ∈ {3, 7, 11, 15, 19}; the certificate's window is not charged twice). The repair is an **M2 suffix** (dt ≤ 4e-4, grid 96, CFL 0.5) from an **M1 shadow** (dt ≤ 8e-4, grid 96, CFL 0.5) run from the true start (8a B2). The anchor failing or no chain failure → full reference rollout.
- **Contact scenes:** the shadow-timed switch of 4o (coarse shadow, k_sw = max(k_sh − 1, 1), d_s = 0.46875) with an **M3 suffix** (dt ≤ 2e-4, grid 125, CFL 0.5). The repair setting is fixed by the 8a development data: M3 is the cheapest setting with risk ≤ 0.02 on the 4l contact scenes (0.004); M2 had 0.031. No switch point → full reference rollout.
- **Costs:** the 4k cost model (A100 seconds; generation 0.236 s; probes by the 2c fit; shadow by the coarse per-substep rate; simulator runs and suffixes as substeps × c_ref / 20,016).

**Fresh data** (all new draws, seeded by qid through the split name):
- **In distribution:** `id_new_22`, `id_new_23` (100 test objects each, 200 queries).
- **Shift:** `ood3_E_low` (E 10^[3.0, 3.9]), `ood3_E_high` (10^[7.1, 7.8]), `ood3_force` (drag 0.3–0.8 coefficient), `ood3_drop` (gravity, floor 0.1–0.5 below). These are the ood2 distributions, new draws; 400 queries.
- **Contact:** two-object scenes of draws 24–27 (the 4l/4o design; about half of 4o's size), with controls.

**Baselines on the same queries:**
- generate only;
- coarse only (coarse_cfl full rollout);
- always M1, always M2, always M3 (the tuned simulators);
- always reference;
- certify-or-simulate (the same one-window certificate, a full reference rollout on rejection);
- momentum-residual gate (λ_R from 7a);
- seed-spread deferral (λ from 4m).
- On contact, additionally the 4o pipelines from the same rows, plus always M1/M3.

**Tests, single objects** (object-cluster bootstrap, 2,000 resamples; Bonferroni over five: 99% two-sided CIs):
- **C1 (ID risk):** risk(ours-v2) ≤ risk(certify-or-simulate) + 0.02 (point estimates, as K1).
- **C2 (ID cost vs exact):** upper CI bound of mean cost(ours-v2) ≤ 0.35 × c_ref.
- **C3 (ID saving):** cost(certify-or-simulate) − cost(ours-v2), lower bound > 0.
- **C4 (shift safety):** risk(ours-v2) ≤ 0.03 (point), and risk(always M1) − risk(ours-v2) lower bound > 0. The tuned cheap simulator is unsafe under shift and ours is not.
- **C5 (shift saving):** cost(certify-or-simulate) − cost(ours-v2), lower bound > 0.

**Tests, contact** (the 4o verdict code, 95% CIs, test prefix CC):
- **CC1:** risk(ours) ≤ risk(certify-or-simulate) + 0.02.
- **CC2:** saving vs certify-or-simulate, lower bound > 0.
- CC3 (timing vs the 4k chain) is reported but not required.

**Confirmed if** C1–C5 and CC1–CC2 pass.

**Reported either way:**
- **in distribution, always-M1 is cheaper** (8a). Stated as the price of safety, not hidden;
- risk–cost tables per set;
- E3 latency estimates.

**Amendment (Oct 19, before job submission and before any Phase 9 data existed):**
- **What triggered it:** the code check of the contact verdict with `cert_windows: one` on the 4o scenes (seen data) gave SF risk 0.095 and certify-or-simulate 0.083. The one-window certificate misses ignored-contact failures: the event window sits at the strongest *generated* event, and a generation that ignores the contact has none there. E2 had validated the one window on single objects only, so using it on contact was an untested extrapolation.
- **Change 1:** contact keeps the **three-window** spread_event certificate confirmed in 4o. `cert_windows: one` is removed from `p9c_summary.yaml`.
- **Change 2:** both relative risk tests gain an **absolute bound**, so a weakened comparator cannot make a weak pipeline pass:
  - C1 also requires risk(ours-v2) ≤ 0.035 in distribution;
  - CC1 also requires risk(ours) ≤ 0.03 on contact.
- Single objects keep the one-window certificate as pre-registered.

**Code checks before submission (Oct 19; not the test):**
- `p9_confirm.py` was run end to end on the 2c development pool, with 8a's B2 rows standing in for the ours-v2 repairs (seen data; output deleted). Every test computed.
- The builder's `ood3_` family re-creates the ood2 draws unchanged (40/40 stored 4m queries match).
- `p4n_summarize` with `cert_windows: one` was run on the 4o scenes (seen data; output deleted).

**Falsifiers:**
- C1/C4/CC1 fail: the cheaper repair is not safe enough. The paper keeps the 4k/4o reference-suffix numbers, and ours-v2 goes to the supplement.
- C2 fails: the efficiency headline stays at the confirmed 0.53×.

**Result (Oct 19, 2026; jobs 428997–429003; verdicts `results/p9_confirm/SUMMARY.md`, `results/p9c_summary/SUMMARY.md`): CONFIRMED.**
- **Data:** every array task exited 0. 600/600 single-object queries, no unstable ground truth. Contact: 168 valid scenes (132 contact, 36 controls; 32 of 200 excluded by the 4l validity rule); 0 missing restarts.
- **Single objects (99% CIs):**
  - C1 PASS: ours-v2 risk 0.020 vs certify-or-simulate 0.015 (limit +0.02) and ≤ 0.035.
  - C2 PASS: 1.219 s [0.984, 1.465] = 0.26× reference; bound 1.633 s.
  - C3 PASS: saving +1.296 s [+1.030, +1.553].
  - C4 PASS: shift risk 0.013; always M1 is riskier by +0.090 [+0.055, +0.128].
  - C5 PASS: shift saving +0.756 s [+0.617, +0.905].
- **Contact (95% CIs):**
  - CC1 PASS: SF 0.012 vs 0.000 (limit +0.02) and ≤ 0.03.
  - CC2 PASS: saving +1.310 s [+1.029, +1.601].
  - CC3 PASS: error(4k chain) − error(SF) +0.014 [+0.008, +0.020].
  - SF costs 0.90× always-reference (4o with a reference suffix: 1.05×).
- **Risk / cost (× always-reference):**

| pipeline | in distribution | shift | contact |
|---|---|---|---|
| generate only | 0.130 / 0.05 | 0.680 / 0.05 | 0.693 / 0.10 |
| coarse only | 0.040 / 0.13 | 0.367 / 0.16 | — |
| always M1 | 0.015 / 0.13 | 0.102 / 0.20 | 0.030 / 0.13 |
| always M2 | 0.020 / 0.25 | 0.102 / 0.29 | 0.038 / 0.25 |
| always M3 | 0.015 / 0.50 | 0.155 / 0.51 | 0.000 / 0.50 |
| momentum-residual gate | 0.010 / 0.77 | 0.007 / 0.97 | — |
| seed-spread deferral | 0.000 / 1.11 | 0.247 / 1.06 | — |
| certify or simulate | 0.015 / 0.54 (one window) | 0.003 / 0.87 (one window) | 0.000 / 1.18 (three windows) |
| **ours-v2** | **0.020 / 0.26** | **0.013 / 0.71** | **0.012 / 0.90** (SF) |

- **Reported as pre-stated:** in distribution, always M1 is cheaper at equal risk.
- **New, also reported:** on contact, always M3 is cheaper (0.000 at 0.50×). M3 was the setting picked on the 4l development scenes, and it held on fresh scenes. Under shift, however, M3 fails (0.155).
- **The claim the data supports:** under shift, ours-v2 is the cheapest pipeline with risk ≤ 0.03. The next cheapest safe pipeline is certify-or-simulate at 0.87×. Every tuned simulator setting is unsafe there (0.10–0.16), so a setting chosen on development data cannot be trusted on a query whose distribution is unknown.
- **Descriptive additions (Oct 19, `code/p9_table.py` → `results/p9_table/`; no new test):**
  - Object-weighted contact ladder (the verdict printed it query-level): M1 0.029 (4/132 bad) at 0.13×, M2 0.035, M3 0.000 at 0.50×. M1 is therefore just inside the 0.03 bound on contact.
  - Certificate on the generations: bad accepted 3/26 in distribution, 1/272 under shift, 0/87 on contact; good accepted 105/174, 84/128 and 1/45.
  - One-window validity: E2's calibrated one-window threshold is 0.03007 ≥ τ = 0.02614, so the deployed rule's accept set is inside the calibrated one and the object-level bound carries over.
  - Cost-model check: the pre-registered model charges the M1 shadow at the coarse per-substep rate and the M1–M3 baselines at c_ref/20,016 per substep. The measured per-substep time barely depends on the grid at 2,048 particles (A100: coarse 0.229 ms vs reference 0.233 ms; H200 Phase 9: 0.29–0.32 ms across 64³–125³), so the two charges are consistent.
  - On contact, the 4k chain timing (with the M3 suffix) has the same risk as SF (0.012) at 0.67× vs 0.90×, with twice the mean error (0.029 vs 0.015). The shadow-timed switch buys accuracy, not risk, on this test set.
- **Exploratory observation (not a test):** the tuned-simulator ladder is non-monotonic on ood3_E_low and ood3_force: M3 has risk 0.24 / 0.27 vs M1 0.02 / 0.20. This suggests step-size sensitivity of soft, strongly driven objects. The reference's own convergence was not tested there, which is a limitation of the ground truth under those shifts.

### Phase 10 (pre-registered Oct 20, 2026, before any Phase 10 code ran or data existed; the user asked for both)

**10a: Is the reference simulator converged at our error scale?** The user doubted the ground truth. Phase 9 also showed a warning sign: M3 was worse than M1 on ood3_E_low and ood3_force.
- **Runs:** the reference setting (grid 125) at half its time step (5e-5) on every Phase 9 query (600 single objects and all 168 contact scenes), and at a quarter (2.5e-5) on a fixed subset. The subset is the first 25 queries per single-object split and the first 7 scenes per contact draw, in manifest order. A rollout's distance to the reference, d = mean over frames and points (the paper's error), estimates the reference's own time-step error. The grid is not refined: with 2,048 particles a finer grid leaves cells empty, so that test would mislead.
- **R1 (converged):** a set counts as converged at our scale if the 95th percentile of d_half is ≤ 0.02 (a fifth of ε). Sets: in distribution, each shift family, contact scenes. Reported per set: median, 95th percentile, maximum of d_half and d_quarter.
- **R2 (claims robust):** the per-point distance obeys the triangle inequality, so for any pipeline output |e_finer − e_ref| ≤ d_half. That gives each pipeline's risk an interval against the finer reference without re-running any pipeline. Claims hold if:
  - (a) ours' upper risk bound is ≤ 0.03 on all three sets;
  - (b) under shift, the lower bound of every tuned setting (M1, M2, M3) is > 0.03.
  - Self-check built into the script: with d = 0 the bounds must reproduce the Phase 9 table exactly.
- **R3 (analytic, drop family only):** before the lowest point reaches the floor, the centre of mass should fall as y0 − ½ g t² with the simulator's own g. The check passes if the maximum deviation over those frames is ≤ 0.01 (descriptive).
- **R3 amendment (Oct 20, after its first output; descriptive check only):** the free-frame rule "lowest point more than one grid cell above the floor" counted frames after landing, because MPM bodies rest about one cell (0.084) above the floor. On those frames the prediction ½gt² runs away (max 19.5 on 2 of 100 queries; median 1e-4). The rule now uses two cells, the reach of the grid interpolation (1.5 cells). Tolerance unchanged.
- **Outcomes either way:**
  - R1 passes: the paper says so with the numbers.
  - R1 fails on a set: report it; R2 decides whether a claim changes. Where R2 fails, the paper states the claim against the finer reference, or drops it for that set.
- **Cost estimate:** about 3.8 GPU-h (H200, about 0.3 ms per substep).

**10b addition (Oct 20, before any 10b data; after the paper review):** report PhysCtrl's own headline metrics as well:
- vIoU (voxel 0.1 over [-1.5, 1.5]³ in model coordinates), Chamfer distance and the correspondence MSE. These are computed with PhysCtrl's `src/volume_iou.py` functions, imported by path, per frame and averaged, as in its Table 2.
- Pipelines: generate only, certify-or-simulate, ours, always simulate.
- Descriptive (development data).
- The same code also runs on the 7b outputs (earlier pipeline) as a secondary row.
- Consistency note: PhysCtrl's reported correspondence MSE (0.0015, 25 steps) and our 7b generate-only MSE (1.49e-3, 4 steps) agree.

**Recorded while preparing 10a (Oct 20, CPU, existing data; descriptive):**
- The tuned settings' frames last int(0.0417/dt)·dt = 0.0416 s, not 0.0417 s, the same expression PhysCtrl uses. So M1–M3 are compared up to about 5 ms early by the last frame.
- The reference sampled at their frame times gives an offset error with median 0.003–0.009 and a maximum of 0.065 (M1, ood3_force).
- Subtracting it from every tuned-simulator error (a lower bound) leaves the shift risk at about 8.8% (M1), 9.3% (M2) and 13.5% (M3), still far above 0.03. In-distribution risks are unchanged.
- Claims unchanged; disclosed in the paper's setup. Phase 10a's half/quarter steps divide the frame exactly (834 / 1,668 substeps).

**10b: PhysCtrl's own metrics with the final pipeline.** 7b (descriptive) used the earlier pipeline (three windows, coarse shadow, reference suffix). The paper describes the Phase 9 pipeline, so its numbers must come from that pipeline.
- **What is re-run:** the same 100 test-set queries (development data, descriptive only) with:
  - the one-window certificate (τ = 0.02614);
  - the M1 shadow and the M2 suffix at the chain restarts;
  - the same metrics: PhysCtrl's position MSE and DeformLoss, our error, risk and cost.
- **Reporting:** 7b stays as a secondary row (earlier pipeline). No test.

### Phase 10 results (Oct 21, 2026; jobs 430617 (96) and 430618 (97); verdicts `results/p10a_convergence/SUMMARY.md`, `results/p10b_testset/SUMMARY.md`)

**10a: the reference is NOT converged at our scale except for stiff materials (R1 FAIL on 6 of 7 sets; R2 FAIL; R3 PASS).**
- All rows present: 600 + 168 half-step rollouts, 150 + 28 quarter-step; 1 unstable half-step rollout (ood3_E_low) and 1 unstable quarter-step; no crashes. GPU 90–95%.
- 95th percentile of d_half: in distribution 0.026, stiffer 0.009 (converged), softer 0.200, drop 0.115, stronger push 0.427, contact 0.044. The quarter step moves the reference further (median ratio ≈ 2), so these trajectories are sensitive to the time step rather than converging.
- Share of queries where the reference itself moves by more than ε = 0.10: in distribution 1%, stiffer 0%, contact 0%, softer 21%, drop 7%, stronger push 18%.
- R2 triangle bounds:
  - (a) ours' upper bound: 0.035 in distribution, 0.138 under shift, 0.031 on contact. FAIL.
  - (b) lower bounds under shift: M1 0.020, M2 0.015, M3 0.033. FAIL.
- R3: free fall exact (max 1.2e-4).
- **Exploratory reading (computed after the verdict, Oct 21):**
  - Under shift, the tuned simulators' errors are about the size of the reference's own time-step sensitivity. Softer: M1 median 0.029 vs d_half 0.030.
  - Only 6 of M1's 40 shift failures occur where the reference is stable (d_half < 0.05): 0/2 softer, 5/18 drop, 1/20 stronger push.
  - PhysCtrl's failures are robust: 173 of its 266 shift failures occur where the reference is stable, and its median error (0.18–0.49) is 6–20× the reference's sensitivity.
- **Consequence for the claims, per the pre-registered rule ("the paper states the claim against the finer reference, or drops it for that set"):** the claims that rest on PhysCtrl's failures, the certificate and ours' own accuracy stand. The claim "every tuned simulator setting fails under shift" (and hence "ours is the cheapest safe pipeline under shift" and "the only one safe everywhere below 1×") is **not supported** at our ε for softer materials and stronger pushes. It is partly supported for drops. How the paper restates it is the user's decision, because it weakens contribution 3.

**10b (descriptive; development data):** on PhysCtrl's 100-object test set with the final pipeline:

| pipeline | vIoU | Chamfer | MSE | risk | cost |
|---|---|---|---|---|---|
| PhysCtrl | 77.57% | 0.0028 | 1.49e-3 | 0.220 | 0.05× |
| ours | 86.81% | 0.0008 | 4.05e-4 | 0.030 | 0.24× |
| certify or simulate | 92.20% | 0.0006 | 3.30e-4 | 0.030 | 0.57× |

### Phase 10c: exact re-scoring against three reference time steps (pre-registered Oct 21, 2026, before any 10c code ran or data existed; the user chose "reframe + extra robustness run")

**Why.** Phase 10a showed the reference moves at our ε for soft materials, strong pushes and drops, and its R2 bounds are worst case. 10c re-scores every Phase 9 pipeline exactly against references at three time steps.

**Runs (GPU, positions saved to `$BIG_ROOT/p10c_traj`), on every Phase 9 query (600 single objects, 132 contact scenes):**
- The reference setting (grid 125) at dt 1e-4 (stored ground truth), 5e-5 and 2.5e-5. The steps divide the frame exactly.
- Each pipeline's output with its Phase 9 decisions (fixed now from the Phase 9 records, written by a CPU step before the GPU job):
  - generation (stored sample 0);
  - always M1 / M2 / M3; coarse only;
  - ours (accept → generation; repair → M1 shadow to k + M2 suffix for single objects, coarse shadow to k_sw + M3 suffix on contact; fallback → reference at dt);
  - certify-or-simulate, the residual gate and seed spread (generation or the reference at dt, by their Phase 9 decisions);
  - always reference (= the dt rollout).
- Recomputed outputs must reproduce the stored Phase 9 per-frame errors against dt within 2e-3 (GPU MPM is not bit-reproducible). A larger deviation stops the job.

**Scoring (CPU).** Risk = P(error > 0.10), object-weighted as in Phase 9, against each reference r ∈ {dt, dt/2, dt/4}. Primary reference for the reworded paper: dt/4, the finest. The others are shown for sensitivity.

**Tests (95% object-cluster bootstrap, 2,000 resamples; Bonferroni over three test families → 98.3% CIs):**
- **T1 (ours vs PhysCtrl):** in each set (in distribution, shift, contact), risk(PhysCtrl) − risk(ours) > 0 against every reference.
- **T2 (ours is safe):** risk(ours) ≤ 0.03 against every reference, per set (point estimates).
- **T3 (tuned simulators under shift):** for each of M1, M2, M3, risk(M) − risk(ours) under shift > 0 against every reference.
  - If it fails for a setting, the paper states that this setting is as safe as ours under shift within the reference's uncertainty.
- Also reported: mean errors, per-family risks, and the reference-vs-reference distances.

**Cost estimate:** about 9–10 GPU-h (H200).

**Amendment (Oct 21, after job 430784; the reproduction check only, not a test):**
- One shard stopped: the recomputed M3 rollout of 36885_000__ood3_force deviated 0.0029 from its Phase 9 error, over the 2e-3 limit. All other 715 recomputed queries were within 0.0012.
- GPU MPM is not bit-reproducible, and 10a found the stronger-push family time-step sensitive.
- The job now stops only above 1e-2 and records every deviation above 2e-3 (`above_tol` in rows.jsonl). The summary reports their count.
- The remaining 17 queries are re-run with `--resume`.

### Phase 10c results (Oct 21, 2026; job 101 (430784 + rerun 430885); verdict `results/p10c_summary/SUMMARY.md`)

- 732/732 queries re-scored; reproduction against the stored Phase 9 errors: max deviation 0.0022, one M3 rollout above 2e-3 (none above the 1e-2 stop). Unstable: one M3 rollout (counted as wrong) and one half-step and one quarter-step reference (left out of those columns).
- **T1 PASS:** PhysCtrl is riskier than ours in every set against every reference (98.3% CIs): in distribution +0.070 to +0.110, shift +0.434 to +0.667, contact +0.675 to +0.710.
- **T2 FAIL:** ours ≤ 0.03 holds against dt in every set and against every reference in distribution, but not under shift against dt/2 (0.108) and dt/4 (0.216), nor on contact against dt/4 (0.062). Always simulating at dt has 0.113 / 0.226 under shift and 0.037 on contact against the same references.
- **T3:** M1 PASS (+0.090 to +0.116), M3 PASS (+0.108 to +0.142), **M2 FAIL** (+0.090 against dt; −0.009 [−0.045, +0.026] against dt/2; +0.010 [−0.013, +0.035] against dt/4). By the pre-registered rule the paper states that M2 is as safe as ours under shift within the reference's uncertainty (M2 costs 0.29× vs ours 0.71×).
- Pre-registered primary reference dt/4 (risk): ours 0.030 / 0.216 / 0.062 (ID / shift / contact); always reference at dt 0.020 / 0.226 / 0.037; M1 0.025 / 0.329 / 0.082; M2 0.015 / 0.226 / 0.041; M3 0.030 / 0.324 / 0.130; PhysCtrl 0.100 / 0.650 / 0.737.
- **Exploratory (defined after the verdict; `code/p10c_explore.py`, `results/p10c_explore/`):**
  - Gap to always simulating at dt, risk(pipeline) − risk(always reference), per set and reference: ours ≤ +0.020 in distribution, ≤ +0.013 under shift, ≤ +0.025 on contact (max +0.025). Max over sets and references: certify-or-simulate +0.015 (1.18×), M2 +0.102, M1 +0.111, M3 +0.155, seed spread +0.247, residual gate +0.290 (contact from 7d), coarse +0.367, PhysCtrl +0.722. Ours is the cheapest pipeline within +0.03 (0.90× max cost).
  - Robust risk (wrong against all three references) under shift: ours 0.005, M1 0.102, M2 0.048, M3 0.157, coarse 0.209, seed spread 0.241, PhysCtrl 0.647. M2's robust failures are in drops (0.07) and strong pushes (0.09).
- **Paper (reframed Oct 21, the user's "reframe + extra robustness run"):** Table 1 keeps the pre-registered Phase 9 risks against dt; its "All three" columns are now the max gap (exploratory) and max cost, replacing worst risk / safe; teaser (b) plots the same. New Table 3 (`tab/reference_table.tex`) shows every main pipeline against dt, dt/2, dt/4 plus robust risk; the full table is in the supplement. Abstract, intro, §5.1, §5.4, §5.5, limitations and conclusion restated: PhysCtrl's failures are robust to the reference; ours tracks the full simulator at 0.26–0.90× its cost; tuned settings are cheaper in known regimes and M2 matches ours under shift within the reference's uncertainty. Moved to the supplement for space: the certificate figure, the frame-time-offset sentence and most of "What did not work".

### Phase 10d: are the "strands" numerical? (pre-registered Oct 22, 2026, before any 10d code ran; the user said "do B" to recommendation R2 of the simulator audit)

**Why.** In 6 of the 23 shifted Phase 11 items, the reference pulls the pushed patch into a long strand while the body stays put (`notes/simulator_audit_2026-10-22.md` §1.3). The six (sponge, jelly, chicken, star, snake, dog chew) were picked by viewing the frames, so their selection is post hoc and stated as such. Phase 10a found the reference not converged in this soft, strong-push regime. Question: do the strands survive finer time steps and finer space?

**Body-preserving refinement.**
- PhysCtrl's particles are surface samples, and their volumes come from the 125³ grid: the body is the occupied cells, a one-cell shell (Phase 12 amendment). Refining the grid with the same particles would halve the shell's thickness, i.e. change the object.
- The spatial refinement therefore fills each occupied 125-grid cell with 3×3×3 regularly spaced particles of volume dx³/27. Shape and mass are identical to the reference body.
- Push on the filled body:
  - the particles inside the same 0.4 box;
  - per-particle force = (coeff·direction) × base_drag_coeff of the filled body, where coeff·direction = stored drag_force / base_drag_coeff(original);
  - so the total force equals coeff × weight, as in the reference.

**Settings (all 23 shifted Phase 11 items; the 17 non-strand items are controls):**

| Setting | Body | Grid | dt |
|---|---|---|---|
| R0 | stored reference | 125 | 1e-4 |
| R1 | stored reference | 125 | 5e-5 |
| R2 | stored reference | 125 | 2.5e-5 |
| R3 | filled body | 125 | 1e-4 (filling alone) |
| R4 | filled body | 250 | 5e-5 (finest) |

**Measures at model frame 11** (simulator frame 23), on each setting's own particles, mass-weighted:
- stretch = longest-axis extent (1st–99th percentile) / initial extent;
- left-behind share = mass fraction displaced by less than 0.2 object sizes;
- also Chamfer distance to R0 (object-size units).

**Decision (median over the 6 strand items; fixed now):**
- **Converged:** |stretch(R4)/stretch(R0) − 1| ≤ 0.15, |left-behind(R4) − left-behind(R0)| ≤ 0.10, and the same two conditions for R2 vs R0. The strands are a property of the specified material and force at higher resolution.
- **Numerical:** median stretch(R4) ≤ 1.5 or median left-behind(R4) ≤ 0.05. The strands are largely a resolution artefact.
- **Otherwise: partly numerical.**
- R3 vs R0 isolates particle filling; R4 vs R3 isolates the grid. The controls should change little; their changes are reported.

**What it means:**
- Converged → the stretching is what this MPM model predicts for the specified (unrealistic) material; the realism issue is the parameters, not the numerics.
- Numerical → part of PhysCtrl's "failure" on these items and in the softer family is reference noise. The softer-family claims are qualified accordingly.
- Either way this is MPM vs MPM. Phase 12's independent FEM can add a descriptive view of the same items afterwards.

**Cost:** A100, about 23 items × 4 runs; the filled body has about 26k particles. Estimated < 1 GPU-h; GPU usage logged.

### Phase 10d results (Oct 22, 2026; A100 job 431955; verdict `results/p10d_summary/SUMMARY.md`)

- **Run:** 23 valid items × 4 settings, no unstable runs, GPU 88%.
  - The two items recorded invalid in Phase 11 (unstable reference: rubber duck, giraffe teether) still had query files from before that rule and were simulated too. The summary excludes them, as pre-registered ("the 23 shifted items"): `cases()` now skips invalid cases.
- **Verdict by the pre-registered rule: PARTLY NUMERICAL.** On the 6 strand items:
  - |stretch ratio − 1| is 0.020 (R4/R0) and 0.075 (R2/R0), both within 0.15;
  - |Δ left-behind| is 0.064 (R4) but 0.113 (R2), above the 0.10 limit;
  - the "numerical" condition is not met (median stretch at R4 is 2.95, left-behind 0.38).
- **Reading:**
  - The stretching itself is resolution-stable. Median stretch is 2.95 at R0 and at the finest R4, and every strand item stays ≥ 2.8 at R4.
  - The mass share left behind is time-step sensitive: sponge 0.49 → 0.27, chicken 0.35 → 0.13 at dt/4.
  - Particle filling alone at the same grid (R3) lowers the stretch for some items (star 1.33, jelly 1.48), but the finer grid (R4) restores it. Grid and particle-density effects partly cancel.
- **Controls:** median |stretch ratio − 1| R4/R0 = 0.016. Chamfer R4–R0 is 0.010–0.036 for the controls and 0.018–0.110 for the strand items.
- **Consequence:** the strands are what this MPM model predicts for the specified soft material and concentrated push, not a resolution artefact. The realism issue lies in the parameters (audit §1.2), not the numerics. The detached-mass detail is not converged in time step; that matches Phase 10a and is stated with it.

### Phase 10e (EXPLORATORY, Oct 24, 2026): the MPM domain wall (`code/p10e_wall.py`, `results/p10e_wall/SUMMARY.md`)

**What was found.** While diagnosing 14a, a survey showed that PhysCtrl's MPM setup has sticky "cut" walls 0.2 from every side of the [0, 10]³ domain (`generate_mpm_data.py` lines 152–159, copied by our bridge). Under a strong push the object reaches them inside the clip.
- Rule (fixed before scoring): wall reach = a reference (dt) particle within 0.5 of the boundary; k_w = the first such model frame.
- Wall reach: in distribution 4% (7/200), shift 33% (132/400): `ood3_force` 99%, `ood3_E_low` 32%, `ood3_E_high` 1%, drop 0%; contact 0%. Median first wall frame 15–16. Objects stop at about 0.16 / 9.84 or rebound.
- The 14a candidate pool: 74% (ACT arm 84%); PhysCtrl's own range (the 4g in-distribution re-simulations): 2.9% of all 3,000 (corrected Nov 3: the Oct 24 inline survey reported 1.3% on an unsorted "first 600" subset; `code/p14a_diagnose.py`, `results/p14a_diagnose/`).

**Re-scoring of the Phase 10c trajectories (risk vs reference at dt, object-weighted, 95% CIs):**

| set | pipeline | full clip | wall-free queries | truncated before the wall |
|---|---|---|---|---|
| shift (pooled) | PhysCtrl | 0.680 | 0.525 (n 268) | 0.640 (n 400) |
| | M1 / M2 / M3 | 0.102 / 0.102 / 0.155 | 0.066 / 0.047 / 0.052 | 0.050 / 0.040 / 0.075 |
| | ours | 0.013 | 0.003 | 0.007 |
| ood3_force | PhysCtrl | 1.000 | (n 1) | 0.840 |
| | M1 / M2 / M3 | 0.20 / 0.16 / 0.27 | — | 0.01 / 0.02 / 0.08 |
| ood3_E_low | M1 / M2 / M3 | 0.02 / 0.13 / 0.24 | 0.00 / 0.00 / 0.04 (n 68) | 0.00 / 0.02 / 0.11 |
| ood3_drop (no walls) | M1 / M2 / M3 | 0.18 / 0.11 / 0.11 | same | same |

**Reading (exploratory):**
- PhysCtrl's failures do not depend on the walls (wall-free shift 0.525; force before the wall 0.84).
- Most tuned-simulator failures in the force and soft families are wall impacts. Before the wall they nearly vanish (force: M1 0.20 → 0.01). The non-monotonic ladder (M3 worse than M1) is a wall effect.
- What remains for the tuned simulators under shift is the drop family (floor contact, also a "cut" collider).
- So "every tuned simulator fails under shift" is mostly "they fail at collisions with PhysCtrl's walls and floor". That is consistent with Phase 10a: the reference itself is not converged for the stronger push, 0.43.
- Ours stays the safest pipeline below 1× on every reading, but its margin over M2 under shift shrinks to 0.04 (truncated) or 0.044 (wall-free), and comes almost entirely from drop.
- **Consequence for the paper (needs the user; it weakens contribution 3 further):** the walls are part of PhysCtrl's convention, but they are invisible in the video. The shift result should be stated per family, with the wall share disclosed. Possible claim: "contact events (walls, floor, objects) are where tuned simulators fail"; "beyond PhysCtrl's push range" would then be the wrong description.

### Phase 11: video-level evaluation of the existing method (pre-registered Oct 20, 2026, before any Phase 11 code ran or data existed; the user said "do all" to the paper-review plan, item ★5)

**Why.** Our evaluation is at the trajectory level. CVPR video papers report a VLM judge and a small preference study on 6–60 cases (PhysCtrl: GPT-4o on 12 videos and 35 raters; NEXUS: 6 cases; PhysFlow: 60). The method is unchanged: this phase only renders and scores what the Phase 9 pipeline outputs.

**Inputs (fixed now).**
- PhysCtrl's 5 elastic examples, already rendered (job 95: as given and 3×).
- Plus 25 synthetic images: FLUX.1-schnell (open licence), 4 steps, 1152×768 resized to 720×480 (amended before any code ran: PhysCtrl's examples are 720×480 and DAS resizes to that, so a square image would be distorted), seed = index, prompt "a photo of a {object} standing on a wooden table, centered, whole object visible, plain light background, studio lighting". The objects are fixed now:
  rubber duck, plush teddy bear, soft rubber toy dinosaur, foam sponge, jelly dessert, stuffed bunny, rubber chicken toy, silicone spatula, plush octopus, soft toy car, rubber boot, foam letter A, squishy star-shaped stress ball, rubber frog toy, plush dog, stuffed elephant, giant gummy bear, plush cactus, rubber giraffe teether, plush fish, foam pillow, rubber snake toy, plush mushroom, stuffed cat, rubber dog chew toy.
- An image is excluded, and the exclusion recorded, only if PhysCtrl's own preprocessing fails (SAM/SV3D/LGM) or the drag box is empty. No visual curation.

**Physics per image (rule, fixed now; elastic).**
- log10 E = 4 + (i mod 3); ν = 0.3.
- Push at the highest point (drag_mode max, axis y).
- Direction by i mod 3: up (0, 1, 0), up-left (−1, 0.3, 0), up-right (1, 0.3, 0).
- Two strengths:
  - **in range:** force_coeff 0.06, inside the training range 0.02–0.2;
  - **shifted:** force_coeff 0.40, inside the ood3_force range 0.3–0.8.
- DAS prompt: "the {object} is pushed {direction phrase} by an invisible force and moves naturally".

**Pipelines.**
- PhysCtrl (generation, sample 0).
- Ours: the Phase 9 single-object pipeline (one-window certificate, τ 0.02614, M1 shadow, M2 suffix).
- Reference simulation.
- All rendered with PhysCtrl's own tracking and DAS, seed 0. When ours accepts, its video is PhysCtrl's.

**Scoring.**
- **(a) Trajectory error**, as everywhere.
- **(b) VLM judge.**
  - Model: Qwen2-VL-7B-Instruct (local; no data leaves the cluster), greedy decoding.
  - PhysCtrl's own GPT-4o prompt (its Appendix A), verbatim except that the force is given in the text, not as a red arrow. It is scored per video: input image + 10 evenly spaced frames, SA/PC/VQ 1–5.
  - Method identity is hidden and the order randomised per case (seeded).
  - The reference video is scored too, as a sanity check.
- **(c) Preference study** (2AFC, a web page the user shares with lab members).
  - Each item is one rejected case: PhysCtrl vs ours, left/right randomised.
  - Questions: "Which video moves more like the reference shown below?" and "Which video looks more physically plausible?" (asked first, without the reference).
  - Raters are blind to the method.

**Tests (on the shifted cases where ours rejected the generation; descriptive on in-range cases; 95% bootstrap CIs over cases):**
- **V0 (judge validity):** PC(reference) − PC(PhysCtrl) > 0. If it fails, the judge cannot see the physics differences; V1 is then reported as uninformative, not as a loss.
- **V1 (VLM):** PC(ours) − PC(PhysCtrl) > 0.
- **V2 (human; only if ≥ 10 raters):** the share preferring ours (physical plausibility) > 50%, CI clustered by rater.
- Also reported: SA/VQ, the in-range cases, the trajectory error per case, and the exclusions.
- Figure cases are chosen by the median-error rule among rejected shifted cases.

**Cost estimate:** about 15–20 GPU-h (25 preprocessings, at most 150 renders, judging).

**Amendment (Oct 21, after render job 430886; a validity rule, not a test):** the pre-registration had no rule for an unstable reference simulation. At force_coeff 0.40 the reference blew up (non-finite from model frame 9) for two of the softest objects (log E = 4): rubber duck and rubber giraffe teether. As for unstable ground truth in every earlier phase (4m, 9), such a case is excluded and counted, never rendered or judged: `p6_video.py` writes `case.json` with `invalid` (the two existing records were converted; the NaN record is kept as `case_v2_unstable_reference.json`), and `p11_summary.py` lists it among the exclusions. The two shards that stopped at these cases (0 and 2; 7 objects not yet run) are re-run.

**Renders complete (Oct 21, job 431046 for shards 0 and 2; GPU 14–40%):** 48 valid Phase 11 cases (25 in range, 23 shifted) plus the 10 PhysCtrl-example cases of job 95, all three variants rendered (174 videos); 2 excluded (unstable reference, above). Frames extracted on crimv3mgpu015. Trajectory level, descriptive:
- In range (force 0.06): 16 accepted, 9 repaired; PhysCtrl median error 0.040 (3/25 above 0.10); ours median 0.028, 1/25 above 0.10 (silicone spatula, accepted at 0.111).
- Shifted (force 0.40): all 23 rejected at the anchor window and simulated in full; PhysCtrl median 0.333 (23/23 above 0.10). So on these cases ours' video is the reference video, and V1 is close to V0 by construction; the PhysCtrl 3× cases (job 95, repaired mid-clip) are the ones where ours differs from both.

**Judge result (Oct 21; job 431265; verdict `results/p11_summary/SUMMARY.md`):** 174/174 videos scored and parsed (a second submission, 431397, found nothing left to do).
- **V0 FAIL:** PC(reference) − PC(PhysCtrl) = +0.07 [−0.07, +0.21] on the 28 rejected shifted cases. By the pre-registered rule, **V1 (+0.07, same CI) is uninformative, not a loss.**
- Means on the rejected shifted cases: PC 4.21 (PhysCtrl) / 4.29 (ours) / 4.29 (reference), although PhysCtrl's mean trajectory error there is 0.507 against 0.0045 for ours. In range: 4.60 / 4.60 / 4.63.
- **Exploratory diagnosis (after the verdict):** the judge is degenerate on this task. It produced only 3 distinct score triples over 174 videos; SA = PC in every row; in 51/58 cases all three videos get the same PC; for the 15 cases where PhysCtrl's error is above 0.3 it ranks the reference higher in only 3. It tracks the object and image, not the motion (Spearman between PhysCtrl's trajectory error and its PC −0.29).
- Figure case by the pre-registered rule: p11_10_rubber_boot__coeff_0.4.
- **Consequence:** the paper reports that a 7B open VLM judge with PhysCtrl's prompt could not separate the reference from PhysCtrl, so it cannot evaluate physics here. V2 (the human study) is the remaining video-level test.

**Preference study: page and analysis rules (fixed Oct 21, before any response existed):**
- **Items:** all 28 rejected shifted cases (23 Phase 11 images at force 0.40, 5 PhysCtrl examples at 3×), built by `code/p11_study_build.py` (seed 11). Candidate videos get opaque names (`<item>_a/_b.mp4`), the reference is `<item>_r.mp4`; which letter is ours is stored only in `results/p11_study_design/mapping.json`, never published.
- **Page:** a claude.ai artifact the user shares with lab members at Contributor level (signed in). Each rater sees all items in a per-rater random order, with left/right randomised per rater and item. Q1 "Which video looks more physically plausible?" is answered before the reference appears; Q2 "Which video moves more like the reference?" after. Answers are saved per item in `responses/<rater id>`; raters cannot read each other's answers. The owner's view shows only progress counts, never answers, so interim results cannot steer the stopping time.
- **Stopping:** the analysis runs once, when the user closes the study or on Nov 7, 2026, whichever comes first.
- **Analysis (`code/p11_study.py`):** every answered item counts, no exclusions. V2 = share of Q1 answers preferring ours, 95% CI by bootstrap over raters; PASS if ≥ 10 raters answered and the lower bound is above 0.5; with fewer than 10 raters V2 is not run and the study is reported as descriptive. Also reported: Q2, and the two item groups separately. On the 23 Phase 11 items ours' video is the reference video, so there Q1 compares PhysCtrl with the reference simulation, and Q2 is trivially answerable; only the 5 PhysCtrl items test a mid-clip repair.
- **Study page published (Oct 21):** https://claude.ai/artifact/EG8oUGHPCvmqrkJbkbKF8c (private; the user shares it with raters as Contributors). Access rules checked once: a Contributor cannot read another rater's document (probe written, listed at `interact`, deleted). No responses yet.
- **Amendment (Oct 22, before any response was saved):** in the user's own view the page received storage but no account identity (`user` capability unavailable), so nobody could save. Raters without an account id now save under a random anonymous code kept in their browser (`anon/<code>`). Consequences, stated before data: (i) in that store raters at Contributor level could read other raters' documents through the browser's developer tools (the page never shows them); (ii) one person using two browsers counts as two raters, and the rater-clustered CI treats them as two. The owner can no longer always be detected, so the intro offers a "show progress counts" link and the instructions tell the owner not to rate. `p11_study.py` reads both stores (documents keyed by account id or code). Write access for Contributors and no read access for view-only visitors checked with a probe (deleted).
- **Amendment 2 (Oct 22, still before any response; the user asked for raters without a claude.ai account):** the page no longer needs a sign-in. Every rater's answers are kept in their own browser, and the last screen shows an answer code (`SOMW1-<rater>-<q1 q2 left per item, page item order>-<FNV-1a checksum>`, holding only a/b letters) to send to the user by email or chat. Signed-in raters still save to the store too and get the code as a backup. `p11_study.py --codes FILE --items items.json` reads the codes: a code with a wrong checksum or shape is rejected and listed; a rater present in both the store and a code keeps the more complete record. Consequences, stated before data: the user collects the codes, so a rater who never sends one is missing; a rater who restarts in a new browser gets a new code and counts as a new rater. Round trip checked in a scratch folder on codes produced by the page's own JavaScript (two complete, one partial, one corrupted: 3 raters read, the corrupted one rejected).
- **Interim look (Oct 22, at the user's request):** the answers of the first rater (1 code, 28/28 items) were shown to the user, descriptively. Recorded here so it is reported; the stopping rule (close or Nov 7) and the analysis are unchanged by it.
- **Simulator audit (Oct 22, exploratory, after the interim look):** `notes/simulator_audit_2026-10-22.md`. No bug (reference = PhysCtrl's data generator). Six of the 23 shifted Phase 11 items are in a stretching regime (log E 4 or small push patch: strands while the body stays); two reference videos contain renderer artifacts; 83–85% of the trajectory error accrues after objects leave the view. Phase 9: 32% of ood3_E_low queries stretch > 2× (PhysCtrl wrong in all of them, 53% outside). Recommendations R1–R5 await the user; nothing changed in the study or the paper.
**Downloads, run by the user on the setup host:** FLUX.1-schnell and Qwen2-VL-7B-Instruct into `$BIG_ROOT/hf_cache`.

### Phase 12: independent-simulator cross-check (pre-registered Oct 22, 2026, before any Phase 12 code ran or second-simulator data existed; the user asked whether ours' advantage is circular)

**Why.** On 52% of shifted and 57% of contact queries ours outputs the reference itself, and repairs re-simulate 61–75% of the frames with the same MPM code (`notes/simulator_independence_2026-10-22.md` §1). A reference that shares no code with PhysCtrl's MPM is needed to test "closer to physics", as distinct from "closer to our own solver".

**Second simulator S2 (fixed now):**
- PolyFEM built from source at commit 008a1aeae97b277c8bcf644adf0673749b7889ef; material `FixedCorotational` with the query's E, ν and density 1000; implicit time integration with Newton solved to a relative gradient norm ≤ 1e-8.
- **Fallback order, if the build fails or S0a/S0c cannot be met:** (1) libuipc (`pyuipc`) with stable neo-Hookean, calibrated μ, λ from E, ν (the large-strain difference is stated); (2) report the check as infeasible. No other substitute.

**Mesh (fixed rule):**
- Nodes = the query's 2,048 initial points.
- Delaunay tetrahedralisation, removing tetrahedra whose circumradius exceeds 2 × the median nearest-neighbour distance.
- Density scaled so that total mass = 1000 · sum(vol).
- A query is excluded (counted, reported) if the mesh volume differs from sum(vol) by more than 30% or more than 1% of nodes are unattached.

**Loads:**
- Drag queries: the MPM per-particle force vector f on every drag-mask node, for the whole clip; no gravity.
- Drop queries: gravity 9.8 on all nodes; floor plane at the query's floor height with IPC contact, friction μ = 0.5 (an approximation of the sticky grid floor, stated as a difference).

**Output:** node positions at simulator frames 2, 4, …, 48 (model frames 1–24, as the stored ground truth).

**Time step:**
- The largest dt in {frame/4, frame/8, frame/16} (frame = 0.0417 s) whose result changes by less than 0.005 median point distance when dt is halved.
- Chosen on 10 development queries only (Phase 4 calibration pool id_new_0..11, never a test set), before any test query runs.

**Gate S0 (development queries only; all must pass before any test query runs):**
- S0a free fall: the centre of mass of 5 development drop queries, before floor contact, within 1e-3 of ½gt².
- S0b small strain: on 20 development drag queries with log E ≥ 6, median MPM–S2 distance over model frames 1–8 ≤ 0.02.
- S0c push check: centre-of-mass acceleration over the first frame equals coeff·g within 2% on those 20 queries.
- **If S0 fails:** no cross-check conclusion. Reported as "the two solvers disagree at our scale"; the paper's claims stay fidelity-only (see the table in the note).

**Sets:**
- Primary (single-object drag): Phase 9 in distribution (200), ood3_E_high (100), ood3_E_low (100), ood3_force (100).
- Secondary: ood3_drop (100), with the contact-model difference stated before data.
- Descriptive: the 23 Phase 11 shifted items.
- Excluded: two-object contact (sticky grid contact vs IPC; a different design would be needed).

**Pipelines re-scored against S2 (exact, from the stored job-101 trajectories):** PhysCtrl, ours, the MPM reference at dt, M1–M3, coarse, certify-or-simulate, residual gate, seed spread. Risk = P(error > 0.10), object-weighted.

**Tests (object-cluster bootstrap, 2,000 resamples; Bonferroni over the 4 primary sets → 98.75% CIs):**
- **X0 (descriptive):** MPM–S2 disagreement per set: median and p95 distance; share above 0.10.
- **X1:** risk_S2(PhysCtrl) − risk_S2(ours) > 0 in every primary set.
- **X2:** Δ_S2 / Δ_MPM ≥ 0.5 in every primary set (point estimates; CIs reported), where Δ = risk(PhysCtrl) − risk(ours).
- **X3 (circularity):** on fallback queries pooled over the primary sets, share where d(MPM, S2) < d(PhysCtrl, S2) > 0.5 (lower CI bound > 0.5). Repairs are reported separately (descriptive).

**Interpretation (fixed now):**
- X1 and X3 PASS → the advantage is not an artifact of scoring with our own solver for single-object drag scenes.
- X3 FAIL → no "more accurate than PhysCtrl" claim beyond fidelity.
- X1 FAIL → fidelity-only claims; discussed with the user before any paper change.

**Cost:** CPU only (dev server or `cpu-all`), 0 GPU-h; runtime measured in the S0 runs and recorded before the test runs.

**Clarifications before any query ran (Oct 22, from a synthetic-cube smoke test; no project data):**
- The mesh rule also drops tetrahedra with near-zero volume (|V| < 1e-6 of the mean).
- The initial acceleration is set to the applied body force per mass, to avoid the zero-start offset measured in the smoke test.
- Gravity is written as `rhs = [0, +9.81, 0]` (PolyFEM's sign convention).
- The solution is exported at the original nodes.
- These are implementation corrections, not changes to tests or thresholds.

**Amendment (Oct 22, before any S0 or test query was simulated in full; found in a 3-frame code trial on one development query):**
- **Mesh rule replaced.** PhysCtrl's 2,048 points are surface samples: its data generator samples mesh faces with face normals (`generate_mpm_data.py`), and the median spacing is 0.045. The MPM therefore simulates a one-cell-thick shell, and sum(vol) = n_occupied_cells × dx³ exactly (dx = 0.08; checked on 5 development queries).
  - A Delaunay mesh on the points gave a thinner body (42% of the volume). Bending stiffness grows with thickness³, so it would compare different objects.
  - New rule: the mesh is the occupied grid cells, each split into 6 Kuhn tetrahedra, with density 1000 (mass matches exactly). Particles are embedded barycentrically in their cell (embedding error 0).
  - A query is excluded if the largest face-connected set of cells holds < 99% of the particles. 1 of 5 development objects fails this.
- **Push as a box body force.** The stored drag_mask equals the box |x − drag_point| < 0.4 on all 10 development queries checked. The push is a body force on that box, sized so the total force equals n_masked·f. The same field is the initial acceleration.
- **Frames:** outputs are at simulator frames 1, 3, …, 47, which is what the stored ground truth x[1::2][:24] holds. The earlier text said 2, 4, …, 48, which was wrong.
- **S0c uses the mass-weighted centre of mass.**
- **Trial (code check, not S0):** on one development query, MPM–S2 distance 0.0002 at frame 1 and 0.0012 at frame 3. Runtime about 2.4 s per time step with 8 threads, i.e. about 7 / 13 / 26 min per full query at dt = frame/4, /8, /16.
- Tests and thresholds are unchanged.
- **Portable binary:** the first build used `-march=native` (AVX-512 on the dev server) and could fail on `cpu-all` nodes. Rebuilt as `build_portable` with `-march=x86-64-v2`; S0 and the test both use it.
- **Code test of `p12_s0.py --smoke`:** 1 development query per step, 3 frames, output in a scratch folder, never `results/`. It ran end to end with the portable binary. These numbers are a code check, not the S0 gate.
- **Job 431597 (S0, first submission) failed before any simulation.** The cpu-all node's glibc and libstdc++ were older than the Ubuntu 22.04 build host's (GLIBC_2.33 / GLIBCXX_3.4.30 missing). Nothing was computed.
- **Rebuilt as `build_conda`.**
  - Toolchain: conda-forge gcc 15.3 with the glibc 2.17 sysroot, -march=x86-64-v2.
  - The bundled core-math library calls `roundeven` (glibc ≥ 2.25), so a 3-line shim defines it as `rint`, which is identical in the default rounding mode (`tools/polyfem/shim/roundeven.c`).
  - The binary now needs glibc ≥ 2.14. The job puts the env's libstdc++/libgomp first on LD_LIBRARY_PATH, sizes workers from `nproc` (the node gave 8 cores, not 32), and has a 24 h limit.
  - The smoke test reproduces the earlier build's numbers exactly (scratch only).
- S0 is ready: `sbatch slurm/102_p12_s0_cpu.sh`.

- **S0 run 1 (job 432388, Oct 23–24): FAIL.**
  - dt rule n_div = 4 (change 0.0003).
  - S0a PASS (max 1.9e-7).
  - S0b PASS (median 0.0047; per-query max 0.056).
  - **S0c FAIL** (max relative error 0.064 vs 0.02; 8/20 above 0.02).
  - Runtime 4.8 min per query (8 threads).
- **Diagnosis (CPU, mesh only, no simulation):** the box force was normalised by the masked particles' volume, but PolyFEM applies it to the mesh volume inside the box. On the 20 S0b queries the two differ by 0–5%, and |ratio − 1| correlates 0.85 with the S0c error. A second, smaller part comes from measuring the centre of mass on the particles: in a FEM the mass is in the mesh, and the particles are non-uniform surface samples. The gate caught an implementation bug in the load, not a disagreement between solvers: S0b passes.
- **Amendment (Oct 24, before any test query):**
  - The push region is the drag box snapped to the grid cells whose centre lies inside it. The field is then constant on every tetrahedron, so quadrature integrates it exactly, and the body acceleration is n_masked·f / (ρ·V_snapped): the total force equals the MPM's exactly. The snapped volume / masked-particle volume ratio is 0.85–1.07 on the 20 queries; the total force is exact either way, and the force spreads over the snapped box.
  - S0c is measured on the mesh's own centre of mass (exact for linear tetrahedra); the particle-weighted one is reported as secondary.
  - The dt rule, S0b and S0c are re-run with fresh tags; S0a (gravity only, untouched code path) is reused. Thresholds unchanged.
  - **One rerun only:** if v2 fails, the pre-registered S0-FAIL outcome stands. Run 1 stays in `results/p12_s0/`; v2 writes `results/p12_s0_v2/`.
  - Code test (`--smoke`, one query, 3 frames, scratch only): mesh centre-of-mass error 0.0012, particle-weighted 0.0078. This is a code check, not the gate.
  - Ready: `sbatch slurm/102_p12_s0_cpu.sh` (about 6–10 h on an 8-core node).
- **Note for the test runs (fixed now):** Phase 10e found that 99% of `ood3_force` references reach PhysCtrl's walls. PolyFEM has no walls, so the force family is compared only on frames before the first wall frame (p10e rule), and E_low likewise per query. The X tests are otherwise unchanged.

- **Test runs prepared (Oct 29, before any test query; `code/p12_test.py`, `code/p12_summary.py`, `slurm/114_p12_test_cpu.sh`):**
  - The test job runs only if `results/p12_s0_v2/s0.json` says S0 PASS, at its n_div. Otherwise it writes nothing (the pre-registered rule).
  - Queries: the 500 primary Phase 9 queries (in distribution 200, `ood3_E_high`, `ood3_E_low`, `ood3_force`), from `p10c_decisions`.
  - **Deviations from the Oct 22 text, fixed now, before data:**
    - The secondary drop set is **not run**: our PolyFEM runner has no floor contact (IPC floor not implemented).
    - The 23 descriptive Phase 11 items are not run.
    - Both are reported as not run.
  - Mesh-rule exclusions and solver failures are recorded, never retried, and reported as missing.
  - Comparison frames: before the MPM reference's first wall frame (p10e rule, recorded Oct 24).
  - Tests X0–X3 exactly as above (98.75% object-cluster CIs).
  - Code test of the verdict on fabricated FEM positions (scratch only, deleted). The test list resolves to 500 queries.

### Phase 13: real-world anchor on PokeFlex drops (pre-registered Oct 22, 2026, before data access; the user requested access the same day)

**Why.** Every result so far is measured against a simulator. The simulator audit and the circularity analysis showed this cannot say whether the simulator, or anything scored against it, is physically right. PokeFlex (Obrist et al., arXiv 2410.07688) has real elastic objects with measured robot pokes and drops from rest, reconstructed in 3D by a 106-camera rig.

**Main question:** is PhysCtrl's MPM (our reference) closer to real motion than PhysCtrl's generator? Secondary: does an independent FEM (Phase 12's PolyFEM) agree with reality as well as the MPM does?

**Data (fixed now):**
- All PokeFlex objects with both poke and drop sequences.
- **Pokes are used only to fit the material; drops are used only to test.**
- An object is excluded (counted, reported) if its reconstruction cannot be sampled to 2,048 surface/volume points, or if its poke fit fails the rule below.

**Scale (fixed rule):**
- Each object is normalised to size 1 as in PhysCtrl, with s = its real longest side in metres.
- In simulator units, gravity = 9.81/s and density = 1000. Forces are scaled so that the measured poke force / object weight is preserved.
- E is fitted in these simulator units, so pokes and drops use one consistent scaling.

**Material fit (pokes only):**
- ν = 0.3 fixed. log E from {2.0, 2.25, …, 7.0}.
- Each poke is simulated with PhysCtrl's MPM at the reference setting: the measured force history applied to the particles within the PhysCtrl drag box (half-size 0.4) around the measured contact point.
- Score: mean symmetric Chamfer distance to the reconstructed point clouds over the poke, in object-size units.
- Chosen E = the minimiser of the mean over the object's pokes.
- Fit-failure rule: the best mean Chamfer exceeds 3 × the object's repeatability floor (below).
- The robot poke is a kinematic contact; applying it as a force on a patch is an approximation, stated.

**Drop prediction (blind; E fixed from the pokes):**
- Initial state: the first reconstructed frame at rest, sampled to 2,048 points; zero velocity.
- Floor plane from the data. Times t_k = k × 0.0834 s (PhysCtrl's model-frame spacing), up to the end of the 1 s capture.
- Predictors:
  1. PhysCtrl MPM at the reference setting;
  2. PolyFEM (Phase 12 settings; reported only if Phase 12's S0 gate passed);
  3. PhysCtrl-large with gravity and floor conditioning (25 steps, sample 0);
  4. PhysCtrl-base cannot take gravity or a floor, so it is not run; this is reported as a scope limit.
- Our pipeline is built on base, so it is not run here. Phase 13 tests the reference and the generator, not ours.

**Metrics:**
- Per-frame symmetric Chamfer distance to the real point cloud (object-size units), averaged over frames.
- Also: centre-of-mass height error, time of first floor contact, first rebound height.
- **Repeatability floor:** Chamfer between two real drops of the same object, after aligning their release times.

**Tests (object level; object-cluster bootstrap, 2,000 resamples; Bonferroni over 2 → 97.5% CIs):**
- **P1:** Chamfer(PhysCtrl-large, real) − Chamfer(MPM, real) > 0. The MPM reference is closer to reality than the generator.
- **P2:** median over objects of Chamfer(MPM, real) / repeatability floor ≤ 2. The reference is realistic to within twice the capture's own drop-to-drop variability.

**Descriptive:**
- FEM vs real and FEM vs MPM;
- sensitivity to E ± 0.5 in log;
- per-object table, including fit failures.

**Interpretation (fixed now):**
- P1 and P2 PASS → the reference is a realistic target for drops of these objects. Scoring against it is justified there.
- P1 FAIL → the generator is as close to reality as the reference, so "closer to the simulator" does not mean "closer to reality". The paper claims fidelity only.
- P2 FAIL → the reference is not realistic to within capture noise. Reported as a limitation of all simulator-scored claims.

**Cost (estimate):**
- MPM poke fits: 21 E values × 4–8 pokes × objects, with 5–6 s pokes. About 50–60k substeps each at about 0.23 ms per substep (A100) gives about 12–14 s per simulation, so roughly 2–4 A100-hours.
- Drops: minutes. PhysCtrl-large: minutes on H200. PolyFEM: CPU.

**Format details** (file layout, mesh correspondence, floor estimation) are fixed after access, before any simulation or comparison is run. They are recorded here as amendments.

**Access (Oct 29):** a working link is in `../PokeFlex/url.txt` (personal; never copied into the project).
- Sizes read from the server with header-only requests (`notes/pokeflex_file_sizes_2026-10-29.tsv`): 54 drop takes 25.9 GB, 116 poke takes 562.5 GB, 588 GB in all (the readme says about 500 GB).
- Download script: `slurm/00d_download_pokeflex.sh format|drops|pokes` (dev server; size-checked; nothing unzipped). Not started; it waits for the user's go.
- Plan: `format` (Sponge, 5.2 GB) first, so the format details above can be fixed before any simulation; then `drops`; then `pokes`, which may be reduced by extracting only the files the fit needs.

### Phase 14: certificate-driven simulator distillation (Option A). Pilot 14a pre-registered Oct 22, 2026, before any 14a code ran; the user chose "A+B" (`notes/model_paper_options_2026-10-22.md`)

**Idea.** PhysCtrl base is fine-tuned on reference simulator rollouts for conditions it gets wrong. The window certificate (probes only) decides which candidate queries are simulated and kept: active acquisition. The trained generator is then used alone, and the certificate stays a held-out safety net, never a training objective (lesson of 4c). This is materially different from 4b–4g-v2:
- supervised targets for new conditions;
- original-distribution replay against forgetting;
- no inference-time constraints and no frame conditioning.

**Pilot 14a (development data only; its purpose is the kill decision and choosing the training variant):**

*Training objects:* the 100 objects of PhysCtrl's `sample/` set (as in 4g). No overlap with the 100 test objects used for all evaluation.

*Candidate pool:*
- 3,000 queries, 30 draws per object, split family `aug_<j>` (j = 0..29), seeded by (qid, split).
- Physics: log₁₀E ~ U(3.0, 8.0); ν ~ U(0.05, 0.45); drag coefficient ~ U(0.02, 0.8) × base_drag_coeff.
- The drag point is a uniformly drawn particle, with PhysCtrl's 0.4 drag box as the mask. The direction comes from the hemisphere around the outward radial direction, as in `id_new_`.
- No gravity, floor at 0.2: base has no gravity input, so drop is out of scope for A.
- Every candidate is simulated at the reference setting; unstable rollouts are excluded and counted. This gives both arms from one pool with no extra cost.

*Certificate on the pool:*
- PhysCtrl base generates 8 samples per candidate.
- One-window certificate score as in Phase 9 (coarse_cfl, m = 4, `spread_event` start), τ = 0.02614.
- Rejected = score ≥ τ or blown.

*Arms (equal simulator budget):*
- **ACT:** all rejected candidates.
- **RND:** a uniformly random subset of the whole pool of the same size, seed 14.
- Probe cost is recorded and reported, not charged; the confirmatory test charges it.

*Training (new `p14_train.py`, PhysCtrl's loss as in 4g):*
- Each batch is 50% arm data and 50% replay of the original distribution: the 3,000 `fc_train` in-distribution re-simulations plus the 100 stored `sample/` trajectories.
- 5,000 steps, batch 8, bf16; the final checkpoint is used.
- Two variants per arm (4 runs):
  - **F:** full fine-tune, lr 1e-5, warm-up 100.
  - **L:** LoRA rank 16 on all attention projections (spatial and temporal), lr 1e-4.

*Evaluation (development sets on the test objects; all seen before, so exploratory):*
- shift-dev = Phase 4m `ood2_E_low`, `ood2_E_high`, `ood2_force` (drop excluded);
- ID-dev = `id_new_0`, `id_new_1` (200 queries).
- Each model generates sample 0 with the same seed as PhysCtrl. Error vs the stored reference; risk = P(error > 0.10), object-weighted.

*Decision rules (fixed now):*
- **KILL A** if no variant of ACT reaches shift-dev risk ≤ 0.7 × PhysCtrl's while keeping the ID-dev mean error within +0.005 of PhysCtrl's.
- **Variant for the full run:** among ACT variants passing the ID rule, the lower shift-dev risk (tie → L).
- ACT vs RND is reported descriptively only; the confirmatory comparison is on fresh data in 14b.

*Cost (estimate):*
- about 4–6 A100-h: 3,000 reference rollouts, more for stiff E;
- about 1 H200-h: generation and probes;
- about 6 H200-h: 4 × 5k steps;
- about 1 H200-h: evaluation;
- **about 12 GPU-h** in total.

**14a stage 1 done (Oct 23; jobs 432496–432498; recorded before any training):**
- Pool: 2,950 of 3,000 candidates (50 unstable reference rollouts excluded). GPU 98–99% for the build, 36–47% for generation and scan.
- Acquisition (`results/p14a_acquire/SUMMARY.md`): the one-window certificate rejects 2,619 candidates (89%). By log E: [3, 4) 96%, [4, 7) 89%, [7, 8) 83%.
- Consequence: arms ACT and RND share 2,316 of 2,619 files (88%). The pilot's ACT-vs-RND contrast is therefore weak by construction, as anticipated in its descriptive-only status. The kill rule and variant choice are unaffected, and 14a proceeds unchanged.
- **Planning note for 14b, written before any 14a training or result:** the certificate can only show its value as a selector if the candidate pool contains many queries the generator already handles. Two candidate designs for 14b:
  1. a pool drawn from a mixture of in-range and shifted conditions;
  2. a budget B smaller than the rejected set, with ACT taking the highest-scoring candidates.

  The choice is made when 14b is pre-registered.

**14a result (Oct 24; jobs 107/108 all exit 0, training GPU 96–98%; `results/p14a_summary/SUMMARY.md`): KILL by the pre-registered rule.**

| model | shift-dev risk | ID-dev mean error | ood2_E_high risk |
|---|---|---|---|
| base | 0.613 | 0.0751 | 0.060 |
| ACT_F / ACT_L | 0.853 / 0.823 | 0.148 / 0.132 | 0.62 / 0.55 |
| RND_F / RND_L | 0.857 / 0.800 | 0.144 / 0.127 | 0.61 / 0.51 |

- The first summary run gave NaN everywhere (`object_rate` was given a list); fixed with an array before any reading, and the NaN "KILL" was discarded.
- **Diagnosis (exploratory, ID-dev, first 120 queries):**
  - The fine-tuned models move the right distance (centre-of-mass travel ratio 0.94–1.00) and deform about right (0.97–1.04), but in the wrong direction or timing: centre-of-mass error 0.016 → 0.061–0.089. Their frame-1 error rises 0.008 → 0.026–0.040, and accelerations are 1.5× the reference's.
  - Training is not broken: the loop matches PhysCtrl's; the targets equal (x[1::2] − 5)/2; force, drag point and mask follow the same convention as the replay data (cos(motion, force) = 1.000); validation loss falls.
  - **Likely cause:** 84% of the ACT targets end in a collision with PhysCtrl's walls (Phase 10e), which occur in only 2.9% of the in-distribution re-simulations (corrected Nov 3 from 1.3%, a subset artefact; `results/p14a_diagnose/`). Together with no protection against forgetting other than replay, the model is pulled towards wall dynamics. Full fine-tunes and LoRA on all attention degrade alike, as in 4g v1.
  - Also, `ood2_force` cannot move at all: every model scores 1.000 because 100% of its references hit a wall.
- ACT vs RND is uninformative (88% overlap).

**Pilot 14a-v2 (pre-registered Oct 24, 2026, after the 14a verdict and before any 14a-v2 code ran; development data, exploratory; about 7 GPU-h, decided by Claude under the user's 50 GPU-h rule).** It tests whether the two diagnosed causes, not the idea, killed 14a. The 14a KILL stands as recorded.
- **Pool:** the same 2,950 candidates, restricted to those whose reference never reaches the wall (p10e rule, 0.5). No new simulation.
- **Arms:** ACT = the wall-free candidates the one-window certificate rejects; RND = a same-size uniform random subset of the wall-free pool, seed 14. (`p14a_acquire.py --wall-free --run-name p14a2_acquire`.)
- **Training:** as 14a (5,000 steps, batch 4 arm + 4 replay, same replay set), but forgetting is controlled as in 4g-v2. The replay half is additionally pulled to the frozen base's prediction on the same noisy input (λ_distill = 1; the arm half is not distilled). Two variants, both LoRA, lr 1e-4:
  - **Ld:** rank 16 on spatial + temporal attention;
  - **Ltd:** rank 8 on temporal attention only (the 4g-v2 adapter).
- **Evaluation:** as 14a (sample 0, seed 0, same development sets; base is the same stored generation). **Errors are taken over frames before the reference's first wall frame** (queries with < 4 such frames dropped and counted), so the walls cannot decide the outcome.
- **Rules (unchanged thresholds):**
  - KILL A for good if no ACT variant reaches shift-dev risk ≤ 0.7 × base's while keeping the ID-dev mean error ≤ base + 0.005, both on the truncated errors.
  - Variant for 14b: the lower shift-dev risk among the ID-passing ACT variants (tie → Ltd).
  - ACT vs RND descriptive.
  - If 14a-v2 also kills, Option A stops and the paper keeps B only (the user is told; no further A pilots).
- **Cost (estimate):** 4 × 5k steps ≈ 6 H200-h, plus evaluation ≈ 1 H200-h.
- **Acquisition done (Oct 24, CPU, before any training; `results/p14a2_acquire/SUMMARY.md`):** 2,196 of the 2,950 candidates reach the wall and are removed. The wall-free pool has 754; the certificate rejects 424 (56%) → ACT; RND = 424, overlap 236 (56%, against 88% in 14a). Rejected share by log E: [3, 4) 0.82, [4, 7) 0.56, [7, 8) 0.40. Ready: `bash slurm/submit_phase14a2.sh`.

**14b (full run, confirmatory).** Pre-registered after 14a and before any fresh data. Planned contents:
- Objaverse training objects;
- 2–3 acquisition rounds;
- ACT vs RND at equal budget with probe cost charged;
- fresh test draws on the held-out test objects;
- a further shift outside the trained range, where the recalibrated certificate must still hold;
- in-distribution non-degradation.

### Phase 15: contact-aware two-object generator (Option B). Pilot 15a pre-registered Oct 22, 2026, before any 15a code ran

**Idea.**
- PhysCtrl composes objects independently, so a pushed object passes through another (contact error 0.305 in 4l).
- A two-object model keeps PhysCtrl's architecture and token count: 1,024 FPS points per object, 2,048 tokens. PhysCtrl's sin-cos positional embedding is fixed at 2,048 points, so weights load unchanged.
- One addition: a zero-initialised learned object-ID embedding added to each point's input embedding.
- One force, on object A only: PhysCtrl base conditions on a single force.

**Scene design (fixed now; `p15_scenes.py`):**
- Objects from the `sample/` set (training) or the test set (evaluation, never trained on).
- Shared in-distribution (E, ν) per scene, as in `id_new_`. A gets one in-distribution drag (coefficient ~ U(0.02, 0.2) × base, stored drag point, hemisphere direction). B has no force. No gravity.
- Placement: A is simulated alone. A target frame k* ~ U{5..18} is drawn, and B is placed at rest so that A's frame-k* configuration first comes within D_C = 2·dx of B, translated along A's displacement direction between frames 0 and k*.
- Valid if: the first approach is at a model frame in [3, 21]; the initial distance ≥ D_C; all points stay in [0.5, 9.5].
- Controls (20%): B placed 0.5 beyond A's swept volume, so there is no contact.
- Joint simulation at the reference setting (4,096 particles). Model targets are the FPS subsets (1,024 per object, fixed per object).

**Pilot data:** 1,500 training scenes on `sample/` objects, plus 200 development scenes on test objects (fresh pairing seed).

**Training:**
- Initialised from base, object-ID embedding zero.
- Each batch is 50% two-object scenes and 50% single-object replay (2,048 points of one object, ID embedding all zero).
- 6,000 steps, batch 8, lr 1e-5, full fine-tune, final checkpoint.

**Baselines on the development scenes:**
1. **Composed PhysCtrl:** A generated alone with its force, B alone with zero force, each evaluated on the same 1,024-point subsets.
2. **Base on the joint 2,048-token input without fine-tuning,** to show the ID embedding and training matter.

**Metric:** mean point error vs the joint reference over both objects, and risk. Separately on contact scenes and controls.

**Decision rules (fixed now):**
- **KILL B** if the fine-tuned model's contact-scene mean error is not ≤ 0.7 × the composed baseline's, or if its control-scene error exceeds the composed baseline's by more than 0.01.
- Otherwise 15b (full run: Objaverse objects, fresh test scenes, the certificate and shadow switch as the safety net) is pre-registered before any fresh data.

**Cost (estimate):** about 4–6 A100-h for 1,700 joint rollouts, about 6 H200-h for training, about 1 H200-h for evaluation; **about 12 GPU-h**.

**Clarifications before any 15a data (Oct 24, 2026, written with the code; the user said "yes" to starting 15a):**
- **Components from existing rollouts** (no new single-object simulation):
  - A = a stored in-distribution reference rollout: training uses the 4g re-simulations of the `sample/` objects (`fc_train/id_new_0..14`, 15 draws × 100 objects = 1,500 scenes); development uses the calibration rollouts of the test objects (`calib/id_new_0..1`, 200 scenes).
  - The scene's (E, ν) is A's own in-distribution draw, so A's stored rollout is exactly its pre-contact motion.
  - B is another object of the same set, from its draw-j file.
  - Pairing: a cyclic shift of a seeded permutation, so every object is A once per draw. Controls: a seeded 20%.
- **Placement:** B is centred on A's frame-k* centre and moved along A's displacement direction (frame 0 → k*) by the smallest s ∈ [0, 6] with no approach < D_C before k*, rounded to whole cells; up to 20 tries with a new k*. Controls: a random direction, s in steps of 0.2 until the objects stay ≥ 0.5 apart. The box rule [0.5, 9.5] is the Phase 10e wall rule, so no scene reaches PhysCtrl's walls.
- **Sanity rule (as 4l):** invalid if, before model frame k_c − 2 (controls: all frames), the joint rollout deviates from the superposition (A's stored rollout + B at rest) by more than 0.02.
- **Model files:** 2,048 points = A's 1,024 FPS points then B's 1,024 (FPS from x0, start index 0), in PhysCtrl's HDF5 layout so its TrajDataset reads them unchanged.
  - The force input is A's: `drag_force` = f / c_A and `base_drag_coeff` = c_A = 9.8 · 1000 · ΣvolA / n_mask,A. This is exactly the number PhysCtrl computes for A alone, so composed PhysCtrl and the joint models see the same force.
  - TrajDataset's 4th drag-point entry (the mask count) is taken over the subset; this cannot be set otherwise.
- **Object-ID embedding:** two zero-initialised learned vectors (A, B) added to the input encoder's output of every frame, for two-object rows only. Single-object replay rows get nothing, so with the embedding at zero the model is exactly PhysCtrl.
- **Composed baseline:** A = base on A's own 2,048-point file; B = base on B's file with the drag force set to 0 (B's own drag box kept as the mask, so the input is well defined). Each is evaluated on the same FPS subset, B translated by t_b.
- **Evaluation:** sample 0, seed 0, for composed / base-joint / fine-tuned; error = mean point distance over the 2,048 subset points and 24 frames vs the joint reference. The kill rule is unchanged.
- **Code:** `p15_scenes.py` (CPU design), `p15_build.py` (A100: joint rollouts + model files), `p14_train.py` with `object_id: true`, `p15a_eval.py` (H200), `p15a_summary.py` (CPU). Code test of the design on 6 scenes per set (scratch run, deleted): all valid, k_c 7–18.

**15a scene design done (Oct 24, CPU; `results/p15a_scenes/SUMMARY.md`; no simulation yet):**
- training: 1,500 scenes, 1,439 valid (1,136 contact, 303 controls);
- development: 200 scenes, 187 valid (151 contact, 36 controls);
- all invalid scenes: no valid placement in 20 tries;
- first approach k_c: median model frame 11 (range 3–21).

**Code tests (scratch only, nothing in `results/` except the design):**
- The build dry run (stand-in spring simulator, softened; files written regardless of the sanity check, dry run only) produces joint, model and rest files.
- PhysCtrl's own TrajDataset reads all three. The joint file's force input equals A's component exactly (force (0.0318, −0.0229, 0.1759), base_drag_coeff 26.27); A's subset equals its component's points; mixed two-object/replay batches collate (F, C stored as 9 values, as PhysCtrl does: found and fixed in this test).
- Object-ID hook unit test: at zero init the output is identical, with or without IDs; a non-zero embedding changes only two-object rows; replay rows are unchanged.
- Trainer dry run passes (object_id on; embedding trainable). The verdict script runs on fabricated generations.
- The evaluation script is import-checked only (it needs a GPU), as `p14_eval_generate` was.

**Ready:** `bash slurm/submit_phase15a.sh` (build ≈ 3 A100-h, training ≈ 2–4 H200-h, evaluation ≈ 1 H200-h).

**Budget.** The user's "A+B" approves the option budgets (~35–55 + ~45–60 GPU-h), spent in stages. Each full run starts only if its pilot passes, and its GPU-hours are recorded here first.

### Paper qualitative figures (Oct 19, 2026; rules fixed before any paper picture)

- **Point clouds** (`code/fig_paper_select.py` → `results/fig_qualitative/paper_selection.json`): the deck's median rule on the Phase 9 data.
  - Shift: lower-median PhysCtrl error per family (ood3_E_low, ood3_force, ood3_drop).
  - Contact: lower median over all contact scenes, and over rejected scenes with a mid-clip switch.
  - Selected: shift_soft (repair k = 3), shift_force (exact fallback), shift_drop (exact fallback), contact_median (repair k_sw = 8), contact_switch (repair k_sw = 14).
  - The always-M1 column replaces coarse only, because M1 is the strongest cheap baseline in Phase 9.
  - Job 94 recomputes ours (M2/M3 suffix) and M1, checked against the stored errors (≤ 2e-3).
- **Video:** the Phase 6a rule (median generation error among rejected cases), applied to ours-v2's decisions after the job 95 re-render.
  - Frames: model frames 0, 5, 10, 15, evenly over the first two thirds. The Phase 6a note says objects leave the view in the last third; the deck strip had used frame 18, which falls in that third.
  - The stand-in layout check on the Phase 6a renders reproduced the Phase 6a pick (giraffeX ×3).
- Layout was checked on the deck's stand-in data only (scratch dir; nothing written to `paper/` or `results/`).

### Ranked plan to the deadline (Oct 14, 2026; senior-review pass)

Based on `notes/audit_2026-10-14.md` and `notes/literature_2026-10-14.md`. Ranking = expected impact on acceptance ÷ (GPU-hours + days). GPU-hours are estimates.

| Rank | Item | Why | GPU-h | Dates |
|---|---|---|---|---|
| 1 | **Phase 4m:** fresh shifted queries + XF + seed-spread deferral + uniform placement | settles objections 1–3 (coarse simulator, exploratory shift results, compute-value audit) on untouched data; decides the positioning | 9–11 (H200) | submit Oct 14–15; verdict Oct 16–17 |
| 2 | **Videos:** PhysCtrl's trajectory-conditioned renderer (CogVideoX-based, H200): generation vs certified repair vs ground truth on the median examples and the contact scenes; then FVD/FVMD/Traj-ADE | a video venue; PhysStream's protocol is the template | 15–25 | renderer setup Oct 15–18 (the user runs the download on the setup host); renders Oct 28–Nov 4 |
| 3 | **Item 3: multi-object contact benchmark** (decisions below) | the only setting where errors are local in time and space, so event-aware placement and repair timing can matter; answers "single object" | ≈ 21 | code Oct 17–19; pilot Oct 20; run Oct 21–23; verdict Oct 24 |
| 4 | **CPU baselines from the literature:** momentum-residual gate (ANCHOR-style), best-of-N with the defect at matched compute, risk–coverage / AURC, the union bound over the chain | reviewer-demanded baselines; zero GPU | 0 | Oct 18–25, in parallel with 3 |
| 5 | **Amortisation:** adaptive/online conformal + a learned probe policy on the query stream (from existing scans) | the promised half of contribution 3; if it fails, the user decides whether the claim stays | 0–2 | Oct 24–31 |
| 6 | **Writing** | abstract by the Nov 10 registration; full paper by Nov 16; supplement (videos, all negative results) by Nov 23 | — | Oct 31–Nov 16 |

**Deprioritised:**
- One more generative-repair attempt (RePaint/PPR re-noise loops): low expected value given 4b–4h, CondMDI and NeuroForge.
- External rigid benchmarks (RigidBench, Physics-IQ runs): rigid is outside the elastic scope; cite only.
- The large model: its generation costs about a reference rollout (4.52 vs 4.67 s), so it is outside the regime where probing pays. Reported as the boundary condition.

**Item 3 decisions (Oct 14; the user asked Claude to decide):**
1. **Un-cut multi-object scenes: yes.** Contact is the only regime where event-aware placement can show value; the literature's strongest metric precedent (RigidBench) is simulator-grounded 3D error, which we match.
2. **Sticky contact: accept and state it.** One MPM grid gives no-slip contact that starts about 1–1.5 dx early. This is the same simulator for ground truth and probes (rule 3 of the scope), and changing the guarded bridge would cost 2–3 days.
3. **Per-object composition pipeline: acceptable**, with the `gt_super` control (superposed true single-object motions). It is how single-object generators are used for multi-object scenes, and the control isolates the ignored interaction.
4. **Extras:**
   - large model: **no** (out of the cost regime, see above);
   - reference-setting scan: **pilot only** (20 scenes, to evaluate the H0 lift-floor gate before the main run; saves ≈ 9 GPU-h);
   - rendered repairs: **yes**, folded into item 2.

### Ranked improvement schedule (dates are targets)

| # | Item | Contribution | Code | GPU | Dates |
|---|---|---|---|---|---|
| 1 | Phase 2c scan | 1, 2, 3 | done | 4 A100, about 3 h | submit Oct 11 → verdict Oct 12 |
| 2 | Calibration data + conformal certificate (4a) | 1 | data done; analysis Oct 12–13 | 4 H200, about 6 h (parallel) | Oct 11–13 |
| 3 | Physics-prior events (in 2c) | 2 | done | — | Oct 12 |
| 4 | Training-free constrained repair (4b) | 2 | Oct 12–15 | 4 H200, about 10 h | Oct 15–16 |
| 5 | Multi-object contact benchmark | 2, 1, 3 | Oct 13–16 | A100 sims + H200 generation, about 15 h | Oct 16–17 |
| 6 | Window packing latency | 3 | Oct 14 | 1 A100, about 1 h | Oct 15 |
| 7 | Compute-accounted Pareto (3b, mostly offline) | 3 | Oct 17–20 | about 5 A100-h | Oct 20 |
| 8 | High-resolution fidelity sweep | 3 | Oct 14 | 4 A100, about 4 h | Oct 15 |
| 9 | Amortisation (5a), minimal: one replay round | 3 | Oct 18–28 | 4 H200, 30–60 h | Oct 25–Nov 2 |
| — | Writing | — | — | rendering on H200 | Nov 3–16 |

## Phase 3: query stream, data and baselines (after both GOs)

- **3a, data** (`20_p3_data_gen_TODO_A100.sh`).
  - Regenerate PhysCtrl-style trajectories with their scripts. Tens of thousands of elastic trajectories, plus rigid via Genesis, plus multi-object contact by extending their scripts.
  - Build the proposal's splits: ID, parameter OOD, geometry OOD (held-out Objaverse categories), force OOD, multi-object.
  - Each query stores: full ground truth, 4 perturbed simulations, contact labels, and a calibration/test flag.
  - Needs Objaverse downloads via their `dataset_toolkits`.
  - Rigid (Genesis) needs a **separate conda env**. `genesis-world==0.2.1` pins pydantic 2.7.1, numpy 1.26.4 and moviepy ≥ 2, which conflict with the main `somw` env (see `00_setup_env.sh`).
  - Embarrassingly parallel: 4 × A100. Idle H200s can take shards, since the simulator is hardware-agnostic.
- **3b, baselines at matched total compute** (`21_p3_baselines_TODO_A100.sh`):
  - always generate; always simulate;
  - uniform and random windows;
  - ANCHOR-style trigger; Hybrid-NWM-style whole-trajectory deferral;
  - best-of-N; oracle placement.
  - Headline figure: error versus total compute (generator + simulator + verifier steps), with area under the curve.
- **Falsifier.** Probing does not beat uniform windows at matched budget (the September 2026 compute audit predicts this). If so, the paper leads with the certificate (contribution 1).

## Phase 4: certificate and repair

- **4a, certificate** (`30_p4_calibrate_TODO_A100.sh`).
  - Split-conformal threshold on the calibration slice, using the best Phase 2 signal.
  - Claim coverage **in-distribution only**, where exchangeability holds. Report OOD coverage as an empirical result, not a guarantee.
  - Recalibrate after every model update.
- **4b, repair** (`31_p4_repair_TODO_H200.sh`, 4 × H200).
  - Constrained re-sampling with verified windows fixed.
  - PhysCtrl's DiT was not trained for frame inpainting. Expect a LoRA fine-tune with random frame-conditioning masks on Phase 3 data (H200s; memory and throughput).
  - Compare against simply replacing the probed frames, and show that a fix at a collision changes earlier frames.
- **Falsifier.** Repair is no better than replacement. Contributions 1 and 3 still stand without it.

## Phase 5: amortisation and breadth

- **5a, online loop** (`40_p5_online_TODO_H200.sh`).
  - Replay verified windows into LoRA updates. Plot probes per query over the stream at stable error.
  - Compare with learning from full rollouts at equal simulator budget.
- **5b, RigidBench** (`41_p5_rigidbench_TODO_A100.sh`).
  - Real rigid contact events. Kabsch lift for the rigid state.
  - Simulator mismatch (Genesis versus Blender Bullet) must be zero or reported.
- **5c, real-world checks** (`42_p5_realworld_TODO_H200.sh`).
  - gs_physics collisions and the Physics-IQ solid subset.
  - Needs the rendering stack (`DOWNLOAD_RENDER=1`, CogVideoX-5B based renderer, which needs H200 memory).

## Phase 6: figures and paper

Teaser (probes per query falling while error stays flat); Pareto curve; lift-floor table; placement and assimilation ablations; coverage plot; qualitative videos rendered with PhysCtrl's trajectory-conditioned renderer on H200s.

---

## GPU assignment

| Work | GPU | Why |
|---|---|---|
| Warp MPM / Genesis rollouts, OOD and data generation, lift floor, probes, fidelity sweep | **4 × A100** (Slurm arrays over queries) | Simulations are small (2k–20k particles), latency-bound and independent per query, so they split perfectly across arrays. They don't need H200 memory, and keeping them off the H200s leaves those free for model work. |
| Generator sampling (cheap), and later LoRA / inpainting fine-tunes and the online loop | **H200** | Training and the 5B video renderer need 80 GB+ memory and bandwidth. Sampling sits next to fine-tuning so checkpoints don't move. |
| Phase 1 timing | **both** (1 each) | Cost ratios are hardware-specific; the GO rule requires both. |
| Phase 0 checks | **both** (1 each) | Proves the environment and Warp kernels on each node type. |
| Video rendering, VLM judge (Phase 5–6) | **H200** | Memory. |

**Parallel schedule.**
- Phase 0: 01 (A100) and 02 (H200) run together.
- Phase 1: 03 (1 A100), 04 (1 H200) and 05 (4-task A100 array) are all submitted after 01. With 4 A100s, two of 05's tasks wait behind 03. Phase 1 needs at most 4 A100s + 1 H200 at once.
- Phase 2: 07 (4 × A100) → 08 (4 × A100) and 09 (2 × H200) in parallel → 10 (4 × A100).
- The remaining 2 H200s are free during Phase 2. After a Phase 1 GO, they can speculatively start Phase 3a data generation, accepting that this is wasted if Phase 2 fails.

---

## Deadline schedule (added Oct 10, 2026)

There are about 5 weeks to the paper deadline.

**The constraint is engineering time, not GPUs.**
- 8 GPUs × 35 days ≈ 6,700 GPU-h available, against the ~500–1,000 GPU-h estimated for all phases.
- Phase 3–5 code does not exist yet.
- So: write Phase 3–4 code *in parallel* with the Phase 0–2 runs, accepting that some of it may be wasted if a gate fails.

All durations below are estimates.

| Dates | Cluster work (GPU) | Code work (no GPU) | Gate / deliverable |
|---|---|---|---|
| **Oct 10–11** | Run `00_setup_env.sh` on `crimv3mgpu015`. Submit `submit_phase0_1.sh` (1 A100 + 1 H200, then 4 A100 + 1 H200) | Fill `cluster.env` and `configure.sh` | Phase 0 pass |
| **Oct 12–13** | Phase 1 jobs (≤ 8 h wall) | Start `code/p3_stream.py` (query stream; MPM part only, reuses `p2_build_ood.py` patterns) | **Phase 1 GO/NO-GO** |
| **Oct 13–17** | Phase 2: 07 (4 A100) → 08 (4 A100) ∥ 09 (2 H200) → 10 (4 A100). The 2 idle H200s start speculative Phase 3a data shards | Phase 3b baselines harness (always-gen, always-sim, uniform/random/oracle windows, best-of-N) on the existing `verifier.py` | **Phase 2 GO/NO-GO** (target Oct 17) |
| **Oct 18–25** | 3a data generation (4 A100 + 2 H200) → 3b Pareto runs (4 A100). 4b inpainting LoRA fine-tune (2 H200) | 4a split-conformal calibration (cheap, CPU/A100). 4b constrained re-sampling | Pareto curve; coverage plot |
| **Oct 26–Nov 3** | 4b repair evaluation; 5a online loop (H200); fill remaining baselines | Figure scripts; start paper (method, setup) | Repair vs replace; probes/query curve |
| **Nov 4–9** | Gap-filling reruns only; qualitative renders (H200, CogVideoX renderer) | Write the paper; abstract from measured numbers | **Nov 10 registration** (title + abstract) |
| **Nov 10–16** | No new experiments. Only reruns needed for tables | Final paper | **Nov 16 paper** |
| **Nov 17–23** | Supplementary videos (H200) | Supplementary | **Nov 23 supplementary** |

**Cut list for this timeline** (extends the proposal's cut order). Cut in this order:
1. Multi-object scenes.
2. RigidBench (5b): it needs a Genesis-vs-Bullet mismatch study.
3. gs_physics and Physics-IQ (5c).
4. User study (keep the VLM judge).
5. The online loop (5a), shrunk to a single replay round.

**Never cut:**
- the matched-compute Pareto curve;
- the uniform-probing and oracle baselines;
- the lift-floor table;
- the calibration/coverage result.

**If Phase 1 or Phase 2 is NO-GO**, follow the fallbacks in those sections, and decide the paper's framing with the user by **Oct 18**. That leaves roughly 3 weeks for whichever version survives.

---

## What to send back after each gate

- **After Phase 0:** `results/p0_check_A100/p0_report.json` (just the `replay_all_pass`, `generator_ok`, `testset_counts_by_mat_type` fields and the `generator` rows), plus the last lines of `physctrl_eval/eval_*.log`.
- **After Phase 1:** `results/p1_summary/SUMMARY.md` (complete), plus any `"error"` rows in `results/p1_cost_*/timings.jsonl` (`grep error`).
- **After Phase 2:** `results/p2_summary/SUMMARY.md`, plus `results/p2_build_ood/shard_*/failures.json` if non-empty.

### Project freeze (Nov 3, 2026; the user asked to freeze the project and archive everything)

- Archive kit in `../freeze/` (README, COMMANDS, scripts). GitHub = code, docs, verdicts, manifests; Hugging Face = `somw-results` (public), `somw-data` and `somw-checkpoints` (private by default: PhysCtrl declares no licence), `somw-private` (study codes, redacted session record). Third-party files are pinned by revision and sha256, not re-uploaded.
- Inline analyses of Oct 24 frozen as scripts with stored output: `code/p14a_diagnose.py` (all numbers reproduced, except the replay wall share: 2.9% over all 3,000 files, not the 1.3% of an unsorted 600-file subset; corrected above), `code/p12_s0_diagnose.py` (correlation 0.85 reproduced).
- `freeze/verify_numbers.py` re-runs every dev-server verdict from the stored data and compares each `SUMMARY.md` byte for byte; report `freeze/VERIFY.md`. Code test on 4 verdicts (P1, 4a, 9, 14a): all IDENTICAL.
- Not run before the freeze (scripts ready, nothing submitted): Phase 12 S0 rerun and test runs, Phase 14a-v2, Phase 15a, PokeFlex download (Phase 13). Human study: 1 answer code.
- **Verification result (run Nov 3, traced Nov 7; `freeze/VERIFY.md`, `freeze/VERIFY_NOTES.md`):** 46 of 50 verdicts byte-identical, both paper tables identical. The other four change no reported number:
  - `p12_s0_diagnose`: harness artefact (data path relative to the code; `p12_s0.py` now uses `DATA_ROOT`); identical after the fix.
  - `p14a_acquire`: one blank line (from the `--wall-free` option).
  - `p4g_summary`: one added line (the summary script later gained the non-degradation check; +0.0329 FAIL for v1, in line with the recorded forgetting).
  - `p10a_convergence`: two 4th-decimal table cells (0.1995 → 0.1994, 0.0585 → 0.0584). **Job 96 ran twice** (430617, then 430786 about 4 h later, not logged at the time); the second overwrote the rows and the stored verdict predates it. `summary.json` is identical; neither cell is cited anywhere.

