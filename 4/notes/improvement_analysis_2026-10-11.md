# Improvement analysis after Phase 2-v2 (Oct 11, 2026)

**Goal:** keep all three contributions and make the results able to support them:

1. A residual-free certified verifier.
2. Active, event-aware probing with repair.
3. Compute-accounted efficiency and amortization.

**Sources:**

- `results/p2_summary_v2/SUMMARY.md`
- `results/p1_summary/SUMMARY.md`
- `results/diag_p2/DIAG.md`, produced by `code/diag_p2.py`. This is exploratory and decides nothing.

All numbers below are measured. Anything not yet measured is marked as an expectation.

## 1. Diagnosis

### 1a. "Placement does not matter" comes from the benchmark, not the method

**Most splits have no localised error.**

- Outside the gravity drop, error growth is spread over the whole trajectory. The worst 1-step window holds 8–18% of the growth; a perfectly even spread would give 4.3%.
- Error reaches half its final value late, around frames 12–17 of 24.
- The error is mostly a non-rigid shape error: the non-shift share is 0.86–1.0.
- In drag-only elastic scenes, the generator's mistake is a wrong overall response (stiffness and amplitude), and every window sees it. No placement can beat uniform there, by construction.

**The gravity drop, the only split with contact, does localise error.**

- The worst three steps hold 50–63% of the growth.
- For base, the worst step is within one frame of the true floor contact in 77% of trajectories.

**The event detector failed exactly where events matter.**

- It reads contact from the *generated* trajectory.
- Base ignores gravity, so its trajectory never falls and never shows a contact. The event windows covered the true contact in 0% of base trajectories (80% for large).
- A detector that trusts the generator cannot find the generator's missed events.

**The oracle was mis-specified.**

- It ranks windows by growth in *position* error, which accumulates over the trajectory. A probe instead measures *local dynamic inconsistency*.
- So it is not a ceiling for what a probe can find. On base gravity drop it scored −0.36.
- The right ceiling is the exhaustive scan: run every window and take the max.

**The metric hides placement.**

- Ranking whole-trajectory error by Spearman rewards any window that sees a global error.
- Placement can only show its value through *capture* (finding the worst window with a small budget) and through *repair*, which needs the right window as a constraint.
- Neither was measured.

### 1b. The large model: windows lose to seed spread, and why

**Large's error is more affine than base's.**

- Its non-affine share is 0.24–0.32, against 0.38–0.62 for base. It deforms the object with the wrong overall shape and scale, smoothly over time.
- Large's own sample spread already tracks this (Spearman 0.82).

**A 2-frame window measures acceleration consistency over 2 frames.**

- A trajectory that is smoothly and consistently wrong builds up a large position error while each short window shows only a small defect.
- **Expectation, not yet measured:** longer windows (m = 2, 4) grow the defect of drift-type errors faster than the lift floor grows. Ground-truth data already show the floor rising only from 0.0014 to 0.0027 between m = 1 and m = 2.

**The signals are complementary.**

- Seed spread sees the model's own uncertainty. The window defect sees physics inconsistency, including confidently wrong outputs, such as base on gravity drop, where seed spread is low and the error high.
- Fusing them is untested.

### 1c. The pooled "beats seed spread by +0.75" is a cross-split effect, and it is the certificate's real strength

**Within each split, base seed spread is competitive:**

| Split | Seed spread | Windows |
|---|---|---|
| `ood_E_low` | 0.88 | 0.83 |
| `ood_force_x3` | 0.83 | 0.59–0.74 |
| in-distribution | 0.82 | 0.67 |

**The pooled advantage comes from scale.** The window defect is on an absolute physical scale that is comparable across conditions. Seed spread is not: base's samples agree with each other while all being wrong on gravity drop.

**This is exactly what a certificate needs:** one threshold that works under shift. The right test is single-threshold detection (AUROC for err > ε, pooled over conditions), then conformal coverage. That evaluation supports contribution 1 directly; Spearman does not.

### 1d. Chaos normalisation adds nothing here, as expected

- The perturbed-rollout spread σ is about 0.0055 in every split, set by the injected position perturbation, so norm = raw / constant.
- Elastic scenes driven by drag have no chaos to normalise. Normalisation matters at contacts, which are chaotic.
- It should be tested where it is supposed to work (contacts and multi-object scenes), and computed only for windows that contain a predicted contact, which avoids paying 3× everywhere.

### 1e. Cost: the main regime makes simulation look cheap, for structural reasons

**Simulation cost tracks substep count, not particle count.** Phase 1 (A100, graph mode):

| Regime | Particles | Grid | Rollout (s) |
|---|---|---|---|
| `physctrl` | 2,048 | 125³ | 4.3 |
| `dense10x` | 20,480 | 125³ | 5.2 |
| `dense10x_fine` | 20,480 | 250³ | 22.8 |

**The GPU is mostly idle at 2,048 particles.**

- Windows from many queries (or the k windows of one query) can be packed into one scene for almost free.
- A full rollout cannot be parallelised in time. Windows can, which is the proposal's wall-clock ≈ Δ/F argument; we currently run windows one after another.

**Two baselines distort the comparison:**

- **The large generator is unusually slow:** 25 DDIM steps, 4.5 s per sample on A100. It is slower than a matched-accuracy simulation (0.55 s).
- **Probes run at the reference fidelity:** 3 windows cost 0.125 of a reference rollout, about the same as one full rollout at the matched-accuracy setting (0.127). Probes have never been tried at that setting.

**Regimes where simulation really dominates:**

- fine grids (×97–110 base generation);
- multi-object contact, which needs finer substeps at impacts (untested);
- long horizons.

These are also the regimes PSIVG-style pipelines need for visual quality.

## 2. Proposed improvements (ranked)

Rank is by expected effect on the contributions, cost, and risk, with the deadline (Nov 10 / 16) in mind.

| # | Improvement | Contribution | Grounded in | Cost | Risk |
|---|---|---|---|---|---|
| 1 | **Exhaustive window scan (Phase 2c)**: every window, m = 1/2/4, reference + coarse fidelity, base/large/ground truth. Evaluates offline: coarse probes, longer windows, fusion with seed spread, physics-prior events vs generation events, exhaustive ceiling, single-threshold AUROC | 1, 2, 3 | 1a–1e | **ready**; about 10 A100-h (estimate) | low |
| 2 | **Held-out calibration set + conformal certificate (Phase 4a)**: 400 fresh in-distribution queries, split by object. The signal is chosen on Phase 2c and confirmed here, then a split-conformal threshold is set, with coverage in-distribution and empirical coverage OOD | 1 ("certified") | 1c | **data jobs ready** (H200, parallel to 1); analysis code after 2c | low |
| 3 | **Physics-prior event prediction** (`verifier.prior_contact_frame`, `event_union`), plus capture against the exhaustive ceiling as the placement metric | 2 | 1a | **implemented**; evaluated by 1 | medium |
| 4 | **Training-free repair**: constrained DDIM resampling with verified windows as hard constraints (RePaint-style replacement of known frames in the x0 estimate at each step), compared with simply replacing probed frames | 2 ("with repair") | proposal; PhysCtrl's `prediction_type='sample'` DDIM makes replacement direct | 2–3 days code, about 10 H200-h | medium |
| 5 | **Multi-object contact benchmark**: compose independently generated objects (the standard per-object pipeline), simulate the true collision. Errors are localised at impacts, which is where event placement, normalisation and repair should matter, and where simulation is costly | 2, 1, 3 | 1a, 1d, 1e; PSIVG, Hybrid NWM | 1–1.5 days code; about 15 GPU-h | medium-high |
| 6 | **Window packing**: run k windows in one MPM scene, spatially separated (legal because the drag impulse spans the whole rollout and gravity is uniform). Report wall-clock and GPU-seconds for probes vs full rollouts, also packed. This measures the latency claim | 3 | 1e; Phase 1 particle-count invariance | 0.5–1 day; about 1 A100-h | low-medium |
| 7 | **Compute-accounted Pareto (Phase 3b)** built mostly offline from scan rows: uniform/random/event/exhaustive at k = 1…n windows, coarse/reference, plus always-simulate at each fidelity from the Phase 1 sweep and best-of-N generation | 3 | compute-value audit (2609.13257) | 2 days analysis | low |
| 8 | **High-resolution fidelity sweep** (250³ grid, 20k particles) to find the matched-accuracy cost where simulation dominates | 3 | 1e | 0.5 day; about 4 A100-h | medium |
| 9 | **Amortisation (Phase 5a)**: replay verified windows into a LoRA update; probes per query against stream position | 3 ("amortisation") | proposal; AdaLED, Neural-Parareal | 4–6 days; 30–60 H200-h | high |
| 10 | Lifting F by projection (positive determinant, clamped singular values) to keep the stress state | 1, 4 | Phase 2 v1 blow-ups | 0.5 day | low; only if repair needs it |

**If contribution 3 still does not hold** after items 1, 6, 7 and 8, the strongest version to keep is three-part:

- latency (windows are parallel, a rollout is not);
- probe cost at matched accuracy;
- the amortisation curve.

All three are measured in the regime where simulation is expensive. Consult the user before narrowing anything (CLAUDE.md rule).

## 3. Expected outcomes and how we will know

These are pre-registered in `code/p2c_summarize.py` and `PLAN.md` (Phase 2c) before the scan runs.

| Question | Expectation (not measured) | Decided by |
|---|---|---|
| A. Fidelity | Coarse probes keep Spearman within 0.05 of reference, with the coarse floor ≤ 0.25 of the defect | PASS-A |
| B. Large | Longer windows and/or fusion beat seed spread | PASS-B (primary m = 2) |
| C. Placement | On gravity drop, `event_union` captures more of the exhaustive defect than uniform | PASS-C; drag-only reported, not judged |
| D. Certificate | One threshold detects err > 0.10 better with the defect than with seed spread (base) | PASS-D; large and fusion reported |

Each outcome is reported whichever way it lands. Results are kept from v1 onward.
