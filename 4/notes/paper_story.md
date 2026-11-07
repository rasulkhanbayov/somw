# Paper story (running; update after every result)

Last updated: Oct 12, 2026. All numbers are measured and pre-registered unless tagged *exploratory*. Sources are in `PLAN.md`.

## Thesis

Generate the whole trajectory once, then spend exact simulation only where needed:
- to **certify** what the generator got right;
- to **repair or replace** what it got wrong;
- and to **learn** from it.

## Contribution 1: residual-free certified verifier

**Status: supported.**

- **Signal:** the max raw defect of 3 restarted windows (`coarse_cfl`, m = 4, `spread_event`), with the lift `lift_noF`. It needs no PDE residual.
- **Ranking:** pooled Spearman with true error 0.92; single-threshold AUROC 0.958 vs 0.616 for seed spread (Phase 2c, base).
- **Certificate:** object-level conformal, false-accept rate 0.084 ≤ α = 0.10 (Phase 4a extension).
- **Under shift:** it rejects 0/50 gravity-drop, 0/32 E_low and 0/36 force failures, where base seed spread accepted 50/50.
- **Limits:**
  - in-distribution power over seed spread is not robust (P2′ FAIL);
  - for the large model, seed spread is already strong;
  - the defect must not be used as an optimisation objective (Phase 4c Goodhart).

## Contribution 2: active, event-aware probing with repair

**Status (Oct 16): repair CONFIRMED (4k gated, in distribution; 4m under shift); event-aware FIDELITY CONFIRMED on contact (4o); event-aware placement NOT supported (4l, 4l-ref).**

- **Placement:** on contacts (gravity drop), event placement captures 0.96 of the exhaustive defect vs 0.61 for uniform (large). On drag-only splits, uniform is best, because the error there is global.
- **Repair:** three training-free attempts failed:
  - x₀ replacement gains about 1%;
  - conditioning search doubles the error (Goodhart);
  - anchored conditioning gives no extrapolation gain.
- **Phase 4g (full fine-tune): FAIL** from forgetting (unconstrained +47% in distribution). Conditioning itself was learned: constraints reduce the fine-tuned model's own error 0.313 → 0.243.
- **Phase 4g-v2: FAIL** (no forgetting, no propagation).
- **Key diagnosis: verification ≠ correction.** Windows restarted from generated states inherit the drift: 20% error reduction at their own frames, vs 94% for the true-start anchor. This bounds all window-constrained repair.
- **Phase 4h (certified-prefix suffix re-simulation): FAIL on risk** (0.070 vs 0.013), PASS on cost (0.71× vs 0.84× always-reference). Two causes: early restarts lose the stress state (F = I floor 0.070 at k = 3), and late restarts begin from positions that drifted while every window stayed locally consistent. **Windows certify consistency, not accuracy.**
- **Phase 4k (800 fresh queries, pre-registered): gated V2 CONFIRMED.** Certificate-gated suffix re-simulation from a coarse-shadow state: cost 0.53× vs 0.65× always-reference for certify-or-simulate, risk 0.025 vs 0.011 (within +0.02); cheaper at all four τ; restart timing matters (shuffled +0.002 error, CI > 0). In distribution only; the OOD view (Phase 2c, exploratory) is in PLAN: risk ≤ 0.04 in every split while coarse-only fails (0.26–0.52).
- **Phase 4j (held-out, 1,200 queries): V2 NOT CONFIRMED.** Risk PASS (0.032 vs 0.023); cost 0.888× cert_ref on average but above 1 in 5.6% of calibration splits (97.5th pct 1.034), so J2 FAILS. Timing matters (shuffled restarts +0.003 error, J3 PASS) but cannot be claimed under the rule. Cause: the 6-window chain doubles the accept-path probe cost when most queries are accepted.
- **Phase 4i: V2 PASS (first working repair).** Certified prefix + suffix re-simulated from a cheap coarse-shadow state: risk 0.017 vs 0.013, cost 0.74× vs 0.84× always-reference (saving +0.51 s/query, 97.5% CI [0.34, 0.66]). V1 (stable F-lift from generated positions) FAIL: the drift, not the stress state, is binding. Needs a held-out confirmation and an ablation of the temporal localisation before it is a paper claim.
- **Phase 4l (multi-object contact, pre-registered): H0–H3 FAIL.** Contact-placed windows capture *less* than uniform/spread; event-timed repair is riskier than chain timing. But the certificate rejects **every** contact failure (FAR 0/188), and generation fails mostly at contact (error 0.022 → 0.305). Diagnosis: the coarse grid (dx 0.156) cannot resolve contact (gap 0.16), so the check is noisy at contact and the shadow state before contact is wrong. Next idea: event-aware *fidelity*, not placement (approved by the user Oct 15, conditional on evidence).
- **Phase 4l-ref (Oct 16): H0 PASS** at the reference setting (the check is reliable at contact when the grid resolves it); placement still not supported. Exploratory EAF repair (switch to the reference setting before the coarse grid couples the objects): risk 0.012 at 1.04× vs cert-or-sim 0.000 at 1.19×. To be confirmed on fresh scenes (4n).
- **Phase 4n (fresh, pre-registered): EAF NOT CONFIRMED** (risk 0.034 vs 0.010 + 0.02, exceeded by 0.004; cheaper and lower mean error than the 4k timing). The confirmed 4k repair stays within the margin on contact (0.026 at 0.99× vs 1.16×). EAF fails when the generation mispredicts the event; candidate: let the coarse shadow time the switch.
- **Phase 4o (fresh, pre-registered): CONFIRMED.** Shadow-timed fidelity switch: risk 0.012 vs 0.000 (within +0.02), saving +0.65 s/query vs certify-or-simulate, error 0.016 below the 4k timing. **This is contribution 2's event-aware claim:** the cheap simulator's own state decides when to hand over to the exact one. Limits: on contact, every certified pipeline costs ≥ 1× always-simulate; most contact scenes start within 3 coarse cells and fall back to a full rollout.

## Contribution 3: compute-accounted efficiency and amortisation

**Status (Oct 16): efficiency supported in distribution; amortisation CONFIRMED but small (5a).**

- **Amortisation (5a, pre-registered):** a learned predictor of the certificate score, calibrated on a random 10% audit stream, skips probes when confident. Probe rate 0.99 → 0.85 over 1,698 queries, risk unchanged, cost −1.3% (probes are cheap; the reference fallback dominates). It never skips the hard shifts. Drift caveat: if the stream shifts, the stale audits let risk rise (0.048 vs 0.008); the guarantee is for stationary streams (online/adaptive conformal is future work). Mechanism: amortising the verifier, since amortising the generator failed (4g).

- **Certify-or-simulate with a reference fallback (Phase 4f, held out):** risk 0.023, cost 0.56× always-reference, beats random fallback.
- **Cheap probes:** the CFL-adaptive coarse probe keeps the signal at 1/8 of the probe cost (Phase 2c A′).
- **Negative findings to report:**
  - a fixed coarse setting violates CFL on stiff materials;
  - a cheap full simulator chosen in distribution is inaccurate OOD (gravity-drop median error 0.103);
  - so the fallback must be the reference setting, or adaptive.

## Key figures and tables (planned)

1. **Teaser:** a certified pipeline, error vs cost.
2. **Certificate ROC/AUROC** under shift: defect vs seed spread, per split.
3. **Placement capture** on contacts.
4. **Repair results:** including the negative ones, and the Goodhart effect.
5. **Cost table:** gen / random / cert / sim; the break-even simulator cost.

## Phase 4m (Oct 15, fresh pre-registered shift queries)

- **The certificate under shift:** FAR 0.012 vs seed spread +0.34 (seed spread accepts 99% of drop failures).
- **Gated V2 under shift:** risk 0.015 at 0.94× always-simulate, while coarse-only reaches risk 0.374.
- **XF (accept coarse when it agrees with the generation):** lower risk but more expensive (0.66× vs 0.44× in distribution). Report it as the low-risk baseline.
- **Uniform windows:** costlier at similar risk.
- **The paper's central contrast is now confirmatory.**

## Positioning (from the Oct 14 literature check)

See `notes/literature_2026-10-14.md`. The closest threats are Hybrid Neural World Models (whole-trajectory deferral on the surrogate's own error map) and ANCHOR (residual-triggered correction). Our differences: a stochastic generative model; exact-simulator probes instead of a self-computed signal or residual; an object-level finite-sample certificate; suffix repair from a simulated state. NeuroForge independently finds that a residual is a trust signal but a poor correction objective, matching our Goodhart result (4c).

## Open risks

- Repair works only through simulation (gated V2, confirmed in 4k); generative repair failed six ways (4b–4h), reported as negative results.
- **Reviewer question to answer:** in distribution, a full coarse_cfl rollout without any generation is already accurate and cheap (Phase 4d data: 0.12× reference). Our case rests on OOD, where it fails (gravity drop, force ×3), and on the certificate. 4j reports coarse-only as a baseline.
- Multi-object contact benchmark not built (decided Oct 14: build it, item 3 after 4m).
- Phase 4m: XF did not dominate (resolved Oct 15).
- Amortisation not shown.
- Everything is on PhysCtrl elastic; rigid has not been done.

## Phase 7a (Oct 17, reviewer metrics and baselines)

- Our window score is the best ranker of generation error: AUROC 0.977 pooled, vs momentum residual 0.886 and seed spread 0.671 (Q1, Q2 PASS).
- **Q3 FAIL:** a simulator-free momentum-residual gate (ANCHOR-style) is as safe as ours (0.018 vs 0.015 under shift; 0.000 in distribution), but costs 0.79× vs 0.44× in distribution. **Claim narrowed:** "cheapest safe pipeline", not "safest".
- Best-of-8 cannot fix shift: even the oracle has risk 0.676 (Q4 PASS). This answers "why not sample more?".
- Weak spots to state: a heavier tail than certify-or-simulate (p99 0.232 vs 0.090 in distribution); at ε = 0.05 ours is worse than coarse only in distribution (needs its own τ).

## Phase 7c (Oct 17, targets recalibrated)

- **Under shift**, ours is the cheapest pipeline meeting the risk target in all six (ε, α) cells with ε ∈ {0.05, 0.10}. This is the robust claim.
- **In distribution at the strict ε = 0.05**, ours misses the target: risk 0.125 vs α 0.10, against certify-or-simulate's 0.085 (S1 FAIL); it is no safer than coarse only (S3 FAIL). It is still cheaper (S2 PASS).
- **Likely cause** (not tested): the coarse-shadow repair has its own error floor near 0.05.
- **At ε = 0.20** the certificate cannot be calibrated for α ≤ 0.10 (too few bad calibration objects).

## Contribution 3 after the efficiency pilots (Oct 19)

- **8a:** a tuned cheap simulator beats us on cost for soft materials (in distribution and on contact).
- **E1/E1b:** in the stiff regime generation is structurally cheaper (ours 0.37–0.54× the best tuned explicit simulator), but the soft-calibrated certificate leaks there: 6/16 bad accepted at E ≥ 1e8.
- **E2:** a one-window certificate cuts the probe cost by 67% at equal AUROC.
- **Framing:** contribution 3 = compute accounting of certified safety, following McGreivy & Hakim.
- **New limitation of the certificate:** stiffness shift beyond 10^8.5.

## Phase 9 (Oct 19, fresh pre-registered confirmation): CONFIRMED

- **The headline that survives every baseline:** under distribution shift, ours-v2 is the cheapest pipeline that stays safe: risk 0.013 at 0.71× always-reference.
  - Certify-or-simulate is safe at 0.87×; the residual gate at 0.97×.
  - Every tuned simulator setting fails: M1 0.102, M2 0.102, M3 0.155. Coarse only fails at 0.367.
  - A user who does not know whether a query is in distribution cannot pick a safe cheap setting in advance; the certificate decides per query.
- **In distribution:** 0.020 at 0.26× (C2 ≤ 0.35× PASS). Always M1 is cheaper (0.13×), reported as the price of safety.
- **Contact:** SF 0.012 at 0.90×; the M3 suffix took it below always-reference. Always M3 is cheaper on contact (0.000 at 0.50×) but unsafe under shift.
- **Paper framing:** no single fixed simulator setting is both cheap and safe on all three benchmarks, and ours is the only pipeline within the risk target on all three at < 1× always-reference. This holds among the pipelines measured in Phase 9; the residual gate's contact risk (0.290) comes from 7d on the 4o scenes, not from the Phase 9 scenes.

- **Two facts the paper states openly (Oct 19, `results/p9_table`):**
  - On contact, tuned M1 is just inside the bound (0.029, 4/132 bad, at 0.13×), and M3 reaches 0.000 at 0.50×. Both fail under shift.
  - On contact, the 4k chain timing matches SF's risk (0.012) at 0.67×, with twice the mean error. Contribution 2's event-aware switch is therefore an accuracy gain on this test set, not a risk or cost gain.

## Phase 10c (Oct 21, exact re-scoring against three reference time steps): the story after the reframe

- **Pre-registered:** T1 PASS (PhysCtrl riskier than ours everywhere, against every reference); T2 FAIL (ours ≤ 0.03 only against dt; under shift 0.108 / 0.216 against dt/2 / dt/4, but always-simulate at dt is 0.113 / 0.226 there); T3 M1, M3 PASS, **M2 FAIL** (as safe as ours under shift within the reference's uncertainty, at 0.29× vs 0.71×).
- **What the paper now claims:**
  1. PhysCtrl's failures are real, not an artefact of the reference (robust risk under shift 0.647).
  2. The certificate catches them (unchanged).
  3. Ours *tracks the simulator it replaces*: within +0.025 of always-simulate's risk on every set against every reference, at 0.26–0.90× its cost; the only pipeline below 1× that does (exploratory max-gap metric).
  4. Tuned settings: cheaper where the regime is known (M1 ID, M3 contact); M1/M3 ≈ 10 points riskier than the simulator under shift against every reference; M2 ties ours under shift within the reference's time-step uncertainty, with no per-query signal.
- **Dropped:** "the only pipeline safe everywhere below 1×" and "the cheapest safe pipeline under shift" (both rested on dt being exact).
- **Reviewer risk:** "why not always run M2?" Answer in the paper: M2's gap to the simulator is +0.10 under shift against dt, it is not robust on drops/strong pushes (robust risk 0.048 vs ours 0.005, exploratory), it is +0.035 on contact against dt, and it gives no per-query decision. Honest: under the finest reference it is as safe and cheaper under shift.

## Phase 11 (Oct 21, video level)
- The 7B VLM judge with PhysCtrl's prompt fails the pre-registered validity check (V0): it scores the reference simulation's videos no higher than PhysCtrl's (+0.07 [−0.07, +0.21]) although PhysCtrl's trajectories are off by 0.51 on average there. So V1 is uninformative, and the paper says a 7B judge cannot measure physics here (limitation + supplement), not that ours wins at video level.
- Remaining video-level evidence: the human 2AFC study (V2), page live, analysis once at close or Nov 7. On 23 of 28 items ours = the reference video, so V2 mostly measures whether humans see PhysCtrl's physics errors; the 5 PhysCtrl 3× items test a mid-clip repair.

## Fidelity vs realism (Oct 22, simulator audit)
- The reference is PhysCtrl's own data simulator (no gravity during a push, constant push, 1 m objects, E 1e4–1e7). Our metric measures fidelity to it, which equals realism only where material and force are realistic.
- Outside that: 6/23 shifted Phase 11 items stretch into strands; 32% of the Phase 9 softer family. Drop and contact are the realism-safe shifts (PhysCtrl wrong 98% in drop).
- Proposed (R1, needs the user): say this explicitly in the paper; claim fidelity, not realism.

## Circularity (Oct 22)
- Ours = the MPM reference on 7% (ID), 52% (shift), 57% (contact) of queries; repairs re-simulate 61–75% of frames with the same code. "Ours beats PhysCtrl" there means "the simulator agrees with itself".
- Safe claims: the certificate detects departure from PhysCtrl's own target; cost; PhysCtrl fails silently against its own simulator. Not yet supported: "physically more accurate than PhysCtrl".
- Phase 12 (pre-registered, CPU): independent FEM (PolyFEM, fixed-corotated) on single-object drag sets; X3 tests whether MPM is closer to FEM than PhysCtrl is on fallback queries.

## Drop caveat (Oct 22)
- PhysCtrl base has no gravity or floor input, so drop queries are inexpressible for it. Present drop as a certificate result (silent failure caught; seed spread accepts 99%), not as a generalisation failure of PhysCtrl.

## Model-paper options (Oct 22; awaiting the user's decision)
- The user wants a model paper. Note: `notes/model_paper_options_2026-10-22.md`.
- New verified threats: **PAVG** (2610.11791, Oct 8, 2026; a PhysCtrl-style successor that beats PhysCtrl in distribution), NEXUS (contact), PhysiFormer, PhysLDM. No paper post-trains a 3D trajectory generator with simulator feedback or reports PhysCtrl's error under material/force shift.
- Recommended: Option A, certificate-driven simulator distillation (the certificate picks which queries to simulate; the generator is fine-tuned on those rollouts with replay; active vs random at equal simulator budget). This is the proposal's original third contribution in its working form.

- The user chose "A+B" (Oct 22). Option B (pilot 15a, contact-aware two-object generator) code is in progress.

## Option A pilot 14a (Oct 24): KILL by the pre-registered rule
- Every fine-tuned model is worse than base: shift-dev risk 0.61 → 0.80–0.86; ID-dev mean error 0.075 → 0.13–0.15; E_high risk 0.06 → 0.51–0.62. Full fine-tune and LoRA alike.
- Diagnosis (exploratory): the targets were dominated by collisions with PhysCtrl's domain walls (84% of the ACT arm), and replay alone did not stop forgetting. The models move the right distance but in the wrong direction or timing.
- Pilot 14a-v2 (pre-registered, ≈ 7 GPU-h): a wall-free pool, the 4g-v2 replay distillation, and scoring before the wall. If it also kills, A stops and the model contribution is B only.

## The domain wall (Oct 24, Phase 10e, exploratory) — affects the shift claim
- PhysCtrl's MPM has sticky walls 0.2 from every side of the domain. 99% of the stronger-push references (`ood3_force`) and 32% of the softer ones hit them by about model frame 15. In distribution: 4%.
- PhysCtrl's failures do not depend on the walls (wall-free shift risk 0.53). The tuned simulators' shift failures are mostly wall impacts: force M1 0.20 → 0.01 before the wall. Without walls, their remaining failures are the drop (floor) family.
- Safe statement: ours is the safest pipeline below 1× on every reading, but under shift its margin over M2 is ≈ 0.04 and comes from contact events (floor, walls). It is not "beyond PhysCtrl's push range" per se. Paper wording waits for the user (it weakens contribution 3).
- Phase 12 (independent FEM): S0 run 1 failed on a load-normalisation bug (fixed, one rerun allowed). The FEM has no walls, so the force family is compared before the wall only.
- Option B pilot 15a (Oct 24): code ready; 1,439 training / 187 development two-object scenes designed (A pushed into B at rest). The kill rule compares against composed PhysCtrl (≤ 0.7× contact error, controls within +0.01).
