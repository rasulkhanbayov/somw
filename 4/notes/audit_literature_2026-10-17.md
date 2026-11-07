# Audit + literature re-check (Oct 17, 2026)

Second senior-review pass, at the user's request ("novel framework, strong results, beat baselines").
Literature verified by a research agent against the arXiv export API (title, authors, v1 date, comment
field); raw dumps in the session scratchpad (`lit/`). **Not yet in `paper/main.bib`**: add an entry only
after re-checking it the same way.

## 1. Weaknesses, root cause, fixable before Nov 16?

| # | Weakness | Root cause | Fixable? |
|---|---|---|---|
| 1 | **The cost premise is fragile.** Our "always simulate" is PhysCtrl's own setting (dt 1e-4, grid 125, 4.67 s). Phase 1's sweep shows (dt 4e-4, grid 96) at median error 0.007 for 1.17 s (0.25×) in distribution. A simulation-savvy reviewer will ask for "always simulate at the cheapest adequate setting". Under shift and contact this is **unmeasured**. | evaluation (baseline choice) + method (repair at reference fidelity) | yes: pilot P1 |
| 2 | Savings are modest: 0.44–0.53× in distribution, ≈ 1× under shift and contact | method: a rejection costs a near-full reference rollout | partly: P1 (cheaper repair tier) |
| 3 | Conservative certificate: rejects 51% in distribution while 14% are bad; 41% of good contact controls rejected | method: score = max of 3 coarse window defects, threshold set on bad draws only | yes: P2 |
| 4 | Strict ε = 0.05 missed in distribution (7c S1/S3) | method: the coarse-shadow state's own error floor | P1 (medium-fidelity shadow), P4 |
| 5 | No video-level metric; every competitor comparing against PhysCtrl reports VideoPhy-2 / GPT-judge / VBench(-2.0) / FVD | evaluation scope | partly: P3 (limited by the 5 image examples) |
| 6 | One generator (base), elastic only. The large model's generation (4.5 s) costs about a reference rollout, so there is no efficiency story there | scope | not before the deadline (a second trajectory generator, e.g. PhysLDM 2610.07609, is too new) |
| 7 | A simulator-free momentum-residual gate is as safe under shift (7a Q3) | fact | fold it into the certificate (P2) |
| 8 | Amortisation is tiny (−1.3%) and needs a stationary stream | method | minor; a shift-aware calibration (Conformal Policy Control 2603.02196) is future work |
| 9 | The repair is "just re-simulate"; generative repair failed 5 ways | method / perception | positioning: certified-prefix compute recycling (cf. GEARS 2608.29322) |
| 10 | The Morpheus critique (2504.02918): trajectory matching penalises valid outcomes | evaluation | pre-empt in text: our ground truth is deterministic given the physical parameters |

## 2. Competitors and what reviewers will expect

- **No direct competitor** combines a generated whole trajectory, exact-simulator window checks, a conformal guarantee and prefix-certified repair. Nearest: PhysMAS 2609.07174 (MPM validity gate + repair, no accuracy guarantee); NEWTON 2605.18396 (VLM verifier loop); Hybrid NWM, ANCHOR, NeuroForge, TACR (known); Conf-Gen 2605.28920, CP4Gen 2601.22298, Selective CRC 2512.12844, Conformal Policy Control 2603.02196 (conformal, no simulator).
- **Video-level competitors that compare against PhysCtrl:**
  - PhysFlow 2609.08215: VideoPhy-2, VBench, WorldScore, FVD; PhysCtrl FVD 83.25.
  - NEXUS 2606.15015: contact-rich, built on PhysCtrl's pipeline; trajectory metrics RE@H, penetration, strain. **Closest to our contribution 2.**
  - 3DPhysVideo 2605.16795: always-simulate with Genesis MPM; GPT-judge vs PhysCtrl.
- **Seed spread is overconfident:** independent support from 2606.12997. DiffusionRollout 2602.13616 *uses* spread as its gate (contrast it).
- **Residual is a poor accuracy proxy:** Error-Conditioned Neural Solvers 2606.27354 (matches 4c and 7a).
- **Metrics reviewers expect:**
  - VideoPhy-2 SA/PC with the released AutoEval (HF `videophysics/videophy_2_auto`);
  - a GPT-judge Likert in PhysCtrl's protocol;
  - VBench-2.0 Mechanics;
  - TRAJAN 2505.00209 (point-track motion metric, can use a reference);
  - FVMD;
  - PhysCtrl's own vIoU / Chamfer / Corr;
  - penetration and strain on contact (NEXUS).
- **Test-time-scaling baselines:** best-of-N with a learned verifier (Video-T1 2503.18942, NEWTON); our oracle best-of-8 (7a) already bounds every selection rule.

## 3. Ideas worth borrowing

- **Fidelity ladder** (cf. multi-fidelity scaling 2610.17221): choose the repair/shadow fidelity per query.
- **Learned correction of the coarse simulator:** PhysCoRe 2607.20653, RECAST 2608.11572, coarse→fine MPM decoder 2605.05053.
- **Shift-aware conformal:** Conformal Policy Control 2603.02196, Selective CRC 2512.12844.
- **Cheap executability gates before probing:** PhysMAS.

## 4. Pilots proposed (exploratory; development data only; confirmation on fresh data afterwards)

See PLAN.md once approved. Summary:
- **P1** fidelity ladder;
- **P2** learned/fused certificate score (CPU);
- **P3** video-level metrics on the existing renders;
- **P4** learned coarse-shadow correction (lower priority).
