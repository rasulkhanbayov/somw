# Contribution 3 (efficiency): diagnosis, literature, pilots (Oct 18, 2026)

Context: the user's prompt to make contribution 3 a defensible win. Literature items verified by a research agent against the arXiv export API / Crossref (raw dumps and helper in the session scratchpad `lit/`). Not yet in `paper/main.bib`.

## 1. Where compute goes (measured; cost relative to one reference rollout, dt 1e-4, grid 125)

| part | soft (E < 1e7) | stiff (E 10^7.5–10^8) | source |
|---|---|---|---|
| generation (PhysCtrl base) | 0.05× | 0.05× (independent of E) | Phase 1 |
| 3 coarse check windows (m = 4) | 0.063× (0.021× each) | about 2× more (CFL) | 2c scan |
| tuned cheap simulator M1 (dt ≤ 8e-4, grid 96, CFL) | **0.12×** | **0.40× (max 0.66×)** | 8a |
| M2 / M3 | 0.25× / 0.50× | 0.40× / 0.55× | 8a |
| repairs of rejected queries (51% rejected in distribution) | ≈ 0.3–0.4× with a reference suffix; ≈ 0.15–0.2× with an M2 suffix | — | 4k, 8a |

**Diagnosis:**
- **Soft materials (PhysCtrl's regime):** generation + windows ≈ 0.11× ≈ M1's whole cost, *before* any rejection. In the break-even formula of 2608.23075, S = 1/(g + d + (1 − d)·r) with r = 0.4 (generation vs M1), g ≈ 0.5 and d ≈ 0.5, so S < 1 for any reject rate. **We cannot beat a tuned simulator on soft materials by construction.** This is structural, not a tuning problem.
- **Under shift:** the gate correctly defers most queries, so the cost approaches always-simulate. The literature reports the same (2608.23075: 97–99.7% deferred, 0.94–0.96×).
- **Stiff materials:** the stable step of explicit MPM is CFL-limited, dt ∝ dx/c with c = √(E/ρ), so every explicit simulator's cost grows ∝ √E. Generation does not. Measured: M1 costs 3.3× more at E 10^7.5–10^8 than below 10^7. PhysCtrl's generations stay decent there: 4m's ood2_E_high (10^7.1–7.8) had generation risk 0.06. With r → 0, S ≈ 1/(g + d). The windows run on a coarser grid than the tuned simulator, so g ≈ 0.3; with d ≈ 0.2–0.4, S ≈ 1.5–2.5×. **This is the regime where generate-then-certify can be structurally cheaper than the best explicit simulator.** Real objects (plastics ~1e9 Pa) lie there.
- **Main threat in the stiff regime:** implicit MPM keeps its step size as stiffness grows: HOT 1911.07913 (TOG 2020), i-PhysGaussian 2602.17117, GeoWarp 2507.09435 (implicit MPM in Warp). Implicit solvers still slow down with stiffness through conditioning (StiffGIPC 2411.06224), but a reviewer will ask. We have no implicit MPM; this must be stated, or GeoWarp tried.

## 2. Literature (verified; selection)

- **McGreivy & Hakim**, Nature Machine Intelligence 2024 (arXiv 2407.07218): 79% of ML-beats-solver claims used weak baselines. Rules: compare at equal accuracy or equal runtime; compare against an efficient method; justify the hardware; account for training cost; plot cost vs accuracy; pre-registration endorsed. **Our efficiency claims must follow this.**
- **Huang & Stuart 2609.24021:** neural operators win only at low-to-moderate accuracy, and when they avoid temporal iterations or predict a reduced quantity. Classical solvers remain important for verification.
- **Break-even analysis 2608.23075:** d_break = (1 − g − r)/(1 − r). A correct OOD gate cannot accelerate a loop that leaves the distribution.
- **Learned large steps, honestly reported:** Voytan & Li 2508.11119 (a neural operator is slower than a tuned GPU solver on the same grid, much faster for reduced outputs); generative wave propagator 2607.04440 (a 10× larger step gives only 2.2× end to end); FlashNeRD 2610.09130 ("NeRD offers little speedup over the simulator it learned from").
- **Parallel verification:** speculative decoding 2211.17192, RandNet-Parareal 2411.06225 (NeurIPS 2024). Independent windows can run in parallel, so latency ≈ generation + one window.
- **MPM time-step theory:** Sun, Shinar, Schroeder CGF 2020 (10.1111/cgf.14101); Bai & Schroeder CGF 2022 (10.1111/cgf.14620); regional time stepping, Fang et al. CGF 2018.
- **No 2024–2026 a-posteriori error estimator for 3D MPM dynamics was found.** A simulator-window certificate fills a real gap.
- **Cascades with guarantees:** Dekoninck et al. ICML 2025 (2410.10347): the quality estimator decides whether a cascade pays. BARGAIN 2509.02896. Calibrate-Then-Delegate 2604.14251. C3PO 2511.07396 (cost-side conformal bound).

## 3. Pilots proposed (round 1)

See PLAN.md once approved:
- **E1** stiff regime;
- **E2** cheaper verification (CPU);
- **E3** latency with parallel windows (CPU estimate; lowest priority).
