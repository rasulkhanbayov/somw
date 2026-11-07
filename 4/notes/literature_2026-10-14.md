# Literature check (Oct 14, 2026)

Done by a research agent with web search; it reports opening every arXiv id below unless marked "unverified". Re-check each id before citing. No paper was found that combines a stochastic generative model of a whole trajectory, sparse exact-simulator probes restarted from lifted states, and an object-level conformal risk guarantee with simulated-suffix repair.

## Table

| Work | Year / venue | Threat | Baseline or metric reviewers will expect | What we borrow |
|---|---|---|---|---|
| Hybrid Neural World Models, [2605.28317](https://arxiv.org/abs/2605.28317) | arXiv, May 2026 | **high**: a per-trajectory error map gates fallback to the solver, including rigid collisions; it compares against conformal and ensembles | whole-trajectory deferral on the model's own uncertainty → our **cert_ss** (4m) | its label-free baseline set |
| ANCHOR, [2512.19643](https://arxiv.org/abs/2512.19643) | arXiv, Dec 2025 | **high (concept)**: online correction triggered by a PDE residual | a residual-triggered gate → **momentum-residual baseline** (plan item 4) | adaptive threshold |
| PSIVG, [2603.06408](https://arxiv.org/abs/2603.06408) | CVPR 2026 | medium: simulates the whole clip, then guides the video | "why not simulate everything?" | framing contrast |
| RealWonder, [2603.05449](https://arxiv.org/abs/2603.05449) | arXiv, Mar 2026 | medium: always simulate (13.2 FPS) | always-simulate baseline (we have it) | its cost argument |
| HyPER, [2503.10048](https://arxiv.org/abs/2503.10048) | 2025 | medium: an RL policy decides when to call the simulator | random calls at the same budget (have: random fallback) | — |
| NeuroForge, [2607.10333](https://arxiv.org/abs/2607.10333) | arXiv, Jul 2026 | medium: residual + split-conformal trust | — | **independent support for our Goodhart finding (4c): a residual is a trust signal, not a correction objective** |
| TACR, [2610.02487](https://arxiv.org/abs/2610.02487) | arXiv, Oct 2026 | medium: conformal routing between surrogate and simulator (scalar QoI) | — | routing-guarantee wording |
| Physics-residual CP, [2502.04406](https://arxiv.org/abs/2502.04406); CP for surrogates, [2408.09881](https://arxiv.org/abs/2408.09881) | 2025 | medium | — | contrast with "residual-free" |
| Round-Trip Consistency, [2608.00675](https://arxiv.org/abs/2608.00675) | arXiv, Aug 2026 | medium: self-predicted rollout error | self-consistency baseline | — |
| Neural-Parareal, [2405.01355](https://arxiv.org/abs/2405.01355); Hybrid Neural-MPM, [2505.18926](https://arxiv.org/abs/2505.18926) | 2025 | medium: shadow + suffix looks like one Parareal step | explain the difference | — |
| Conformal Cascade, [2607.25018](https://arxiv.org/abs/2607.25018) | arXiv, Jul 2026 | low–medium | — | **a 1 − Kα union bound for our certify → repair chain** |
| Dunn, Wasserman & Ramdas (JASA 2023), hierarchical conformal | — | tool | — | **formal basis for object-level exchangeability** |
| PhysStream, [2609.17521](https://arxiv.org/abs/2609.17521) | arXiv, Sep 2026 | **low**: autoregressive video, rigid bodies, no simulator check | FVD, FVMD, Traj-ADE (2D), failure rate, Physics-IQ | its video-metric protocol |
| Proprio, [2605.28230](https://arxiv.org/abs/2605.28230); GEOPHYS, [2606.20707](https://arxiv.org/abs/2606.20707) | 2026 | medium (test-time verification, best-of-N) | **best-of-N with our defect at matched compute** (plan item 4) | — |
| TourPhysics, [2609.04911](https://arxiv.org/abs/2609.04911); PhysAgent, [2607.16355](https://arxiv.org/abs/2607.16355) | 2026 | low–medium | — | — |
| RigidBench, [2608.15555](https://arxiv.org/abs/2608.15555) | arXiv, Aug 2026 | none | 3D trajectory error vs simulated reference | **metric precedent for item 3** |
| Physics-IQ, [2501.09038](https://arxiv.org/abs/2501.09038); VideoPhy-2, [2503.06800](https://arxiv.org/abs/2503.06800) | WACV / ICLR 2026 | none | video-level physics scores | — |
| Break-even deferral, [2608.23075](https://arxiv.org/abs/2608.23075) | 2026 | — | — | closed-form break-even deferral fraction (a correct gate can still slow things down) |
| PCFM, [2506.04171](https://arxiv.org/abs/2506.04171); Predict-Project-Renoise, [2601.21033](https://arxiv.org/abs/2601.21033); CondMDI, [2405.11126](https://arxiv.org/abs/2405.11126); Diffusion Forcing, [2407.01392](https://arxiv.org/abs/2407.01392) | 2024–2026 | — | — | **CondMDI: inference-time imputation underperforms masked training; consistent with our 4b/4g** |
| Adaptive conformal, [2106.00170](https://arxiv.org/abs/2106.00170); OGAS, [2606.09949](https://arxiv.org/abs/2606.09949); budgeted deferral, [2510.26706](https://arxiv.org/abs/2510.26706) | — | — | — | amortisation tools (plan item 5) |
| PhysWeep, [2609.06207](https://arxiv.org/abs/2609.06207) (Khanbayov & Kurban) | arXiv, Sep 2026 | none | — | possibly the user's own work; check and cite |

**Corrections to the proposal:**
- **gs_physics could not be verified** (the agent found no such benchmark; closest: GaussianBench, [2610.10554](https://arxiv.org/abs/2610.10554)). Do not cite it until checked.
- **StEvo-Bench** ([2603.13215](https://arxiv.org/abs/2603.13215)) is about state evolution; it is weakly relevant.

## Positioning sentences (draft)

1. **Against gated hybrids** (HNWM, ANCHOR, HyPER): "Prior hybrids gate a deterministic surrogate on a signal computed from the surrogate itself and give no finite-sample statement; we gate a stochastic generative model of a whole trajectory with exact-simulator probes from lifted states, need no residual, and certify P(err > ε) ≤ α over accepted trajectories with object-level conformal calibration."
2. **Against conformal for surrogates** (TACR, NeuroForge, CP-PRE): "We calibrate a simulator-measured defect of a generated trajectory and control the selective risk of the accepted set, under object-level exchangeability."
3. **Against simulator-in-the-loop video** (PSIVG, RealWonder): "They simulate every query in full; we simulate short windows, certify, and re-simulate only uncertified suffixes." This needs the cost advantage at matched accuracy; 4k shows it.
4. **Against Parareal**: "A single verification-gated restart, not an iteration to convergence; the predictor is a learned generative prior; the coarse solver only supplies the full state at the restart."
5. **Against best-of-N verifiers**: "They rank samples by plausibility; our certificate bounds error against ground truth."

## Baselines reviewers will most likely demand (ranked)

1. A cheap simulator with a certificate, or cross-fidelity agreement: **XF in 4m**.
2. Conformalised model-uncertainty deferral: **cert_ss in 4m** (seed spread).
3. A residual-triggered gate (momentum/energy residual on generated positions): plan item 4.
4. Best-of-N with the defect as verifier at matched compute: plan item 4.
5. Random placement and fallback at the same budget: have the fallback; uniform placement is in 4m M4.
6. Video metrics (FVD/FVMD, Traj-ADE, VideoPhy-2 AutoEval), a small human study: plan item 2/6.
7. Risk–coverage curves, AURC, per-split risk, and the union bound over the chain: writing.
8. RigidBench-style 3D error for item 3.
