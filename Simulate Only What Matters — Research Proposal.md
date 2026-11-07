# Simulate Only What Matters — Research Proposal

Oct 7, 2026 · @Hasan

## Summary

Generate the whole physics trajectory once with a diffusion prior, then spend exact simulation only on the few short windows where it is likely wrong — to certify the result, repair it, and learn from it.

- **Working title:** *Simulate Only What Matters: Certified Sparse Simulator Probing for Physics-Grounded Video Generation*
- **Target venue:** CVPR 2027 — registration Nov 10, 2026; paper Nov 16, 2026; supplementary Nov 23, 2026.
- **Novelty verdict:** the full loop (sparse probes on a generated trajectory, active window choice, conformal accept/reject, constrained repair, amortization) has no match in a sweep of 2025–2026 work. Most single components exist in scientific ML, so the claim is the system, the residual-free verifier, and the compute-accounted protocol.
- **Base:** fork PhysCtrl (NeurIPS 2025) — released checkpoints, MPM and rigid data-generation scripts, test set.
- **Biggest risk:** probes must beat uniform compute *net of simulator cost*; a September 2026 audit found adaptive compute rarely does.

## Problem and motivation

Physics-grounded generation today forces a choice between two camps, and neither is satisfying.

| Camp | Examples | Strength | Weakness |
| --- | --- | --- | --- |
| Learned prior (diffusion, VLM) | PhysCtrl, PhysFlow, NGFF, Force Prompting | Fast; no scene specification at inference | Fails silently on contacts and out-of-distribution scenes; no error signal |
| Full simulation, then render | PSIVG, RealWonder, 3DPhysVideo, TourPhysics, PhysAgent | Exact, given correct parameters | Simulates every frame of every query; needs full scene setup |

**The original idea.** Train a model that learns physics continually. Clear cases are generated directly; ambiguous ones trigger a simulation, and the result is learned so similar cases don't need simulation next time.

**How it evolved.** The binary "simulate or not" became *where* to simulate: short windows checked against the exact simulator, at the moments most likely to be wrong (contacts, high uncertainty). This keeps the simulator's exactness without a full run — the core of the original intuition.

**Evidence the gap is real.** On Physics-IQ the best model scored 29.5% and visual realism did not track physical understanding. On StEvo-Bench every model scored under 10%. On RigidBench, higher SSIM went with *larger* 3D trajectory error across eight models (r = 0.89).

## Related work and novelty check

No paper found runs the full loop, but most single mechanisms already exist in scientific ML — so novelty is claimed for the system, not any one part. The sweep covered about 45 searches across vision, graphics, robotics and scientific ML up to late September 2026; it is strong evidence, not proof.

### Verdict per component

| Component | Verdict | Closest prior art | What remains yours |
| --- | --- | --- | --- |
| A. Event-aware probing | Partially covered | Hybrid Neural World Models (error peaks at contacts); PhysRVG (collision-weighted reward) | Using events to decide where exact simulator compute goes, at generation time |
| B. Active in-trajectory window selection | Covered in sciML; new in video/4D | ANCHOR, HyPER, AdaLED, Hybrid Neural-MPM | Non-causal choice on a trajectory generated in one pass; parallel probes |
| C. Probe assimilation by constrained re-sampling | Partially covered | Score-based data assimilation, Shysheya et al., PhysDiff/SimDiff, PSIVG | Model chooses its own simulator observations; repair propagates backward and forward |
| D. Audited certificate | Statistics exist; application new | Gopakumar CP-PRE, NeuroForge, prediction-powered and label-efficient conformal | Residual-free score (chaos-normalized simulator defect) for generative physics |
| E. Learning from probes | Done in sciML; new in video | AdaLED, Neural-Parareal, OGAS | Amortization curves for a generative image-to-video pipeline |
| Combined framework | No match found | PSIVG, PhysAgent, Hybrid Neural World Models | The full loop + matched-compute protocol |

### Most dangerous papers and one-line differentiation

| Paper | What it does | Our difference |
| --- | --- | --- |
| [Hybrid Neural World Models](https://arxiv.org/abs/2605.28317) (May 2026) | Step-doubling self-consistency error map; defers whole trajectories to a solver | We verify selected windows with the exact simulator, certify, and repair |
| [PSIVG](https://arxiv.org/abs/2603.06408) (CVPR 2026) | Simulates the full clip with MPM, guides video diffusion | Sparse, audited simulator calls only where needed |
| [PhysAgent](https://arxiv.org/abs/2607.16355) (Jul 2026) | VLM agent verifies and repairs the simulator program | We verify the generated trajectory, quantitatively, with a guarantee |
| [ANCHOR](https://arxiv.org/abs/2512.19643) (Dec 2025) | PDE-residual trigger for solver correction, sequential | No residual needed (contact, plastic MPM); non-causal windows |
| [HyPER](https://arxiv.org/abs/2503.10048) (2025) | RL picks which timesteps call the simulator | We verify a one-pass trajectory and assimilate via inpainting |
| [Hybrid Neural-MPM](https://arxiv.org/abs/2505.18926) (2025) | Heuristic step-wise fallback to MPM | Calibrated accept/reject with probe-based repair |
| [CP-PRE](https://arxiv.org/abs/2502.04406) / [NeuroForge](https://arxiv.org/abs/2607.10333) | Physics-residual conformal calibration | Simulator defect replaces the residual |
| [Compute-value audit](https://arxiv.org/abs/2609.13257) (Sep 2026) | Adaptive compute rarely beats uniform at matched cost | The bar we must clear; our main evaluation answers it |

### Concurrent work to watch

- [PhysStream](https://arxiv.org/abs/2609.17521) (Sep 2026) — from the PhysCtrl group; no simulator at inference, but a verification follow-up is a live risk.
- [TourPhysics](https://arxiv.org/abs/2609.04911) (Sep 2026) — simulate first, quality-gate each observation.
- [Prediction-powered quantile learning for conformal prediction](https://arxiv.org/abs/2609.33482) (Sep 2026) — pre-empts the statistics of D.
- [RigidBench](https://arxiv.org/abs/2608.15555) (Aug 2026) — simulator-grounded rigid evaluation; also a dataset for us.

## Title options and naming

Recommended: **Simulate Only What Matters: Certified Sparse Simulator Probing for Physics-Grounded Video Generation.** It states the idea in four words; "certified" and "sparse" separate it from dense simulation (PSIVG) and no simulation (PhysCtrl). Avoid "simulator in-the-loop" — PSIVG owns that phrase.

| Title | Emphasizes |
| --- | --- |
| Simulate Only What Matters: Certified Sparse Simulator Probing for Physics-Grounded Video Generation | The whole idea (recommended) |
| Trust, but Verify: Sparse Simulator Probes for Certified Physics-Grounded Video Generation | The verifier |
| Physics on Demand: Event-Aware Simulator Probing for Image-to-Video Generation | Events and contacts |
| Where to Simulate? Active Verification and Repair of Generated Physics | Active selection |
| Generate, Probe, Repair: Compute-Accounted Physics Verification for Video Diffusion | Pipeline and compute protocol |

**Method name.** "ProbeSim" is taken by a [2017 SimRank algorithm](https://arxiv.org/abs/1709.06955) — different field, but it clutters search. Candidates: **PhysProbe** or **SparseSim** (not yet checked for clashes).

## Contributions and what not to claim

**Thesis:** instead of choosing between a learned physics prior and a full simulation, generate the whole trajectory once and spend exact simulation only on the few short windows where it is likely wrong. Those windows certify the result and repair it, and the model learns from them so it needs fewer probes over time.

1. **Certified verification without a residual (strongest claim).** Short exact-simulator windows restarted from states lifted out of the generated trajectory give a defect score. Normalizing by the spread of perturbed simulations separates model error from chaos. Conformal calibration turns it into accept/reject with a coverage guarantee. Prior verifiers need a PDE residual (ANCHOR, CP-PRE, NeuroForge) or use self-consistency (Hybrid Neural World Models), so they don't cover rigid contact or plastic MPM.
2. **Active, event-aware probing with repair.** Windows are chosen after generation, at predicted contacts and high ensemble disagreement, on a one-pass trajectory. Verified windows are folded back by constrained re-sampling, so a fix at a collision also repairs earlier frames — *active smoothing with a simulator oracle*. Every prior adaptive scheme triggers sequentially inside an autoregressive rollout.
3. **Compute-accounted evaluation and amortization.** Accuracy vs total compute (generator steps + simulator steps + verifier) at matched budgets, plus curves of probes per query falling as the model learns from its own probes. No physics-grounded video or 4D paper reports either.

**Do not claim:**

- First to put a simulator in the loop (PSIVG, PhysAgent).
- Novel conformal statistics — cite prediction-powered and label-efficient conformal prediction.
- Novel adaptive solver triggering in general — ours is non-causal, residual-free and generative.

### Draft abstract

> Physics-grounded video generation faces a trade-off: learned dynamics priors are fast but fail silently on contacts and out-of-distribution scenes, while simulator-driven pipelines are exact but simulate every frame of every query. We propose \[Name\], which generates a full 3D trajectory with a diffusion prior and then spends exact simulation only where it matters. Short simulator windows, restarted from states lifted out of the generated trajectory at predicted contact events and high-uncertainty frames, provide a defect score that needs no PDE residual and therefore applies to rigid contact and plastic materials. A conformally calibrated threshold certifies accepted trajectories. Rejected ones are repaired by re-sampling with the verified windows as hard constraints, propagating corrections both forward and backward in time. Verified windows are replayed into the generator, so the number of simulator calls per query falls as the model learns. Under matched total compute, \[Name\] achieves \[X\]× fewer simulator steps than full simulation at \[Y\]% of its accuracy, and reduces physical error by \[Z\]% over learned-prior baselines, with empirical coverage at the target level.

Bracketed numbers are placeholders. If probing does not beat uniform compute, lead the abstract with contribution 1 instead.

## Framework

Most queries pass verification on the first try and cost one diffusion sample, a few short parallel simulator windows, and rendering; failures are repaired, and only repeated failures fall back to full simulation.

&#91;embedded content: full pipeline · main path, repair loop, fallback, learning loop\]

The right loop re-samples the whole trajectory with verified windows fixed, so a fix at a collision also corrects earlier frames. The dashed loop runs offline between batches: every probe and full simulation updates the generator, which is what makes probes per query fall.

1. **Generate.** A trajectory diffusion model (PhysCtrl's spatial–temporal DiT) produces the full 3D point trajectory in one pass from image, force and material.
2. **Locate.** Pick probe windows at predicted contacts and high-disagreement frames; always include frame 0 from the true initial state; add one random window.
3. **Probe.** Lift simulator state from generated positions; run short exact-simulator windows in parallel.
4. **Certify.** Accept if the chaos-normalized defect is below a conformally calibrated threshold.
5. **Repair.** Otherwise re-sample with verified windows as hard constraints; after K failures, run the full simulation.
6. **Learn.** Add verified windows to a replay buffer; LoRA-update the generator and a learned verifier.
7. **Render.** A frozen trajectory-conditioned video model turns tracks into video.

### Formal sketch

```latex
\begin{aligned}
&\hat{\tau} = (\hat{x}^0,\dots,\hat{x}^F) \sim p_\phi(\tau \mid s_0, \theta, a) \\
&W = \mathrm{select}(\hat{\tau}) \supseteq \{0\} \cup \mathrm{contacts}(\hat{\tau}) \cup \mathrm{disagree}(\hat{\tau}) \\
&\tilde{x}_w = S_\Delta\big(\mathrm{Lift}(\hat{\tau}, w)\big), \quad w \in W \\
&\delta = \max_{w} \lVert \tilde{x}_w - \hat{\tau}_w \rVert / \sigma_w ; \quad \text{accept if } \delta \le \hat{\lambda} \\
&\text{else } \hat{\tau} \sim p_\phi(\tau \mid s_0, \theta, a,\ \tau_W = \tilde{x}_W)
\end{aligned}
```

**Online objective** over a query stream t = 1…T: minimize total trajectory error plus λ × total compute (generator + simulator + verifier), subject to the certificate holding for accepted outputs.

**Reused from PhysCtrl:** perception lift, trajectory DiT backbone, trajectory-conditioned renderer. **New:** window verifier with state lifting, constrained re-sampling, calibrated threshold, probe replay.

## Window verifier details

The verifier is the component reviewers will scrutinize most; each part below is a design decision with a reason.

&#91;embedded content: window verifier · one window in detail\]

Each window restarts the exact simulator from a state lifted out of the generated frames and measures how far the generated frames drift from it, relative to the natural spread of perturbed simulations.

| Part | Design | Why |
| --- | --- | --- |
| Lift state | Velocity by central difference. Elastic: per-point deformation gradient F by least-squares fit of neighbours vs rest shape, plus a local velocity-gradient term. Rigid: Kabsch transform + angular velocity | The simulator needs more than positions |
| Material scope | Elastic and rigid only | Plastic and granular state depends on history that positions don't reveal |
| Placement | One window always at frame 0 from the true initial state; others at predicted contacts; one random | Anchor catches a wrong response to the force; random stops the model learning where it is checked |
| Cost | Windows run in parallel: wall-clock ≈ Δ/F of a full rollout, compute ≈ k·Δ/F. Target k·Δ ≈ 10–20% of F (e.g. 3 windows of 1–2 frames) | Savings must survive the compute-matched comparison |
| Compare | Per-point L2 at window end (point indices correspond), divided by the spread of a few perturbed simulations | Avoids flagging sensitive contacts as model error |
| Aggregate and gate | Max over windows; threshold set by conformal calibration on held-out full simulations; recalibrate after each model update | Max is conservative, which suits a guarantee; updates break exchangeability |

```latex
\delta = \max_{w \in W} \frac{\lVert \tilde{x}_w - \hat{\tau}_w \rVert_2}{\sigma_w}, \qquad \text{accept if } \delta \le \hat{\lambda} \text{ where } P\big(d(\hat{\tau}, \tau^*) > \varepsilon \mid \text{accept}\big) \le \alpha
```

**Sanity test to run first.** Feed ground-truth simulator trajectories through the lift and windows; the defect should be near zero. What remains is lifting error — the floor on what the verifier can detect. Report it as a table.

**Go/no-go experiment.** On held-out and out-of-distribution queries, check the rank correlation between the window defect and true trajectory error, against diffusion sample variance and a VLM judge. Strong correlation at 10–20% of full simulation cost means the paper has a core.

## Datasets

Most training and benchmark data must be generated by us, with the same simulator the probes call — otherwise the defect measures the gap between two simulators, not model error. Released datasets serve as starting points and external evaluation.

### What is available

| Dataset | Content | Status | Use for us |
| --- | --- | --- | --- |
| [PhysCtrl](https://github.com/cwchenwang/physctrl) | 550K MPM + rigid point trajectories (2,048 points) on Objaverse objects | Full set not released; [sample (1.33 GB) and test set (854 MB)](https://huggingface.co/datasets/chenwang/physctrl/tree/main); generation scripts for elastic, plasticine, sand, rigid; checkpoints released | Generator base + our own data generation |
| [RigidBench](https://github.com/swarnim-j/RigidBench) | 5 rigid-body tasks; per-frame 6-DoF poses, velocities, contacts (Blender Bullet) | 100-example eval set on [Zenodo](https://doi.org/10.5281/zenodo.21649156) (18.2 GB, CC BY 4.0); 5,000-example training set must be rendered from its manifest (Blender 4.5.9 + BlenderKit, some assets paid; 3 legacy assets delisted) | Rigid contact track; ground-truth contact events |
| [gs\_physics](https://huggingface.co/datasets/chkim403/gs_physics) | 529 real multi-object collision scenes, 4 synchronized views, Gaussian and pose data | Released | Real-world check of the full pipeline |
| [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark) | 396 real videos, 66 scenarios, motion-mask metrics | Released (materials CC BY 4.0) | Final video sanity check, solid-mechanics subset |
| NGFF GSCollision | 640K videos, \~4.25 TB, 10 objects | Skip | Too little object variety for its cost; different simulator |

### Examples from the datasets

&#91;image: PhysCtrl example inputs: sand bear, elastic chair, penguin, blanket and headphones, captioned with material and force\]

PhysCtrl's released image-to-video example inputs, each captioned with the material and force set in its config. The same scripts produce the trajectories we train and probe on.

&#91;image: Physics-IQ real videos at frames 0, 20, 40 and 59: domino chain, liquid pouring, balloon pop, kettlebell on pillow\]

Physics-IQ real footage at four time steps: a domino chain (a contact cascade — exactly where window probes should go), liquid pouring, a balloon pop, and a kettlebell deforming a pillow. Images: Physics-IQ (Google DeepMind, CC BY 4.0) and the PhysCtrl repository.

### Plan by purpose

| Purpose | Dataset | Source |
| --- | --- | --- |
| Generator training | PhysCtrl-style trajectories regenerated with their scripts; fine-tune from their checkpoint | Released scripts + checkpoint |
| Query stream (main benchmark) | Our own split, below | We generate |
| Rigid contact track | RigidBench eval + rendered training manifest | Released / we render |
| Conformal calibration and audits | Held-out slice of our stream with full simulations | We generate |
| Real-world check | gs\_physics; Physics-IQ solid mechanics | Released |
| Final video quality | PhysCtrl test set + VLM judge + user study | Released / we run |

### Query stream design

| Split | What changes | Why it matters |
| --- | --- | --- |
| In-distribution | Nothing — training materials, stiffness and categories | Probes should rarely fire here |
| Parameter OOD | Stiffness (Young's modulus) outside the training range | Tests the gate on known physics, new values |
| Geometry OOD | Held-out Objaverse categories | Tests shape generalization |
| Force OOD | Non-gravity forces on plasticine, sand, rigid | PhysCtrl's large checkpoint supports only gravity there — a clean gap from the released model itself |
| Multi-object contact | Two or more colliding objects (extend their scripts) | Where simulation is expensive and the method should help most |

Each query stores: the full ground-truth simulation; a few perturbed simulations for chaos normalization; contact-event labels; and a calibration/audit vs test flag.

### Size, compute and checks

- **Size:** 2,048 points × 24 frames × 3 floats ≈ 0.6 MB per trajectory; all 550K ≈ 320 GB. Tens of thousands for fine-tuning plus a 5–10K-query stream is enough.
- **Cost to measure in week 1:** simulator seconds per trajectory vs generator seconds — this decides whether the savings story holds.
- **Replay determinism:** re-simulate stored trajectories from their initial state and match the stored result. Matters most for RigidBench: probing with PyBullet instead of Blender's Bullet would add simulator mismatch.
- **Lift floor:** push ground-truth trajectories through lifting and probing; the defect should be near zero.

## Experiments, baselines and metrics

The headline result is an accuracy-vs-total-compute Pareto curve at matched cost; everything else supports it.

### Baselines

| Baseline | What it tests |
| --- | --- |
| Always generate (PhysCtrl, PhysFlow) | Value of any simulation |
| Always simulate (PSIVG-style) | Upper bound on accuracy, cost ceiling |
| Uniform windows, same budget | Value of choosing where to probe |
| Random windows, same budget | Same, weaker control |
| Threshold gating, ANCHOR-style | Sequential trigger vs non-causal choice |
| Whole-trajectory deferral, Hybrid-NWM-style | Window probes vs trajectory deferral |
| Best-of-N selection (compute-value audit) | Generic test-time compute |
| Oracle placement (knows true error) | Ceiling for window choice |

### Results the paper must show

- **Pareto:** error vs total compute (generator steps + simulator steps + verifier), area under curve.
- **Placement ablation:** event-aware vs uncertainty-only vs combined.
- **Assimilation ablation:** constrained re-sampling vs simply replacing probed frames; evidence that an event fix changes earlier frames.
- **Certificate:** empirical coverage at target α, as audit rate falls and under drift.
- **Learning:** probes per query over the stream at stable error; vs learning from full rollouts at equal simulator budget.
- **Lift floor:** defect on ground-truth trajectories (sanity table).
- **Video level:** VLM judge + user study vs PhysFlow, RealWonder, 3DPhysVideo; RigidBench and Physics-IQ subsets.

### Metrics

| Level | Metrics |
| --- | --- |
| Trajectory | Per-point L2, Chamfer distance, volume IoU (PhysCtrl's), ATE-3D (RigidBench's) |
| Compute | Simulator steps, generator steps, wall-clock, total GPU-seconds |
| Certificate | Coverage vs α, risk–coverage curve, rank correlation of defect vs true error |
| Video | Physics-IQ motion-mask scores, VLM judge, user preference |

## Risks, reviewer objections and fallbacks

The two risks that can sink the paper are a small simulation-cost gap and a verifier that doesn't track true error; both are tested in the first two weeks.

| Risk or objection | Answer or mitigation |
| --- | --- |
| "This is ANCHOR / Neural-Parareal applied to vision" | No residual needed; choose where in a trajectory; whole-trajectory generative repair; compute-matched visual evaluation. Cite them prominently. |
| Simulation is cheap anyway (RealWonder runs in real time) | Measure the cost ratio in week 1; focus on regimes where it is large: multi-object contact, high-resolution MPM |
| Probes don't beat uniform compute (compute-value audit) | Uniform-probing baseline at matched cost is mandatory; if lost, lead with the certificate |
| Perception is the real bottleneck | Measure all errors against a simulation from the same lifted state; perception is a separate error term |
| Lifting MPM state from positions | Restrict to elastic + rigid; report the lift floor |
| Chaos at contacts inflates the defect | Normalize by perturbed-simulation spread |
| Guarantee is weak | State the marginal, exchangeable assumption; recalibrate after updates; show empirical coverage under drift |
| Constrained re-sampling converges no faster than Parareal | Contributions 1 and 3 still stand without it |

### Publishable even if the full vision falls short

- The compute-accounted benchmark and protocol alone.
- A calibration study: when can a physics diffusion prior's uncertainty be trusted?
- Residual-free certification for contact and plastic materials.
- A trajectory-space model that keeps evolving state under occlusion (StEvo-style) — untested hypothesis.

## Timeline to CVPR 2027 and what to cut

Five work weeks remain before writing; the week-1 checks decide whether the efficiency story holds or the paper leads with the certificate.

&#91;embedded content: CVPR 2027 plan · 5 work weeks, 3 deadlines\]

| Week | Go/no-go check |
| --- | --- |
| Verifier go/no-go | Simulation ≥ \~10× slower than generation in our regime; window defect rank-correlates with true error better than sample variance |
| Stream + baselines | Gate beats random windows at matched budget |
| Repair + calibration | Constrained re-sampling beats replacing probed frames; coverage near target α |
| Online loop, curves | Teaser figure exists: probes per query fall, error flat |

**Cut in this order if behind:** multi-object scenes → Physics-IQ and gs\_physics evaluation → user study (keep the VLM judge). **Never cut:** the matched-compute Pareto curve, the uniform-probing and oracle baselines, the calibration result.

## References

**Physics-grounded generation**

- [PhysCtrl](https://arxiv.org/abs/2509.20358) · [code](https://github.com/cwchenwang/physctrl) · [data](https://huggingface.co/datasets/chenwang/physctrl)
- [Force Prompting](https://force-prompting.github.io/)
- [PhysFlow](https://arxiv.org/abs/2609.08215)
- [Neural Gaussian Force Fields](https://github.com/lishiqianhugh/NeuralGaussianForceField)
- [PSIVG — Physical Simulator In-the-Loop Video Generation](https://arxiv.org/abs/2603.06408)
- [3DPhysVideo](https://arxiv.org/abs/2605.16795)
- [TourPhysics](https://arxiv.org/abs/2609.04911)
- [PhysAgent](https://arxiv.org/abs/2607.16355)
- [PhysStream](https://arxiv.org/abs/2609.17521)
- [PhysRAG](https://arxiv.org/abs/2606.26916)
- [PhysRVG](https://arxiv.org/abs/2601.11087)

**Hybrid simulation, verification and calibration**

- [Hybrid Neural World Models](https://arxiv.org/abs/2605.28317)
- [Hybrid Neural-MPM](https://arxiv.org/abs/2505.18926)
- [ANCHOR](https://arxiv.org/abs/2512.19643)
- [HyPER](https://arxiv.org/abs/2503.10048)
- [AdaLED](https://arxiv.org/abs/2304.01732)
- [Learning Where to Simulate (OGAS)](https://arxiv.org/abs/2606.09949)
- [Neural-Parareal](https://arxiv.org/abs/2405.01355)
- [Calibrated Physics-Informed UQ (CP-PRE)](https://arxiv.org/abs/2502.04406)
- [NeuroForge](https://arxiv.org/abs/2607.10333)
- [Prediction-powered quantile learning for conformal prediction](https://arxiv.org/abs/2609.33482)
- [Compute-value audit of test-time scaling for video world models](https://arxiv.org/abs/2609.13257)

**Benchmarks and datasets**

- [Physics-IQ](https://arxiv.org/abs/2501.09038)
- [NewtPhys](https://astra-vision.github.io/NewtPhys/)
- [StEvo-Bench](https://arxiv.org/abs/2603.13215)
- [RigidBench](https://arxiv.org/abs/2608.15555) · [code](https://github.com/swarnim-j/RigidBench)
- [Real multi-object collision dataset (gs\_physics)](https://huggingface.co/datasets/chkim403/gs_physics) · [paper](https://arxiv.org/abs/2605.23845)

**Venue**

- CVPR 2027 dates: registration Nov 10, paper Nov 16, supplementary Nov 23, 2026.
