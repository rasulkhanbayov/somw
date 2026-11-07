# Turning the work into a model paper: options (Oct 22, 2026)

**Brief (the user).** Make this a model paper, not an evaluation or analysis paper. Do a full literature search, audit every weakness, and give concrete options, including changes to the pipeline and new experiments.

**Status.** Proposal only. No code ran, nothing was pre-registered, and the paper is unchanged. Every external claim comes from three research agents who opened each source (arXiv abstract or full HTML); their raw notes are summarised here. Numbers about our own work come from PLAN.md and `results/`.

**Bottom line.**
- The strongest model-paper route is **Option A: certificate-driven simulator distillation**. The certificate decides *which* queries are worth simulating. Only those rollouts are fed back to fine-tune the generator, so the trained generator alone becomes reliable on materials and forces where PhysCtrl fails. This is the proposal's original third contribution ("verified windows are replayed into the generator"), done in the form the literature says works: supervised training on simulator targets with replay, not repair at inference and not optimising the checker.
- **Option B** (a contact-aware two-object generator) has the most visible video wins, but it is more crowded and riskier before Nov 16.
- **C** (a learned verifier) and **D** (a learned corrector of the cheap simulator) are weaker as standalone model papers.

---

## 1. What the literature says (2024–2026, verified)

### Physics-grounded trajectory and video generators (the field we would join)

| Work | What it trains | Relevance |
|---|---|---|
| **PAVG** (arXiv 2610.11791, posted **Oct 8, 2026**) | A PhysCtrl-style dual-branch (solid/gas) trajectory diffusion model on 700K simulated trajectories, using PhysCtrl's own data recipe for solids | **Strongest threat for a "better generator" paper.** It already beats PhysCtrl in distribution (solid vIoU 0.612 vs 0.588; CD 0.017 vs 0.023). It does not test out-of-range materials or forces. |
| **NEXUS** (2606.15015) | An energy + dissipation GNN with an integrator; gravity, floor, multi-object contact | Replaces PhysCtrl's generator inside PhysCtrl's own pipeline and compares at video level only. Threat for any contact model. |
| PhysiFormer (2606.27364), PhysLDM (2610.07609) | Diffusion over multi-object mesh trajectories (Genesis, >100K scenes, no force input); latent diffusion over deformable trajectories (IPC-FEM) | Trained trajectory generators. PhysLDM notes that GPU-simulator non-determinism is amplified by chaos, which argues for distribution-level scoring. |
| DSO (ICCV'25, 2503.22677) | DPO/DRO fine-tuning of an image-to-3D generator (TRELLIS) on **simulator stability labels**, self-improving on its own samples | **Same recipe family as Option A**, but for static shapes. |
| Phys4D (2603.03485) | Video diffusion: supervised training on simulation, then RL with reward = −4D Chamfer to the simulated trajectory | Simulator-grounded post-training, in video space. |
| PISA (2503.09595), NewtonRewards (2512.00425), PhyCo (2604.28169), PhysMaster (2510.13809), PhysRVG (2601.11087) | Video models: supervised training on simulated videos, then rewards or DPO | Most of the gain comes from **supervised training on simulated targets**. PISA documents weak out-of-distribution transfer. |
| Morph (ICCV'25, 2411.14951), SC-SC (ICML'24, 2402.07087) | Human-motion generators fine-tuned on their own samples after simulator correction (physics-based imitation), over rounds | **Closest template for "feed corrected outputs back into the model".** In distribution only. |

**Gap (verified):**
- No paper post-trains a **3D point-trajectory generator** with simulator feedback.
- No paper reports PhysCtrl's **trajectory error under material/force shift**. Every comparison against PhysCtrl uses VLM or human video judges, except PAVG, which is in distribution only.
- Kang et al. (ICML'25, 2411.02385) show that video diffusion trained on simulation generalises "case-based": coverage of the training data decides out-of-distribution behaviour.

### Training with simulator feedback: what works and what doesn't

- **Works:** supervised distillation of simulator rollouts, with replay of original data against forgetting (PISA, NewtonRewards, PhyCo, Morph, SC-SC).
- **Low value for us:**
  - DPO/GRPO ranked by the simulator. Under shift all 8 samples are bad (our 7a: oracle best-of-8 risk 0.676), so the pairs are "bad vs worse". PhyGDPO uses real videos as winners for this reason.
  - Rewards on residuals or checkers get hacked. NewtonRewards saw velocity collapse with the residual reward alone, the published analogue of our 4c.
- **Unlikely:** PhysDiff-style projection at sampling. With 4 DDIM steps there is no room. PhysDiff itself finds post-hoc projection useless. For a passive object with a known start, the projection is simply simulation.
- **Why our 4b/4g failed (independent support):** CondMDI (2405.11126) finds that imputed known frames are "completely ignored" unless the model is trained with random keyframe masks from the start. LoRA trades plasticity against forgetting (2405.09673).

### Learned simulators, hybrids, verifiers

- **Hybrids with solver fallback exist,** with heuristic or quantile triggers and no guarantees: Hybrid Neural-MPM (2505.18926), Hybrid Neural World Models (2605.28317), ANCHOR (2512.19643).
- **MPM plus a learned residual on deformables exists:** PhysCoRe (2607.20653), WorldParticle (2605.15305).
- **"Learned verifier + conformal acceptance" is a known recipe** (Conformal Alignment 2405.10301; Freno–Carlberg error models). It would be new only for its simulator labels on deformables and its behaviour under shift.
- **Self-uncertainty fails out of distribution.** Round-Trip sampling dispersion inverts; generative coverage is poor. This matches our seed-spread result.

---

## 2. Weaknesses a CVPR reviewer would raise (most serious first)

| # | Weakness | Reject risk | Fixed by a model contribution? |
|---|---|---|---|
| 1 | "A wrapper, not a model: if in doubt, run the simulator." On 52–57% of shifted and contact queries our output *is* the simulator's | very high | **A, B yes** (a trained generator is the contribution). C partly. |
| 2 | Circularity: "better than PhysCtrl" is measured by the solver we call | high | **A partly:** the trained generator runs *without* the simulator at test time on held-out objects and conditions. It is still scored against the simulator, as PhysCtrl and PAVG are. Phase 12 (independent FEM) and Phase 13 (PokeFlex) address the rest. |
| 3 | Weak cost premise: a tuned simulator is cheaper in known regimes | high | **A yes, if it works:** the improved generator costs generation time only (about 0.05× a reference rollout) on the conditions it learned. |
| 4 | No video-level win (VLM judge invalid; first rater preferred PhysCtrl) | high | **B best** (ignored collisions are visible). A modest. Neither fixes the soft-material realism issue. |
| 5 | The reference is unconverged or unrealistic in some regimes (soft materials and strong pushes; no gravity during a push; 1 m hollow shells) | medium–high | **No.** Needs decision C (realistic-regime split), Phase 12, Phase 13, and realistic test settings. |
| 6 | New competitors: PAVG beats PhysCtrl in distribution; NEXUS handles contact | medium–high | A competes where they are untested (out-of-range materials/forces). B competes head-on with NEXUS. |
| 7 | PhysCtrl base has no gravity or floor input, so drop is an unfair shift | medium | A can add a zero-initialised gravity/floor token and train on drops, or drop the family. |
| 8 | The certificate leaks for stiff materials (6/16 bad accepted) | medium | A trains on stiff data; the certificate is recalibrated. |
| 9 | Narrow scope (elastic, one generator) | medium | No, within the time available. |
| 10 | A complex story with many negatives | medium | A gives one clean story: the certificate tells the simulator where to teach the model. |

---

## 3. Options

### Option A: certificate-driven simulator distillation (recommended)

**Idea.**
1. The generator proposes; the window certificate (probes only) flags queries where it departs from physics.
2. **Only flagged queries are simulated at the reference setting.** These rollouts are aggregated over rounds and used to fine-tune the generator, with replay of the original distribution.
3. The result is a generator that, *alone*, is reliable across a wider range of materials and forces. The certificate remains a held-out safety net and is never a training objective.

**What is new:**
- simulator-verified post-training of a 3D point-trajectory generator, which no verified paper does;
- **active acquisition by a calibrated certificate**, compared against random acquisition at equal simulator budget ("simulate only what matters" becomes a training principle);
- measured out-of-distribution gains on held-out objects.

**Closest prior work:**
- DSO, which does simulator-labelled DPO for static 3D shapes, not trajectories, and has no certificate-driven acquisition.
- Phys4D and PISA, in video space. PISA reports poor out-of-distribution transfer.
- Morph and SC-SC, human motion, in distribution, with no acquisition rule.
- Kang et al., which predicts coverage matters. Our certificate is a way to buy coverage cheaply.

**Why this is not a repeat of our failures:**
- It trains on full simulator targets for *new conditions*, so it is supervised data, not inference-time constraints (4b/4e) and not frame conditioning (4g, which also used in-distribution data only).
- It never optimises the window defect (4c).
- It uses replay and a non-degradation check against the forgetting seen in 4g.

**Fixes:** weaknesses 1, 3, 8, 10; 2 partly; 7 if a gravity token is added.
**Does not fix:** 5 (realism), 9 (scope); 4 only modestly.

**Experiments (to be pre-registered):**
- **Data:**
  - Training objects disjoint from the test objects: PhysCtrl's `sample/` 100 objects plus about 1–2k Objaverse objects via PhysCtrl's own generator and UID list (168k valid IDs; HTTPS download, CPU).
  - Condition space: log E 3–8.5, force 0.02–0.8. Optionally gravity drops with a zero-initialised gravity/floor token.
- **Arms at an equal simulator budget B:**
  - **R:** random queries over the expanded space.
  - **A:** candidate queries are generated, the certificate rejects some, and only rejected ones are simulated. Probe cost counts towards B.
  - **A-rounds:** two to three rounds with the updated generator.
- **Training:** fine-tune base with at least 50% replay of the original distribution; learning rate as in 4g-v2 or lower; LoRA vs low-learning-rate full fine-tune chosen on the pilot only.
- **Tests (fresh draws, the 100 held-out PhysCtrl test objects):**
  - **M1:** the trained generator alone has lower risk than PhysCtrl on shifted conditions inside the trained range (CI).
  - **M2:** arm A beats arm R at equal budget (the active-acquisition claim).
  - **M3:** in-distribution non-degradation, |Δ error| ≤ 0.005.
  - **M4:** on a further shift *outside* the trained range, the recalibrated certificate's false-accept rate ≤ α. The certificate still matters.
  - **M5:** the certified pipeline built on the new generator is cheaper at equal risk.
- **Pilot and kill criterion:**
  - One round of about 1,500 rollouts on development objects and conditions.
  - Kill if development-shift risk does not fall by ≥ 30% relative, or in-distribution error degrades by > 0.005.
  - About 6–8 GPU-h, 2–3 days.
- **Cost:**
  - Data about 8–16 A100-h per arm (4.67 s per reference rollout, measured), training about 4–6 H200-h per arm, evaluation about 4 GPU-h.
  - **About 35–55 GPU-h in total**, at or slightly above the 50 GPU-h line, so your approval is needed.
  - Engineering: Objaverse preparation, 1–2 days.
- **Calendar:** pilot Oct 24–27 → main Oct 28–Nov 5 → fresh test Nov 6–8 → writing Nov 9–16. Tight but feasible.
- **Headline if it works:** "A generator that learns from only the queries it gets wrong: X% lower risk under material/force shift with no simulator at inference, Y× fewer simulations than random augmentation, and a certificate that still guards beyond the trained range."
- **Main risks:**
  - The gain is "just more data". M2 (active vs random) is the answer.
  - Gains do not transfer to held-out objects (PISA's warning). The pilot tests this.
  - Reviewers ask for PAVG. Its code availability is unknown; we would state that it is untested out of distribution.

### Option B: contact-aware two-object generator

**Idea.** Train a joint two-object trajectory model (4,096 points) with an object-ID embedding and cross-object attention, initialised from base, on simulated two-object scenes. Keep the shadow-timed switch and certificate as the safety net. Headline: PhysCtrl composes objects independently and ignores collisions (contact error 0.305 in 4l); this model resolves them.

- **New:** a force-conditioned point-trajectory generator with object–object contact.
- **Threats:**
  - NEXUS: GNN plus integrator with contact; it replaces PhysCtrl's generator, compared at video level.
  - PhysiFormer: multi-object trajectory diffusion, no force input.
  - PAVG.
  - The field is crowded.
- **Fixes:** 1 and **4 (visible video wins)**; 6 partly.
- **Cost:**
  - Data: about 10k scenes × ~9 s, roughly 25 A100-h, plus engineering. We have 1,336 joint scenes on 100 objects.
  - Training: a 4,096-token DiT, roughly 15–30 H200-h.
  - **About 45–60 GPU-h**, and the riskiest option for Nov 16.
- **Pilot:** fine-tune on the existing 1,336 scenes; kill if held-out contact error does not drop below the composed baseline by ≥ 30%.

### Option C: amortised verifier ("a generator that knows when it is wrong")

**Idea.** A network that predicts the simulator defect, or the true error, from (generation, conditions), calibrated conformally, with sparse audited probes kept (our 5a design).

- **New:** simulator labels for a 3D trajectory generator.
- **Threats:**
  - Conformal Alignment makes the recipe known.
  - Hybrid Neural World Models finds a learned error head no better than a free signal.
  - Thümmler (2608.23075) argues a correct out-of-distribution gate cannot save cost when the stream leaves the training distribution.
- **Our own evidence is against it:** 8b fused certificate NO-GO (held-out false-accept 0.35); 5a saved 1.3%.
- **Cost:** about 1–3 GPU-h, since we have about 10k labelled generations.
- **Verdict:** weak as the main contribution; useful inside A as a cheaper acquisition filter.

### Option D: learned corrector of the cheap simulator (neural–numerical hybrid)

**Idea.** A Kochkov-style learned correction makes the M1 shadow and suffix approach the reference, so repairs get cheaper.

- **Threats:** Hybrid Neural-MPM, PhysCoRe, Solver-in-the-loop.
- **Weaknesses:** fragile out of distribution (Stachenfeld et al.); the reference itself is unconverged under shift (10a).
- **Verdict:** a physics-ML paper rather than a CVPR generation paper, and it does not fix weakness 1. Not recommended.

### Option E: same assets, different frame

A + benchmark: "PhysCtrl under shift". The first simulator-scored out-of-distribution benchmark for trajectory generators (our Phase 9/10c sets, plus the FEM cross-check and PokeFlex if they land), together with the Option A model. This fills the verified evaluation gap and gives the model a home. It is an add-on to A, not a separate path.

### Not proposed (dead ends, with reasons)

- **DPO/GRPO from simulator rankings:** under shift all samples are bad.
- **PhysDiff-style projection:** no room with 4 steps; it collapses to simulation.
- **Another inference-time repair or frame-conditioning attempt:** CondMDI says it needs masked training from scratch, which costs about full pre-training.
- **Optimising the certificate:** Goodhart (4c; NewtonRewards; TaRoS).

---

## 4. Recommendation and plan to Nov 16

**Rank by (chance of a strong CVPR result) ÷ (cost and risk):** A > B > C > D.

**Plan:**
1. **Now:** pre-register Option A's pilot (development objects only; kill criterion fixed). Prepare Objaverse object preparation on CPU in parallel. Keep Phase 12 (CPU) and Phase 13 (when the data arrives) running as independent evaluations.
2. **Pilot (Oct 24–27):** if killed, stop A and finish the current paper honestly (fidelity framing). If it passes, pre-register the full A on fresh data.
3. **Full A (Oct 28–Nov 8):** arms R and A at equal budget, rounds, fresh held-out test, certificate recalibration.
4. **Writing (Nov 9–16):** the paper becomes *model first*. The certificate and repair become the safety net and the acquisition rule, not the headline. Abstract registration by Nov 10.

**Cuts:**
- Option B (unless A's pilot passes by Oct 26 and A100s are idle).
- Option D.
- More video studies beyond the running one.

**If A fails:** keep the current paper (certificate + repair + fidelity switch), add A's negative result honestly, and consider a venue that values careful evaluation.

**Decisions needed from you:**
1. The option (A recommended; B as a stretch).
2. Approval for about 35–55 GPU-h for A, which may exceed 50.
3. Objaverse download (a few GB of meshes over HTTPS, to `$BIG_ROOT`).
4. Decisions A and C from the simulator audit still stand. Under Option A, decision C's realistic-regime split becomes part of the evaluation.
