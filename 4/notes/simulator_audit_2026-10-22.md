# Simulator and metric audit after the first study rater (Oct 22, 2026)

**Question (the user's hypothesis).** The first rater found PhysCtrl's videos more plausible than ours in 24/28 items, and said some were easy: two "heads dropped to the floor", two looked weird, four or five "changed direction when lifted". Is our reference simulator set up wrong, so that the error we reduce is the wrong target?

**Short answer.** No bug was found: the reference reproduces PhysCtrl's own training-data simulator line by line. But "faithful to that simulator" and "physically realistic for the pictured object" separate in three identifiable places: an unrealistic material and force regime in 6 of the 23 Phase 11 items, artifacts added by the video renderer, and an error metric dominated by motion the viewer never sees. The hypothesis is partly supported (realism, perception) and partly not (simulator correctness, Phase 9 results outside one family).

Everything here is exploratory, computed after the rater's answers were seen. No paper text, study page or pre-registered analysis was changed.

---

## 1. Verified facts

### 1.1 The reference is PhysCtrl's own data simulator (no setup bug)

Compared against PhysCtrl's training-data generator, `data/physctrl/src/data_generation/generate_mpm_data.py`, and inference, `src/inference.py`:

| Setting | PhysCtrl data generator | Ours |
|---|---|---|
| Young's modulus | log E ~ U(4, 7) (l. 109) | Phase 11: log E ∈ {4, 5, 6} (`p11_images.py` rule); Phase 9 per family |
| Poisson ratio | ν ~ U(0.05, 0.45) (l. 111) | 0.3 |
| Gravity | **off** when a drag force exists (l. 130–131) | off (`p6_video.py:88`, `gravity: 0`) |
| Floor | y = 0.2, far below the object at y ≈ 4.5 (l. 124, 158) | 0.2 (`p6_video.py:88`) |
| Force magnitude | coeff ~ U(0.02, 0.2) × base_drag_coeff (l. 168, 185) | Phase 11: 0.06 (in range), **0.40 (shifted)** |
| base_drag_coeff | 9.8·ρ·mean masked volume / mask ratio (l. 183) | identical (`regimes.py:25–28`) |
| Push duration | constant for the whole clip (`num_dt = step_per_frame × frame_num`, l. 187–188) | identical (`physctrl_bridge.py:313–314`) |
| Push patch | box of half-size 0.4 (l. 37) | 0.4 (`physctrl_bridge.py:56`; `p6_video.py:79`, as `inference.py:217–218`) |
| Density, grid, domain | 1000, 125, 10 (`configs/objaverse_mpm.json`) | same |

- The force normalisation matches PhysCtrl's dataset loader (`traj_dataset.py:226–235`).
- With base_drag_coeff, the total force is coeff × the object's weight, so **force 0.40 means a push equal to 40% of the object's weight, held for the whole clip, with no gravity and no floor**.
- An object pushed this way accelerates without limit and leaves the frame. That holds in both PhysCtrl's videos and the reference, and it is how PhysCtrl's training data behaves.
- **Discrepancy noted:** PhysCtrl's paper (Appendix A) gives the drag range as 0.02G–0.3G, while its code samples 0.02–0.2. Our 0.40 is outside both.

### 1.2 Units: the objects are about 1 m tall and, at log E = 4, extremely soft relative to their weight

- The simulator uses g = 9.8, density 1000 and objects normalised to size 1, so one unit is about one metre.
- What sets how much an object sags or stretches under forces proportional to its weight is roughly E/(ρ·g·L):
  - log E = 4 at L = 1 m gives about 1;
  - a 10 cm rubber toy (E ≈ 10⁷–10⁸ Pa) gives about 10⁴–10⁵.
- The rubber value comes from the research agent's sources (Engineering ToolBox; Wikipedia table, a weak citation).
- **Interpretation (order-of-magnitude argument, a hypothesis to keep in mind):** PhysCtrl's softest training materials behave like a block of soft jelly, not like the rubber or plush toys in our images.

### 1.3 The cases the rater most likely meant

Contact sheets of all 28 items are in `notes/simulator_audit_2026-10-22/sheet_*.png`. Each shows video frames 0–24, the half of the clip in which objects are still in view. Trajectory measurements are at model frame 11:

| Group | Items | What the reference shows | What PhysCtrl shows | Measured |
|---|---|---|---|---|
| **Stretching (likely "weird")** | foam sponge, rubber chicken, star stress ball, rubber snake, rubber dog chew (all log E = 4); jelly dessert (log E = 5) | the pushed patch (head or top) is pulled into a long strand while the body stays put | moves the whole object, nearly rigid | reference stretch 2.4–3.0× vs 1.2–1.8× for PhysCtrl; 13–58% of reference particles moved < 0.2 object sizes (jelly 6%, star 0%) |
| **Renderer artifacts in the reference video (likely "head on the floor", "odd")** | stuffed bunny: a second bunny head appears lying on the table; plush mushroom: flips and appears to land upright on the table (`bunny_mushroom.png`) | — | — | the trajectory has 0% of particles left behind and no floor nearby, so these come from the video model (DAS), not the simulation |
| **Tipping and rotating while lifted (likely "changed direction")** | rubber boot, plush cactus, gummy bear, plush dog, cat, frog; giraffe ×3 (the neck bends down, in ours and the reference) | rotation from an off-centre push on the top patch | similar | no meaningful difference: displacement percentiles agree within about 0.05 |
| **Near-identical** | teddy bear, dinosaur, spatula, octopus, toy car, letter A, elephant, fish, pillow, mushroom (trajectory) | — | — | displacement p5/p50/p95 within about 0.05 |

- The rater's comments are not tied to item ids, so this mapping is a judgement from the frames.
- What the stretching items share is a very soft material with the force on a small patch: 4–8% of the particles. The toy car and the elephant are equally soft (log E = 4) but have patches of 26% and 18%, and they do not stretch.

### 1.4 The error metric is dominated by frames the viewer does not see

- On the 23 Phase 11 shifted items, 83% (normal items) and 85% (stretching items) of PhysCtrl's mean point distance to the reference builds up in model frames 12–23.
- By then, the contact sheets show, the objects have mostly left the view.
- Items that look almost the same in view still score above the 0.10 risk threshold. Examples: teddy bear 0.196, boot 0.333, plush dog 0.332.

### 1.5 Does this affect the Phase 9 results? Only one family substantially

Reference stretch (extent at the last frame / first frame) on all 600 single-object Phase 9 queries, from the job-101 trajectories:

| Family | Share of queries with reference stretch > 2× | PhysCtrl wrong (> 0.10): stretch > 2× vs ≤ 2× |
|---|---|---|
| In distribution | 2% | 1.00 vs 0.11 |
| Stiffer (ood3_E_high) | 1% | 1.00 vs 0.05 |
| **Softer (ood3_E_low, log E 3.0–3.9)** | **32%** | 1.00 vs 0.53 |
| Drop under gravity | 0% | — vs 0.98 |
| Stronger push (ood3_force) | 8% | 1.00 vs 1.00 |

PhysCtrl's own outputs barely stretch: the median ratio is 0.95–1.15 in every family.

---

## 2. Hypotheses (not verified)

- **H-a. The strands may be partly numerical.**
  - Phase 10a found the reference not converged for soft materials and strong pushes, exactly this regime.
  - A strand a few grid cells thick (dx = 0.08) is where MPM is least accurate.
  - Not tested; see recommendation R2.
- **H-b. The rater's preference partly reflects a stiffness heuristic.**
  - Perception research finds people judge stiffness by how much an object deforms, not by its material (Paulun et al., JoV 2017).
  - A rubber toy that stretches like jelly looks wrong even if the simulation is right for jelly.
- **H-c. Renderer artifacts appear more often on the reference than on PhysCtrl.** Large, unusual motions (strands, fast flips) may push the video model to hallucinate. We have two clear examples, both in the reference; not quantified.

---

## 3. Arguments, both ways

### For the hypothesis ("our reference and metric are not what we want")

1. The reference's conventions do not describe the pictured scene: no gravity while pushed, a constant push for the whole clip, 1 m objects, jelly-soft at log E = 4. "Lower error against it" does not mean "more realistic video" there.
2. In the 6 stretching items, PhysCtrl's motion is closer to what a real toy would do, and the rater saw it.
3. The metric counts motion after the object has left the view, and calls visually near-identical motions failures (teddy 0.196).
4. The reference is not converged in the soft and strong-push regime (10a, 10c), so absolute risk there is partly simulator noise.
5. The literature supports keeping the two apart:
   - Physics-IQ (arXiv:2501.09038) finds visual realism "unrelated" to physical understanding.
   - PhysCtrl's own paper reports its generator judged at least on a par with a simulator, and preferred by 81% of 35 users on plausibility.

### Against the hypothesis ("the simulator and metric are fine for what we claim")

1. **No bug:** every setting matches PhysCtrl's data generator, and drop free fall is exact (10a R3). Our claim is fidelity to the simulator PhysCtrl was trained to imitate, and that claim is measured correctly.
2. **17 of 23 items** move almost identically in both videos. The rater's "easy" items are mostly explained by the soft-and-concentrated regime and by renderer artifacts, not by a general simulator flaw.
3. **Human plausibility is not physical correctness:**
   - Intuitive physics is biased (McCloskey 1980; Sanborn 2013).
   - Graphics treats "plausible" as a separate target from "correct" (Barzel 1996; O'Sullivan 2003).
   - In the strand items, the simulator gives the consistent answer for the material as specified; PhysCtrl ignores the material.
4. **Renderer artifacts are not in the trajectories**, which is what we measure and certify; they would affect any trajectory sent to that renderer.
5. **The main Phase 9 claims hold outside the problem regime:**
   - drop (realistic gravity physics, R3 exact) has PhysCtrl wrong in 98% of queries;
   - softer queries that do not stretch still have PhysCtrl wrong in 53%;
   - 10c T1 (PhysCtrl worse than ours against every reference) passed.
6. **Statistics:** one rater; the test needs at least 10. On Question 2 the rater picked ours 28/28 and all 5 repaired items, so they can see fidelity; they judged plausibility differently.

### Weighing

- The simulator is not wrong; it is PhysCtrl's own.
- **"Lower error" means "more faithful to PhysCtrl's physics convention", which is realistic only when the material and force are realistic.** For log E = 4 with a concentrated push, and for force 0.40 generally, it is not.
- Our Phase 11 settings put 6 of 23 shifted items in that regime. The ood3_E_low family puts 32% of its queries there.
- The paper's fidelity claims stand. Any wording that implies more realistic videos does not, for those regimes.
- The video-level human study measures realism as perceived through the renderer, so it may well fail V2. That should be reported as a fidelity-versus-realism finding, not hidden.

---

## 4. Impact on existing results

| Result | Change? |
|---|---|
| Phases 0–9 code and data | none; no bug |
| Phase 9 / 10c, in distribution, stiffer, drop, contact | none |
| Phase 9 / 10c, softer family | about a third of its queries are in a stretching regime that is unrealistic for everyday objects and not converged. The PhysCtrl-failure claim there partly reflects that regime (53% wrong outside it, 100% inside). |
| Phase 9 / 10c, stronger-push family | 8% stretching. Already flagged as time-step sensitive in 10a; unchanged. |
| Phase 11 judge (V0 fail) | unchanged; consistent with the literature on VLM physics judges |
| Phase 11 human study | still valid as pre-registered, but Q1 is "plausibility through the renderer". On 6 items the reference is unrealistic, and 2 reference videos contain renderer artifacts. Expected to favour PhysCtrl on Q1; the interpretation must say so. |
| Paper wording | (proposal; no edit made) the reference's conventions should be stated explicitly, together with the fidelity-versus-realism distinction |

---

## 5. Recommendations, ranked

1. **R1 (writing, 0 GPU-h; needs your go).**
   - In the paper, state the reference's conventions: no gravity during a push, constant push, 1 m-scale objects, E 10⁴–10⁷.
   - Call our metric a fidelity metric.
   - Rest realism-flavoured claims on drop and contact; describe softer and stronger-push as fidelity shifts.
   - Add the renderer-artifact observation to the limitations.
2. **R2 (GPU, about 1–2 GPU-h; pre-register first). Are the strands numerical?**
   - Re-simulate the 6 stretching items at dt/2, dt/4 and grid 250.
   - If the strands shrink substantially, part of the softer-family failures is numerical.
3. **R3 (CPU, 0 GPU-h; pre-register before computing). Realistic-regime subset analysis.**
   - Define, from inputs only, a criterion for the stretching regime: estimated patch strain ≈ coeff·ρ·g·L / (mask fraction · E), threshold fixed on development data.
   - Report the Phase 9 / 10c tables on the queries inside and outside it.
   - The threshold must not be tuned on the Phase 9 outcomes.
4. **R4 (CPU, 0 GPU-h; pre-register). A renderer-free fidelity check.** A two-choice study on point-cloud videos (PhysCtrl vs ours, with the reference shown), which takes the video model's artifacts out of the comparison.
5. **R5 (GPU, about 5–10 GPU-h; your decision). A realistic-settings video study.**
   - log E 6–7, forces within 0.02–0.2, all fixed before rendering.
   - Expectation: PhysCtrl fails less often and the visible gain shrinks. It would show where certification matters for realistic content.
6. **Keep the running study unchanged** and analyse it once at close, as pre-registered.

## Sources (opened by the research agent; see its report for quotes)

- **Perception of physics:**
  - McCloskey et al., *Science* 1980, doi:10.1126/science.210.4474.1139;
  - Kaiser et al., *JEP:HPP* 1992;
  - Sanborn et al., *Psych. Rev.* 2013, doi:10.1037/a0031912;
  - Battaglia et al., *PNAS* 2013;
  - Barzel et al., *CAS* 1996;
  - O'Sullivan et al., *ACM TOG* 2003;
  - Reitsma & Pollard, *ACM TOG* 2003;
  - Paulun et al., *JoV* 2017, doi:10.1167/17.1.20.
- **Video physics benchmarks and judges:** Physics-IQ arXiv:2501.09038; VideoPhy arXiv:2406.03520; VideoPhy-2 arXiv:2503.06800; PhysBench arXiv:2501.16411; WorldModelBench arXiv:2502.20694; TRAVL arXiv:2510.07550.
- **Physics-grounded generation:** PhysCtrl arXiv:2509.20358 (training ranges E ∈ [10⁴, 10⁷], drag 0.02G–0.3G; user study 81%); PhysGaussian arXiv:2311.12198; PhysDreamer arXiv:2404.13026; DreamPhysics arXiv:2406.01476.
- **Material values:** rubber 10⁷–10⁸ Pa (Engineering ToolBox; Wikipedia table, weak); Ecoflex 00-30 100% modulus ≈ 69 kPa (Smooth-On).

## Correction (Oct 22, later): the drop family cannot be expressed by PhysCtrl's base model
- PhysCtrl base (`src/configs/eval_base.yaml`) has `floor_cond: false` and no `gravity_emb`. Only the large model (`eval_large.yaml`: `gravity_emb: true`, `floor_cond: true`) takes gravity and a floor as input (`src/model/spacetime.py:972–1047`).
- In our drop queries the base generator therefore receives no force and no gravity. Its 98% failure on ood3_drop is structural (an input it cannot represent), not a failure to generalise.
- What still holds: the certificate catches these silent failures, while PhysCtrl's own seed spread accepts 99% of failed drops (4m).
- What changes: "PhysCtrl fails under shift" must not lean on drop. Recommendation R1 above ("rest realism claims on drop") is withdrawn. Drop is evidence for the certificate, not for a generalisation failure. Phase 2 already recorded large at 0.120 vs base 0.401 on drop (PLAN l. 204).

## Update (Oct 22): R2 done (Phase 10d, job 431955)
- Pre-registered verdict PARTLY NUMERICAL. Stretch is resolution-stable (median 2.95 at the reference and at the finest body-preserving setting); only the left-behind share is time-step sensitive (Δ 0.113 at dt/4). So hypothesis H-a (strands are numerical) is not supported. The strands are what the specified material and push produce in this MPM; the problem is the parameters.

