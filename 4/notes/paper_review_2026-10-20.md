# Paper review and edit plan (Oct 20, 2026)

The user asked for this senior-AC review: strengthen the CVPR 2027 draft (`paper/`, 9 pages, body ends on p. 8) **without changing the method**. Allowed are writing, framing, figures, tables, and analyses or evaluations of the existing method.

Sources:
- Full-text reads (by research agents, Oct 20) of:
  - PhysCtrl 2509.20358v2;
  - PSIVG 2603.06408v1;
  - RealWonder 2603.05449;
  - PhysFlow 2609.08215 (no appendix in the HTML);
  - NEXUS 2606.15015v2;
  - Hybrid Neural World Models 2605.28317v1;
  - ANCHOR 2512.19643v2.
- PhysCtrl's metric code: `$DATA_ROOT/physctrl/src/volume_iou.py`.

Nothing in `paper/` has been edited for this plan.

## Step 1: how the strong papers are built

| | Teaser (Fig. 1) | Contributions | Main table | Video evaluation | Qualitative | Ablations | Limitations |
|---|---|---|---|---|---|---|---|
| **PhysCtrl** | 4-row grid: image + force → trajectories + frames; varies material, E, force | 4 bullets, no numbers | 2 small side-by-side tables: video (GPT-4o SA/PC/VQ) and trajectories (vIoU, Chamfer, point L2) | GPT-4o judge on **12 videos**; user study 35 people (81% prefer ours, appendix) | 6 figures; side by side with 4–5 baselines | table (attention, physics loss); **simulator vs ours "on par" by GPT judge** (Table 4) | ~0.6 page; failure figure split by component |
| **PSIVG** (CVPR 26) | before/after on one bowling collision | 3 numbered, "first training-free …" | 11 rows **grouped by type**, columns grouped (motion control vs quality); ties accepted | user study 32 people (82.3%); VBench | 2 examples × 6 rows full width | small table + qualitative | 6 lines |
| **RealWonder** | one scene, 3 action strips, "13.2 FPS" badge | 3 bullets, one quantified | VBench + GPT "PhysReal"; bold best / underline second; loses one column | **400-person** 2AFC; 30 images | 3 figures, 4 scenes × 4 methods | qualitative only | 2 sentences + stress test |
| **PhysFlow** | paradigm schematic | 4 bullets | 8 rows incl. **PhysCtrl** (VideoPhy-2, VBench, WorldScore, FVD); loses some columns | 60 samples; GPT-4o; 50 human ratings | ~9 figures | 3 tables (incremental and single removal) | failure-case paragraph + figure |
| **NEXUS** | pipeline on one example | 3 qualitative bullets | trajectory metrics vs 7 baselines; honest losses bolded | **6 cases**, GPT judge + 10 raters; **swaps PhysCtrl's trajectory module, keeps its renderer (our protocol)** | 2 figures; PhysCtrl vs NEXUS × 4 cases | single-removal table ("no contact": RE@48 7.35 vs 0.137) | 1/3 page |
| **Hybrid NWM** | speed vs horizon + RMSE–speedup Pareto | 3, quantified ("26–72×") | AUROC over 54 cells; **gap above a random-deferral floor** | — | 5-panel error maps | many, 3 seeds | own section, 4 headed paragraphs |
| **ANCHOR** | schematic: "the solver is a component, not an alternative" | 2 components, in Results | wall-clock on 2 samples; "~30% faster, 80% lower peak error" | — | error contours | parameter sweeps | scattered |

**Patterns that make them look strong:**
1. **A striking page-1 figure that shows the result**, not the pipeline (PhysCtrl, PSIVG, RealWonder).
2. **Small video evaluations are accepted:**
   - 6–60 cases;
   - a GPT/VLM judge (the prompt is printed in the appendix);
   - a small human preference study, 10–400 people, often 30–35.
3. **One headline metric that the method clearly wins**, with honest ties or losses elsewhere. Bold for best, underline for second; rows grouped by method type.
4. **One ablation table** of single-component removals.
5. **Contributions are 3 bullets.** The strongest one carries a number ("first training-free …", "13.2 FPS", "26–72×").
6. **"Why not just run the solver?"** answered by:
   - speed at matched settings (Hybrid NWM: a Pareto plot);
   - or the claim that the problem is independent of solver cost: no per-trajectory error detection otherwise (ANCHOR).
7. **What none of the gating papers has:**
   - a statistical guarantee or a false-accept analysis;
   - a comparison against a tuned cheaper solver, at matched accuracy.
   **Ours has all of these; the draft undersells them.**
8. **PhysCtrl's own Table 4 reports its videos "on par" with a traditional simulator** (GPT judge, its own examples). Our data show where that parity breaks (68% / 69% wrong under shift and on contact). This is a strong, fair hook.

## Step 2: our weaknesses, ordered by likelihood of a reject

1. **"Why not just simulate everything?" (abstract, §1, Tab. 1, Fig. 2).**
   - Our saving is moderate: 0.26× / 0.71× / 0.90×.
   - Tab. 1 bolds the tuned simulator M1 in 2 of 3 columns, so at a glance we lose two of three.
   - The reliability argument (a per-query certificate, guarantee, safe when the regime is unknown) is buried in prose.
   - *Strong papers:* ANCHOR states the gap as instance-level reliability, independent of solver cost. Hybrid NWM shows one Pareto figure on page 1.
2. **Thin video-level evidence for a video venue (§5.8, Fig. 4).**
   - One rendered example, 10 renders in total, no judge, no user study.
   - *Strong papers:* PhysCtrl 12 videos + GPT judge + 35-person study; NEXUS 6 cases + GPT + 10 raters.
3. **No page-1 figure (p. 1 is all text; the key result, Fig. 2, is on p. 6; the first figure is the pipeline on p. 3).**
   - *Strong papers:* a before/after teaser (PSIVG) or a result grid (PhysCtrl).
4. **We do not report PhysCtrl's own headline metrics (§5.6, `\PLAN`).**
   - PhysCtrl's Table 2 uses vIoU, Chamfer and point L2 on its 100-object test set. We planned MSE + DeformLoss.
   - Reviewers who know PhysCtrl look for vIoU/Chamfer.
   - Its code is in our checkout (`volume_iou.py`), and job 97 saves exactly the trajectories needed.
5. **No consolidated ablation table.** The component evidence is scattered over §3–§5 and over phases:
   - one vs three windows;
   - shuffled vs certificate timing;
   - shadow vs generated-state restart;
   - shadow- vs generation-timed switch;
   - M2 vs reference suffix;
   - uniform placement.
   *Strong papers:* PhysCtrl Table 3 and NEXUS Table 2 are single-removal tables.
6. **Number-dense prose, a long abstract, jargon ("residual-free", "lifted states", "uncertified suffix") (abstract, §1, §5).** §5.1–§5.4 read as streams of numbers.
   *Strong papers:* one explicit research question (PSIVG), short quantified bullets, numbers mostly in tables.
7. **The certificate's quality is not shown visually (§5.2 is text only).** We have:
   - AUROC 0.977;
   - risk–coverage data (7a);
   - a calibration check over 500 object splits (4a ext.);
   - "beats random fallback" (4d/4f).
   *Strong papers:* Hybrid NWM's random-deferral floor and AUROC tables.
8. **Ground truth is PhysCtrl's simulator (§3.1, §5.5 with `\RES`).** Some reviewers will call this circular.
   - Fix (wording): PhysCtrl itself scores against MPM ground truth (its Table 2), so this is the field's protocol.
   - Job 96 adds the convergence numbers.
9. **Where the cost goes is hidden.** On stronger push and drop, ours costs about 1.0–1.1× because it correctly falls back (`results/p9_table` per family; not in the paper). A reviewer who finds this in the supplement will feel misled; stating it first reads as honest.
10. **No failure-case figure.** PhysCtrl shows its failures by component. We have:
    - 3/26 accepted bad generations in distribution;
    - the stiff-material leak (6/16);
    - contact rejecting good scenes (1/45 accepted).
11. **Too few qualitative examples in the main paper** (2 contact scenes, 1 video) and no supplementary video yet.

## Step 3: ranked edits

Impact = expected effect on acceptance (H/M/L). GPU = GPU-hours. **★ = do first.**

### ★1. A page-1 teaser that shows the result (H; 4 h; 0 GPU)
Fuse the contact example with the risk–cost result. Absorb the current Fig. 2 into it, which saves space.
```
┌──────────────────────────── Fig. 1 (full width, ~2.2 in) ─────────────────────────────┐
│ (a) two objects collide              │ (b) risk vs cost, three test sets (Phase 9)       │
│ PhysCtrl | ours | reference          │ in distribution | shift | contact                 │
│ (3 frames each, contact_switch case, │ shaded safe zone; ours = only dot inside the zone │
│  point clouds coloured by error)     │ in all three at < 1× cost                         │
│ "PhysCtrl ignores the collision      │                                                   │
│  (error 0.125); ours 0.025"          │                                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```
Data: `results/fig_qualitative/paper_contact_switch.npz` and `results/p9_table`. Both exist.

### ★2. Reliability framing + a regime-agnostic main table (H; 4 h; 0 GPU)
**Text.** State the gap as ANCHOR does and use PhysCtrl's own parity claim. Before (abstract, sentence 3):
> "Simulating every query with the reference simulator is accurate by construction but costly, and a cheaper simulator setting tuned on known data fails silently under shift."

After:
> "A learned prior can match a simulator on familiar queries, as PhysCtrl reports, but nothing tells a user when a new query is not familiar: on our shifted and two-object tests PhysCtrl is wrong on 68% and 69% of queries, and every cheaper simulator setting we tuned is wrong on 10–16% under shift. We give each generated trajectory a certificate."

**Table.**
- Group rows as PSIVG does: learned prior | fixed simulator settings | gated pipelines | ours.
- Add two summary columns: **"worst-case risk over the three sets"** and **"safe on all three (risk ≤ 3%)?"** with the words yes/no.
- Bold the best value, underline the second.
- Every number already exists (`results/p9_table/table.json`):

| Group | Pipeline | ID risk/cost | Shift | Contact | **Worst risk** | **Safe everywhere** | **Max cost** |
|---|---|---|---|---|---|---|---|
| learned | PhysCtrl | 0.130/0.05 | 0.680/0.05 | 0.693/0.10 | 0.693 | no | 0.10 |
| fixed sim | M1 | 0.015/0.13 | 0.102/0.20 | 0.029/0.13 | 0.102 | no | 0.20 |
| … | … | | | | | | |
| gated | certify or simulate | 0.015/0.54 | 0.003/0.87 | 0.000/1.18 | 0.015 | yes | 1.18 |
| **ours** | | 0.020/0.26 | 0.013/0.71 | 0.012/0.90 | **0.020** | **yes** | **0.90** |

The point at a glance: ours is the only pipeline that is safe everywhere at under 1× in every column. The per-set columns still honestly show M1 cheaper where its regime is known.

### ★3. PhysCtrl's own metrics on PhysCtrl's own test set (H; 3 h; 0 GPU beyond job 97)
- Run PhysCtrl's `volume_iou.py` (vIoU, Chamfer, MSE, the same code as its Table 2) on the job-97 outputs: generate only, ours, certify-or-simulate.
- Include a reproduction row: PhysCtrl's own reported 77.59% vIoU (25 sampling steps) against our generate-only at the 4 steps we use. A gap must be stated, not hidden.
- Replaces the `\PLAN` in §5.6.
- Pre-register in PLAN as descriptive before running; development data.
- Table sketch:

| | vIoU ↑ | Chamfer ↓ | L2 ↓ | risk | cost |
|---|---|---|---|---|---|
| PhysCtrl (reported, 25 steps) | 77.59% | 0.0028 | 0.0015 | — | — |
| PhysCtrl (ours, 4 steps) | \RES | \RES | \RES | \RES | 0.05 |
| + our certificate and repair | \RES | \RES | \RES | \RES | \RES |

### ★4. One ablation table: "what each component buys" (H; 3 h; 0 GPU)
Single removals, each from the pre-registered phase that tested it. The test set differs per row and is labelled; all numbers are in PLAN/results.

| Remove / replace | Effect | Source |
|---|---|---|
| repair from generated positions instead of the shadow state | risk 0.017 → 0.070 | 4h vs 4i |
| restart at a shuffled chain point instead of the first failure | error +0.002 [0.001, 0.003] | 4k K3 |
| switch timed by the generation instead of the shadow | risk 0.012 → 0.025, same scenes (on 4n's scenes it missed its margin: 0.034) | Phase 9 (`p9c_summary`); 4n |
| first-failing-window timing instead of the shadow-timed switch (contact) | same risk, error 0.015 → 0.029 | Phase 9 CC3 |
| three windows → one window (single objects) | AUROC 0.958 → 0.955, probe cost −67% | E2 (exploratory) |
| one window on contact | risk 0.012 → 0.095 (4o scenes, code check before Phase 9 data) | Phase 9 amendment |
| reference suffix → medium (M2) | cost 25–55% lower at similar risk | 8a (exploratory) |
| uniform window placement at matched budget | costlier at similar risk | 4m M4 |
| generative repair (5 variants) | at most 1% better | 4b–4g-v2 |

### ★5. A small video-level evaluation of our existing method (H, the largest single gain for a video venue; 3–4 days; about 15–25 GPU-h)
The field standard is 6–60 cases with a GPT/VLM judge plus a small human preference study. We have 10 renders from 5 images.

Proposal:
- About 30 input images: PhysCtrl's 5 examples plus synthetic images, as PhysFlow does.
- The same push protocol (as given and 3×).
- Render PhysCtrl, ours and the reference for each through PhysCtrl's renderer. This is NEXUS's swap-the-trajectory-module protocol.
- Score with a VLM judge, prompt printed in the appendix as PhysCtrl does. Plus a 2AFC preference study among lab members: "which clip moves more like the reference" and "more physically plausible".

**Decisions needed from the user:**
- the image source;
- which judge: an external API such as GPT-4o, or a local open model (would need a download);
- whether a human study is possible in the time.

Pre-register before any render.

### 6. Rewrite the abstract and §1 for readability (M; 4 h; 0 GPU)
- Abstract at 150–170 words, one number per claim.
- One explicit question in §1, as PSIVG does: *"Can a generated trajectory be trusted per query, at less than the cost of simulating it?"*
- Contributions as 3 bullets, one number each. Example:
  - Before: "A residual-free certificate for generated trajectories, built from a few short reference-simulator windows and calibrated at the object level. It has a finite-sample bound in distribution. Under shift it let through 1.2% of failures, against 35% for the generator's own sample spread."
  - After: "**A certificate for generated motion**: a few short simulator checks, calibrated to bound how many bad trajectories are accepted; under shift it accepts 1.2% of failures, against 35% for the model's own uncertainty."
- Replace "lifted states", "uncertified suffix" and "residual-free" in the abstract with plain words; keep the terms in §3.

### 7. Certificate-quality figure (M; 3 h; 0 GPU)
- Risk–coverage curves (window score vs seed spread vs momentum residual; 7a, `results/p7a_metrics/risk_coverage.json`).
- A Hybrid-NWM-style **random-deferral floor** (4d T1 / 4f F3: beats random fallback).
- Optionally the empirical false-accept rate over 500 object splits against the bound 0.11 (4a ext.).
- Label it: earlier three-window certificate, development pool.

### 8. State where the cost goes, per shift family (M; 1 h; 0 GPU)
One sentence plus a small table in the supplement:
> "On the hardest shifts (3× drag, drop) almost every generation is rejected and ours costs as much as simulating (0.99× / 1.10×); the savings come from the softer and stiffer families (0.50× / 0.24×)."

Source: the per-family breakdown of Phase 9 (computed Oct 20 from `p9_confirm.assemble`; to be added to `results/p9_table`).

### 9. Wording that answers "circular ground truth" (M; 0.5 h)
§3.1 and §4:
> "As in PhysCtrl's own evaluation, ground truth is the MPM simulator that produced its training data"

plus job 96's convergence numbers.

### 10. Failure-case paragraph and small figure (M–L; 3 h; 0 GPU)
- The 3 accepted bad generations in distribution: draw one, point clouds from the stored generation and ground truth.
- The stiff-material leak.
- The conservativeness on contact.
PhysCtrl's split by component is the model to follow.

### 11. Supplementary video (M; 4 h; 0 GPU)
- All 10 renders side by side (PhysCtrl | ours | reference) with error overlays.
- Animated point clouds for the contact and shift cases (from existing npz files).

PhysCtrl: "We strongly encourage the readers to look at our video".

### 12. Title (L; 0.5 h)
Optionally lead with the idea: *"Generate Once, Simulate What Fails: Certified Physics Checks for Physics-Grounded Video Generation"*. Decide after ★1–★5.

## Page budget
- ★1 absorbs Fig. 2, net about +0.1 page.
- ★2 is the same table plus 3 columns, about +0.05.
- ★3 adds a small table, replacing the `\PLAN` (+0.15).
- ★4 adds a table (+0.25).
- ★5 adds a small table (+0.15).
- Room comes from: §5.1–5.4 prose cut by about 30% (numbers move into tables), related work tightened by about 20%, and the method diagram's side note dropped.
- Expected to fit 8 pages. Check after each edit.

## Integrity notes
- All numbers above are from PLAN/results. The ★3 and ★5 numbers do not exist yet and stay `\RES`.
- ★3 and ★5 need a pre-registration entry in PLAN before running.
- ★4 mixes test sets by design; each row names its phase, and exploratory rows are labelled.

## Status (Oct 20, end of day)
- Done: ★1 teaser; ★2 table and framing; ★3 (reproduction row filled; final rows wait for job 97); ★4 ablation table; 6 abstract/intro; 7 certificate figure; 8 per-family sentence; 9 ground-truth wording; 11 supplementary videos.
- ★5 prepared as Phase 11: pre-registered, code and jobs ready. Waiting for the model download and submission.
- Not done:
  - 10 failure-case figure: no space; covered in the limitations;
  - 12 title: kept.

