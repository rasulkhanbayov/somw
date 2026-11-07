# Item 3: multi-object contact benchmark, design memo (Oct 14, 2026)

Scoping only; nothing here is pre-registered yet. Written by a planning agent from the code and PLAN, then checked by Claude. Needs the user's approval on the points under "Scope" before any pre-registration.

## Construction
- **Components:** fresh single-object, in-distribution, drag-only queries `mo_comp_0..7` (100 test objects × 8 draws). Each scene shares one (E, ν); gravity stays off; floor 0.2.
- **Scenes (CPU, from component ground truth):** pair the objects within each draw into 40 contact pairs + 10 no-contact controls, giving 400 scenes (320 contact + 80 controls).
  - Object B is placed along A's displacement. Translations are integer multiples of dx = 0.08, so particle volumes and isolated dynamics are unchanged.
  - The gap is set so that the superposed isolated ground truths first come within 2·dx at a target model frame k* ~ U{5..18}.
  - Rejection rules are fixed in advance (gap < 2·dx; within 0.3 of a wall).
  - Measured on id_new_12: objects ≈ 2.0 units across; median centroid travel 2.4 units in 48 frames.
- **Ground truth:** one joint reference rollout per scene: 4,096 particles in one 125³ grid, two drag impulses with per-object masks.
- **"Generated" trajectory:** each object generated separately by base, then translated and concatenated. This is the realistic failure, since neither generation knows about the other object.
- **Second source:** the superposed isolated ground truths ("gt_super"), a perfect single-object generator whose only error is the ignored interaction.

## Scope: decisions for the user
1. Multi-object scenes are item 1 on PLAN's cut list; un-cutting them is the user's call.
2. One MPM grid gives sticky, no-slip contact that starts about 1–1.5 dx before the surfaces touch. This is valid under the same-simulator rule, but the paper must say so.
3. The evaluation tests a per-object composition pipeline, not PhysCtrl itself.
4. Contribution 3 does not benefit: simulation cost tracks substeps, not particles, and generation cost doubles.
5. Different materials per object would touch the guarded bridge; defer.

## Pre-registerable tests (cluster bootstrap over object pairs)
- **New placements:** `contact` (a window at the generated objects' first approach, kc − 1) and `contact_spread` (that window plus two spread windows).
- **H0 gate:** the lift floor on contact windows must be ≤ 0.25 × the base defect; otherwise the analysis switches to the reference m = 4 scan (pre-registered switch).
- **H1 placement:** capture(`contact_spread`) − capture(`uniform`) and − capture(`spread_event`), both with a 95% lower bound > 0, on base and on gt_super.
- **H2 certificate:** fixed τ = 0.02614; false-accept rate among bad scenes ≤ 0.15; `contact_spread` lower than `uniform`.
- **H3 repair:** event-gated V2 (restart at max(kc − 1, 0)) against cert_ref and against 4k gated V2. Requirements: risk ≤ cert_ref + 0.02; cost saving > 0; event timing better than chain timing and than shuffled timing.

## Code changes
- **New:** `p4l_mo_design.py`, `p4l_mo_build.py` (joint ground truth, scene HDF5 with object_id/translation), `p4l_mo_compose.py`.
- **Verifier:** block-diagonal kNN in `nbrs_for` (neighbours must not cross objects); `gen_contact_frame`; the new placements; per-object defects.
- **Existing scripts:** extra restart points in `p4j_allk`; `p4l_summarize.py` reuses the `p4k_gated` code and charges generation at 2×.
- **Housekeeping:** synthetic two-object backend, self-tests, slurm 58–61.

## GPU estimate (planning agent's estimate)
About 70 A100 s per scene (≈ 150 s on H200), i.e. ≈ 17 GPU-h for 400 scenes. With components, generation and a pilot: **≈ 21 GPU-h core, ≤ 33 with options (large model)**. Dropping the reference scan saves about 9 h.

## Timeline (proposed)
- Oct 15: code, self-test, dry run.
- Oct 16: components and pilot. The pilot freezes validity rules only.
- Oct 17–18: 400 scenes, scan and all-k rollouts.
- Oct 19: verdict.
- Oct 20: buffer.
