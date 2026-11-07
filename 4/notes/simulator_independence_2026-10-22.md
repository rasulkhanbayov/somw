# Is our advantage circular? Simulator independence check (Oct 22, 2026)

**The user's concern.** Our pipeline is built from the MPM simulator we score against: its windows, shadow and repair suffixes all run PhysCtrl's Warp MPM, and so does the reference. A low error for ours may mean "we agree with the reference because we are the reference", not "our motion is physically better".

**Short answer.**
- Yes, most of ours' advantage over PhysCtrl is circular by construction (§1).
- The claims that do not depend on the reference being physically true survive: the certificate, the cost accounting, and PhysCtrl's silent failure against its own target.
- "Ours is more accurate than PhysCtrl" in the physical sense is untested until an independent simulator agrees.
- No other MPM code is independent of PhysCtrl's. The best independent reference is PolyFEM, a finite-element solver with the same material model. Its packaged version is unusable, so it is being built from source.
- A cross-check is pre-registered in PLAN (Phase 12) with fixed tests and a fallback.

Everything here is exploratory, computed after Phase 9/10c/11 results existed. No paper text was changed.

---

## 1. Circularity in numbers (Phase 9 test sets; 10c trajectories; query counts)

| Test set | Ours accepted PhysCtrl | Ours repaired | Ours = reference (fallback) |
|---|---|---|---|
| In distribution (200) | 108 (54%): PhysCtrl wrong 3, ours wrong 3 | 79 (40%): PhysCtrl wrong 11 → ours wrong 1; median error 0.066 → 0.018 | 13 (7%): PhysCtrl wrong 12 → ours 0 (error 0 by construction) |
| Shift (400) | 85 (21%): 1 → 1 | 107 (27%): 65 → 4; 0.128 → 0.019 | **208 (52%): 206 → 0** |
| Contact (132) | 1 (1%): 0 → 0 | 56 (42%): 33 → 1; 0.127 → 0.026 | 75 (57%): 54 → 0 |

Repairs re-simulate on average 61% (in distribution), 75% (shift) and 63% (contact) of the 24 frames. They use the same MPM code at cheaper settings: an M1 shadow with an M2 suffix for single objects, and a coarse shadow with an M3 suffix on contact.

**What "ours beats PhysCtrl" means in each case:**

- **Accepted:** ours is PhysCtrl, so there is no advantage, and none is claimed.
- **Fallback (most of the shift and contact gain: 206 of 267 and 54 of 86 fixed failures):** ours is the reference. "Ours beats PhysCtrl" is then "the simulator agrees with itself; PhysCtrl does not". Against any other truth, ours scores exactly the MPM's own error.
- **Repair:** a certified generated prefix plus an MPM suffix at different settings. This is partly independent of the reference (different time step and grid), but the same code and the same physics.

**Claims that survive circularity:**

1. **The certificate detects when a generation disagrees with the simulator PhysCtrl was trained to imitate.** This is a statement about consistency with the intended target. It holds whether or not that target is physically true. Examples: it accepted 1 of 272 bad generations under shift, and seed spread is far worse.
2. **Cost accounting:** ours reaches the reference's answer at 0.26–0.90× the reference's cost.
3. **PhysCtrl fails silently against its own training simulator** (Phase 9; 10c T1 against three time steps). This is a statement about PhysCtrl.

**Claims that do not survive without an independent reference:**

4. "Ours is more physically accurate than PhysCtrl", in any sense beyond "closer to PhysCtrl's MPM".
5. "Tuned M1–M3 are less safe than ours". This compares discretisations of the same code family. Phase 10c already showed the reference itself moves under time-step changes.

---

## 2. Independent simulators (survey by a research agent; sources opened; key facts re-checked here)

| Candidate | Independent of PhysCtrl's MPM? | Material model | Notes |
|---|---|---|---|
| taichi_elements MPM | **No.** It is the ancestor: PhysCtrl's surface collider is a line-by-line port of it, down to the comments and error strings | fixed-corotated | catches nothing new |
| PhysGaussian / warp-mpm | **No:** the same lineage | fixed-corotated | — |
| Genesis MPM | separate code, same method and the same fixed-corotated formula | fixed-corotated | catches code bugs, not MPM artefacts |
| **PolyFEM** (C++ FEM, MIT) | **Yes** | **fixed-corotated** (added June 2024), neo-Hookean and others | implicit Newmark/BDF, Newton, IPC contact; CPU |
| libuipc (GPU IPC, Apache-2.0) | Yes | stable neo-Hookean (no fixed-corotated) | implicit; per-vertex external force = drag box; ground plane; `pip install pyuipc` |
| Genesis FEM | yes | linear / stable neo-Hookean / linear corotated | default implicit solve runs one Newton iteration, which is not converged |
| warp.sim XPBD (installed) | separate code | **ignores E and ν**: stiffness comes from a relaxation parameter (`integrator_xpbd.py:546`, `k_mu`/`k_lambda` commented out) | unusable as a physical reference |
| warp.fem | yes | no built-in hyperelastic dynamics | would mean writing our own solver |

**Checked here:**

- conda-forge `polyfempy` 0.5.2 is a September 2023 build (feedstock path in its error messages). It has no fixed-corotated material, uses the legacy API, and our free-fall smoke test returned NaNs.
- The current PolyFEM (commit 008a1aea, Oct 13, 2026) is being built from source into `$BIG_ROOT/tools/polyfem` (CPU only; separate conda env `somw-fem` holding cmake 4.4 and ninja).
- The system cmake on this host is broken: `/usr/local/bin/cmake` is a pip stub with a missing package.

**Meshing.**
- Our queries are 2,048-point clouds with no mesh.
- Plan: use the points themselves as FEM nodes. Build a Delaunay tetrahedralisation, carved by an alpha rule, so trajectories compare node to node with no interpolation, and the drag box is exactly the same nodes.
- Risks: thin parts bridged or cut, and a coarse mesh (spacing about 8 cm). A per-query volume check excludes bad meshes.

**Matching the push.**
- MPM gives every pushed particle the same force vector f for the whole clip (`mpm_solver_warp.py:885–896`). The total is n_masked·f, which equals coeff × the object's weight (`regimes.py:25–28`).
- S2 applies the same nodal force to the same nodes.

**Expected disagreement, stated before any data.**
- Contact differs by design: MPM's grid floor is sticky and acts early (its grid contact acts before the surfaces touch, at about one grid cell; Fu, Jiang & Li, arXiv 2604.18917), while IPC is frictional and separating. So the drop family is secondary and two-object contact is excluded.
- Smooth single-object drag motion should agree well at small strain.
- Large-strain stretching (the "strand" items) may disagree. That is informative, not a failure of the check.

---

## 3. Real-world anchor (research only; a research agent opened every source)

No dataset gives a known distributed force, a measured E and ν, and 3D point tracks together. Every one lacks the material at least.

| Candidate | What it offers | Gap |
|---|---|---|
| **PokeFlex** (arXiv 2410.07688; registration with an institutional email, non-commercial) | 18 everyday deformables (sponge, plush octopus, foam dice…) dropped 3× from rest at about 2 m; 1 s at 60 fps; 106-camera 360° meshes per frame; robot pokes with a measured wrench | E must be estimated, e.g. fitted on the pokes and then tested on the drops; no point correspondence (score with Chamfer); floor friction unknown |
| Spring-Gaus captures (ECCV 2024; Google Drive) | 5 elastic objects hitting the ground, 3 calibrated RGB views | no 3D ground truth; image-space scoring only |
| PhysTwin, Deform360 | real 3D tracks of plush toys and other deformables | interactions are hand- or gripper-imposed displacements, which PhysCtrl cannot take as input |
| Physics-IQ, Phys101, MIT Push | real videos or forces | mostly rigid; no 3D points or no force |

**Realistic option before mid-November:** PokeFlex drops, which needs the registration request now. Any real-world result holds only up to the error in the estimated E and contact parameters, and must say so.

---

## 4. Arguments, both ways

### Circularity is acceptable

1. PhysCtrl was trained on exactly this simulator, so its output is meant to reproduce it. Measuring against the target it imitates is the standard protocol (PhysCtrl's own evaluation does the same).
2. The certificate's value is detecting departure from that target; it needs the target, not physical truth.
3. Using the simulator as the fallback is the method, not a trick: the paper's claim is "certify or re-simulate", and cost is the honest axis.
4. Phase 10c already shows the advantage does not depend on one time step of the reference (T1 PASS at dt, dt/2 and dt/4).

### Circularity is a real problem

1. On 52–57% of shifted and contact queries ours *is* the reference. "Lower error" there is guaranteed and says nothing about physics.
2. The dt/2 and dt/4 references in 10c are the same code, so they cannot reveal shared modelling errors: constitutive model, grid contact, the unrealistic regime found in the simulator audit.
3. A reader may read "more accurate than PhysCtrl" as physical accuracy, which no evidence supports.
4. The first human rater preferred PhysCtrl's motion on plausibility. That is consistent with the reference being unrealistic in some regimes (simulator audit, Oct 22).

### Weighing

- The circularity is real and should be stated plainly.
- Our defensible claims are about fidelity to PhysCtrl's own simulator: certification, repair and cost.
- An independent FEM cross-check decides whether "closer to physics" can be claimed for single-object drag scenes, by testing whether MPM is closer to FEM than PhysCtrl is. The pre-registration fixes in advance what each outcome means.

---

## 5. Pre-registered cross-check

The full text is in PLAN.md, "Phase 12". Summary:

- **Second simulator (S2):** PolyFEM, fixed-corotated, built from commit 008a1aea, implicit integration.
  - The time step is chosen by a fixed convergence rule on development queries only.
  - Fallback if the build or the gate fails: libuipc with stable neo-Hookean (the material difference is stated). If both fail, the check is reported as infeasible.
- **Sets:**
  - Primary (single-object drag): Phase 9 in distribution, stiffer, softer, stronger push.
  - Secondary: drop.
  - Descriptive: the 23 Phase 11 shifted items.
  - Excluded: two-object contact, because the contact models differ by design.
- **Gate S0 (development queries only):** free fall; small-strain agreement; matched push.
- **Tests:**
  - X1: ours less risky than PhysCtrl against S2.
  - X2: at least half of the advantage measured against MPM is retained against S2.
  - X3, the circularity test: on fallback queries, MPM is closer to S2 than PhysCtrl is.
- **Cost:** CPU only (dev server or `cpu-all`); runtime measured in the smoke test; 0 GPU-h.

## 6. What each outcome would mean for the paper

| Outcome | Meaning |
|---|---|
| S0 fails | No two independent solvers agree at our error scale. The paper claims fidelity to PhysCtrl's simulator only, and says so. |
| X1, X3 pass | Ours' advantage is not an artifact of scoring with our own solver for single-object drag scenes. "Closer to physics than PhysCtrl" can be claimed there, within FEM–MPM agreement. |
| X1 passes, X3 fails | Ours beats PhysCtrl against S2 mainly through repairs or noise, not because MPM is closer to S2 on fallback queries. The physics claim is limited to what X3 supports. |
| X1 fails | Against an independent solver, ours is not safer than PhysCtrl. Claims become fidelity-only, a substantial weakening that is discussed with the user before any paper change. |

## 7. Build and smoke test (Oct 22; synthetic unit cube only, no project data)

- **Build:** PolyFEM 008a1aea built from source (38 min, 16 cores); binary at `$BIG_ROOT/tools/polyfem/build/PolyFEM_bin`. `FixedCorotational` is in this build's material list.
- **Mesh:** a Delaunay mesh of a regular lattice gives zero-volume tetrahedra ("element is flipped"). Jittered points with degenerate tetrahedra removed run fine. Our point clouds are irregular, but the Phase 12 mesh rule must also drop near-zero-volume tetrahedra. This is a validity rule, added before any query is meshed.
- **Free fall** (gravity only, implicit Newmark, dt 0.01 s, t = 0.25 s): mean displacement 0.2946 vs ½gt² = 0.3066.
  - The 0.012 gap equals ½·g·dt·t, i.e. the integrator starts from zero acceleration.
  - The fix is to set the initial acceleration. Otherwise the gap grows with clip length (about 0.05 over 2 s at dt = frame/8). S0a will catch it if it is not fixed.
- **Sign convention:** PolyFEM's `rhs` produced motion opposite to its sign. Gravity downward must be written as `rhs = [0, +9.81, 0]`.
- **Interior nodal forces** (`nodal_neumann_boundary`, force units): 28 interior nodes loaded with a total force of 1000 N gave a centre-of-mass displacement of 0.0277. The expected value is 0.0277, using the mesh's actual mass (volume 1.083 m³) and the same zero-start offset. Interior loading works, which is the drag-box equivalent.
- **Cost:** 1,291 tetrahedra, 25 steps, 8 threads: 31–38 s.
  - Estimate for Phase 12: about 2,048 nodes and about 10⁴ tetrahedra per query, with 192–768 steps depending on the time-step rule. That suggests minutes to about half an hour per query, i.e. tens to a few hundred CPU-hours for the 500 primary queries.
  - This is an estimate only, to be measured in S0. It is CPU work (`cpu-all` or a dev server), 0 GPU-h.
- **Output:** the paraview export samples an upsampled visualisation mesh. The runner must export the solution at the original nodes (`output/data` with `reorder_nodes`, or vismesh off) so the comparison stays node to node.

## Correction (Oct 22, later): the drop family cannot be expressed by PhysCtrl's base model
- PhysCtrl base (`src/configs/eval_base.yaml`) has `floor_cond: false` and no `gravity_emb`. Only the large model (`eval_large.yaml`: `gravity_emb: true`, `floor_cond: true`) takes gravity and a floor as input (`src/model/spacetime.py:972–1047`).
- In our drop queries the base generator therefore receives no force and no gravity. Its 98% failure on ood3_drop is structural (an input it cannot represent), not a failure to generalise.
- What still holds: the certificate catches these silent failures, while PhysCtrl's own seed spread accepts 99% of failed drops (4m).
- What changes: "PhysCtrl fails under shift" must not lean on drop. Recommendation R1 above ("rest realism claims on drop") is withdrawn. Drop is evidence for the certificate, not for a generalisation failure. Phase 2 already recorded large at 0.120 vs base 0.401 on drop (PLAN l. 204).

