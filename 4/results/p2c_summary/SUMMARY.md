# Phase 2c summary (pre-registered Oct 11, 2026)

- A fidelity, fixed coarse dt 8e-4 (contribution 3): FAIL
- A' fidelity, CFL-adaptive coarse (contribution 3): PASS
- B window length + fusion, large (contribution 1): FAIL
- C placement on contact events (contribution 2): PASS
- D single-threshold certificate, base (contribution 1): PASS

## A. Simulator fidelity (m = 1, uniform-3)

| fidelity | model | floor / defect (ID) | rho all [95% CI] | rho OOD |
|---|---|---|---|---|
| ref | base | 0.269 | +0.82 [+0.78,+0.85] | +0.79 |
| ref | large | 0.203 | +0.54 [+0.43,+0.62] | +0.57 |
| coarse | base | 0.233 | +0.62 [+0.52,+0.70] | +0.43 |
| coarse | large | 0.176 | +0.45 [+0.34,+0.55] | +0.39 |
| coarse_cfl | base | 0.233 | +0.81 [+0.77,+0.84] | +0.78 |
| coarse_cfl | large | 0.176 | +0.57 [+0.47,+0.66] | +0.62 |

Windows skipped by the CFL gate and scored as blown (+inf): coarse/ood_E_high 1017

## B. Window length and fusion with seed spread ('ref', uniform-3)

| model | m | rho defect | rho fusion | rho seed spread | fusion - seed spread, OOD [95% CI] |
|---|---|---|---|---|---|
| base | 1 | 0.82 | 0.64 | 0.21 | +0.52 [+0.39,+0.63] |
| base | 2 | 0.86 | 0.67 | 0.21 | +0.55 [+0.43,+0.67] |
| base | 4 | 0.91 | 0.71 | 0.21 | +0.61 [+0.48,+0.73] |
| large | 1 | 0.54 | 0.76 | 0.82 | -0.05 [-0.11,-0.01] |
| large | 2 | 0.64 | 0.80 | 0.82 | -0.02 [-0.07,+0.02] |
| large | 4 | 0.82 | 0.86 | 0.82 | +0.04 [+0.02,+0.07] |

## C. Placement: capture of the exhaustive defect ('ref', m = 1, k = 3)

| model | split group | uniform | random | event | event_union | disagree | oracle_v1 | event_union - uniform [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base | ood_gravity_drop | 0.99 | 0.97 | 0.98 | 0.99 | 0.97 | 0.98 | +0.01 [+0.00,+0.01] |
| base | drag-only | 0.98 | 0.68 | 0.84 | 0.84 | 0.87 | 0.72 | -0.14 [-0.17,-0.12] |
| large | ood_gravity_drop | 0.61 | 0.65 | 0.96 | 0.77 | 0.57 | 0.82 | +0.17 [+0.06,+0.27] |
| large | drag-only | 0.99 | 0.58 | 0.73 | 0.73 | 0.72 | 0.60 | -0.26 [-0.29,-0.23] |

Defect localisation (max window / sum over windows, 'ref', m = 1; uniform spread = 1/23 = 0.043): base/ood_E_high 0.098, base/ood_E_low 0.082, base/ood_force_x3 0.085, base/ood_gravity_drop 0.045, base/test_id 0.091, large/ood_E_high 0.117, large/ood_E_low 0.116, large/ood_force_x3 0.088, large/ood_gravity_drop 0.092, large/test_id 0.112

## D. One threshold across all splits: AUROC for err_mean > eps ('ref', m = 1, uniform-3)

| model | eps | positives | AUROC defect | AUROC seed spread | AUROC fusion | defect - seed spread [95% CI] |
|---|---|---|---|---|---|---|
| base | 0.05 | 210/300 | 0.882 | 0.702 | 0.878 | +0.18 [+0.11,+0.25] |
| base | 0.1 | 145/300 | 0.911 | 0.616 | 0.829 | +0.30 [+0.21,+0.38] |
| base | 0.2 | 95/300 | 0.918 | 0.462 | 0.723 | +0.46 [+0.35,+0.56] |
| large | 0.05 | 276/300 | 0.605 | 0.901 | 0.792 | -0.30 [-0.44,-0.16] |
| large | 0.1 | 200/300 | 0.733 | 0.924 | 0.874 | -0.19 [-0.26,-0.13] |
| large | 0.2 | 85/300 | 0.825 | 0.926 | 0.910 | -0.10 [-0.15,-0.05] |
