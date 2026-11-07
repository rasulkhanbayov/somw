# Pilot 8a: the fidelity ladder (EXPLORATORY, development data; defined Oct 17, 2026)

Cost relative to one reference rollout (dt 1e-4, grid 125). Unstable rollouts count as failures.

## Part A: full rollouts from the true initial state

| set | n | fidelity | risk > 0.10 | risk > 0.05 | median error | stable | cost |
|---|---|---|---|---|---|---|---|
| 4l contact | 255 | M1 | 0.020 | 0.106 | 0.0181 | 1.00 | 0.13 |
| 4l contact | 255 | M2 | 0.031 | 0.145 | 0.0177 | 1.00 | 0.25 |
| 4l contact | 255 | M3 | 0.004 | 0.055 | 0.0158 | 1.00 | 0.50 |
| 4l control | 79 | M1 | 0.000 | 0.000 | 0.0088 | 1.00 | 0.13 |
| 4l control | 79 | M2 | 0.000 | 0.000 | 0.0056 | 1.00 | 0.25 |
| 4l control | 79 | M3 | 0.000 | 0.013 | 0.0075 | 1.00 | 0.50 |
| id_new_0 | 100 | coarse_cfl | 0.070 | 0.130 | 0.0149 | 1.00 | 0.13 |
| id_new_1 | 100 | coarse_cfl | 0.060 | 0.150 | 0.0159 | 1.00 | 0.13 |
| id_new_10 | 100 | coarse_cfl | 0.030 | 0.070 | 0.0141 | 1.00 | 0.13 |
| id_new_11 | 100 | coarse_cfl | 0.030 | 0.070 | 0.0153 | 1.00 | 0.12 |
| id_new_2 | 100 | coarse_cfl | 0.050 | 0.120 | 0.0144 | 1.00 | 0.13 |
| id_new_3 | 100 | coarse_cfl | 0.020 | 0.080 | 0.0124 | 1.00 | 0.13 |
| id_new_4 | 100 | coarse_cfl | 0.040 | 0.070 | 0.0142 | 1.00 | 0.13 |
| id_new_5 | 100 | coarse_cfl | 0.030 | 0.060 | 0.0163 | 1.00 | 0.13 |
| id_new_6 | 100 | coarse_cfl | 0.040 | 0.080 | 0.0145 | 1.00 | 0.13 |
| id_new_7 | 100 | coarse_cfl | 0.040 | 0.100 | 0.0152 | 1.00 | 0.13 |
| id_new_8 | 100 | coarse_cfl | 0.020 | 0.060 | 0.0132 | 1.00 | 0.13 |
| id_new_9 | 100 | coarse_cfl | 0.070 | 0.100 | 0.0156 | 1.00 | 0.13 |
| ood_E_high | 50 | coarse_cfl | 0.040 | 0.060 | 0.0153 | 1.00 | 0.27 |
| ood_E_high | 50 | M1 | 0.020 | 0.040 | 0.0059 | 1.00 | 0.40 |
| ood_E_high | 50 | M2 | 0.020 | 0.040 | 0.0059 | 1.00 | 0.40 |
| ood_E_high | 50 | M3 | 0.000 | 0.000 | 0.0041 | 1.00 | 0.55 |
| ood_E_low | 50 | coarse_cfl | 0.260 | 0.440 | 0.0449 | 1.00 | 0.12 |
| ood_E_low | 50 | M1 | 0.040 | 0.180 | 0.0311 | 1.00 | 0.12 |
| ood_E_low | 50 | M2 | 0.040 | 0.120 | 0.0107 | 1.00 | 0.25 |
| ood_E_low | 50 | M3 | 0.160 | 0.300 | 0.0261 | 1.00 | 0.50 |
| ood_force_x3 | 50 | coarse_cfl | 0.340 | 0.580 | 0.0650 | 0.94 | 0.16 |
| ood_force_x3 | 50 | M1 | 0.060 | 0.220 | 0.0311 | 1.00 | 0.13 |
| ood_force_x3 | 50 | M2 | 0.060 | 0.140 | 0.0244 | 1.00 | 0.25 |
| ood_force_x3 | 50 | M3 | 0.080 | 0.280 | 0.0224 | 1.00 | 0.50 |
| ood_gravity_drop | 50 | coarse_cfl | 0.520 | 0.780 | 0.1034 | 1.00 | 0.12 |
| ood_gravity_drop | 50 | M1 | 0.220 | 0.540 | 0.0560 | 1.00 | 0.13 |
| ood_gravity_drop | 50 | M2 | 0.100 | 0.380 | 0.0381 | 1.00 | 0.25 |
| ood_gravity_drop | 50 | M3 | 0.160 | 0.280 | 0.0232 | 1.00 | 0.50 |
| test_id | 100 | coarse_cfl | 0.050 | 0.140 | 0.0171 | 1.00 | 0.13 |
| test_id | 100 | M1 | 0.000 | 0.030 | 0.0087 | 1.00 | 0.13 |
| test_id | 100 | M2 | 0.010 | 0.020 | 0.0065 | 1.00 | 0.25 |
| test_id | 100 | M3 | 0.010 | 0.020 | 0.0070 | 1.00 | 0.50 |

## Part B: pipelines on the 2c pool (gated V2 = the 4k rule at tau = 0.02614)

| set | pipeline | risk > 0.10 | mean error | cost |
|---|---|---|---|---|
| in distribution (test_id) | gated V2, M2 suffix, coarse shadow (B1) | 0.030 | 0.0360 | 0.30 |
| in distribution (test_id) | gated V2, M2 suffix, M1 shadow (B2) | 0.030 | 0.0324 | 0.30 |
| in distribution (test_id) | certify or simulate (reference) | 0.030 | 0.0192 | 0.66 |
| in distribution (test_id) | always simulate M1 | 0.000 | 0.0133 | 0.13 |
| in distribution (test_id) | certify, else simulate M1 | 0.030 | 0.0283 | 0.20 |
| in distribution (test_id) | always simulate M2 | 0.010 | 0.0106 | 0.25 |
| in distribution (test_id) | certify, else simulate M2 | 0.030 | 0.0259 | 0.27 |
| in distribution (test_id) | always simulate M3 | 0.010 | 0.0119 | 0.50 |
| in distribution (test_id) | certify, else simulate M3 | 0.040 | 0.0287 | 0.39 |
| shift (4 OOD splits) | gated V2, M2 suffix, coarse shadow (B1) | 0.015 | 0.0203 | 0.72 |
| shift (4 OOD splits) | gated V2, M2 suffix, M1 shadow (B2) | 0.010 | 0.0183 | 0.72 |
| shift (4 OOD splits) | certify or simulate (reference) | 0.005 | 0.0075 | 0.95 |
| shift (4 OOD splits) | always simulate M1 | 0.085 | 0.0402 | 0.20 |
| shift (4 OOD splits) | certify, else simulate M1 | 0.090 | 0.0456 | 0.28 |
| shift (4 OOD splits) | always simulate M2 | 0.055 | 0.0301 | 0.29 |
| shift (4 OOD splits) | certify, else simulate M2 | 0.060 | 0.0360 | 0.36 |
| shift (4 OOD splits) | always simulate M3 | 0.100 | 0.0355 | 0.51 |
| shift (4 OOD splits) | certify, else simulate M3 | 0.105 | 0.0418 | 0.55 |

## Pre-stated rules

- (a) medium full rollout risk <= 0.02 in every single-object split at <= 0.30x: M1 no, M2 no, M3 no
- (b) gated V2 with a medium suffix beats always-simulate-medium (risk +0.02, cost -20% ID / -10% shift): not applicable ((a) failed)
- contact: medium full rollout risk <= 0.02 on 4l contact scenes: M1 yes, M2 no, M3 yes

Exploratory pilot on development data; nothing here is a confirmation.
