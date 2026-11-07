# Phase 4a: conformal certificate (pre-registered Oct 11, 2026)

Calibration set: 800 trajectories; 100 objects split 50/50 by object (seed 0). Accept if score < lambda; guarantee P(accept | bad) <= alpha in distribution.

- **Primary (base_defect|eps=0.1|alpha=0.1)**: P1 validity PASS; P2 power vs seed spread PASS


## base_defect (base; score = base_cfl_m4_spread)

| eps | alpha | n bad (cal) | lambda | test bad | false-accept rate [95% CP] | P1 | test good | good accepted | seed-spread cert: good accepted | defect - seed spread [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 76 | 0.01528 | 96 | 0.021 [0.003,0.073] | PASS | 104 | 0.433 | 0.452 | -0.019 [-0.101,+0.061] |
| 0.05 | 0.1 | 76 | 0.01961 | 96 | 0.094 [0.044,0.171] | PASS | 104 | 0.654 | 0.625 | +0.029 [-0.033,+0.096] |
| 0.1 | 0.05 | 25 | 0.01905 | 36 | 0.111 [0.031,0.261] | PASS | 164 | 0.409 | 0.293 | +0.116 [+0.070,+0.164] |
| 0.1 | 0.1 | 25 | 0.03007 | 36 | 0.194 [0.082,0.360] | PASS | 164 | 0.689 | 0.488 | +0.201 [+0.133,+0.273] |
| 0.2 | 0.05 | 8 | -inf | 14 | 0.000 [0.000,0.232] | PASS | 186 | 0.000 | 0.000 | +0.000 [+0.000,+0.000] |
| 0.2 | 0.1 | 8 | -inf | 14 | 0.000 [0.000,0.232] | PASS | 186 | 0.000 | 0.000 | +0.000 [+0.000,+0.000] |

P3 (shift, empirical) for eps = 0.1, alpha = 0.1, lambda = 0.03007:

| split | n | bad | false-accept rate among bad | good | acceptance among good |
|---|---|---|---|---|---|
| ood_E_high | 50 | 5 | 0.200 | 45 | 0.756 |
| ood_E_low | 50 | 32 | 0.000 | 18 | 0.389 |
| ood_force_x3 | 50 | 36 | 0.000 | 14 | 0.500 |
| ood_gravity_drop | 50 | 50 | 0.000 | 0 | nan |
| test_id | 100 | 22 | 0.136 | 78 | 0.641 |

## large_seed_spread (large; score = seed_spread)

| eps | alpha | n bad (cal) | lambda | test bad | false-accept rate [95% CP] | P1 | test good | good accepted | seed-spread cert: good accepted | defect - seed spread [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 188 | 0.01628 | 183 | 0.033 [0.012,0.070] | PASS | 17 | 0.412 | 0.412 | +0.000 [+0.000,+0.000] |
| 0.05 | 0.1 | 188 | 0.02092 | 183 | 0.071 [0.038,0.118] | PASS | 17 | 0.824 | 0.824 | +0.000 [+0.000,+0.000] |
| 0.1 | 0.05 | 108 | 0.0243 | 113 | 0.009 [0.000,0.048] | PASS | 87 | 0.356 | 0.356 | +0.000 [+0.000,+0.000] |
| 0.1 | 0.1 | 108 | 0.02776 | 113 | 0.027 [0.006,0.076] | PASS | 87 | 0.552 | 0.552 | +0.000 [+0.000,+0.000] |
| 0.2 | 0.05 | 18 | -inf | 26 | 0.000 [0.000,0.132] | PASS | 174 | 0.000 | 0.000 | +0.000 [+0.000,+0.000] |
| 0.2 | 0.1 | 18 | 0.02968 | 26 | 0.000 [0.000,0.132] | PASS | 174 | 0.339 | 0.339 | +0.000 [+0.000,+0.000] |

P3 (shift, empirical) for eps = 0.1, alpha = 0.1, lambda = 0.02776:

| split | n | bad | false-accept rate among bad | good | acceptance among good |
|---|---|---|---|---|---|
| ood_E_high | 50 | 26 | 0.038 | 24 | 0.417 |
| ood_E_low | 50 | 41 | 0.024 | 9 | 1.000 |
| ood_force_x3 | 50 | 48 | 0.021 | 2 | 0.500 |
| ood_gravity_drop | 50 | 26 | 0.038 | 24 | 0.833 |
| test_id | 100 | 59 | 0.051 | 41 | 0.561 |

## large_fused (large; score = fused:large_ref_m4_spread)

| eps | alpha | n bad (cal) | lambda | test bad | false-accept rate [95% CP] | P1 | test good | good accepted | seed-spread cert: good accepted | defect - seed spread [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 188 | 0.0825 | 183 | 0.049 [0.023,0.091] | PASS | 17 | 0.706 | 0.412 | +0.294 [+0.111,+0.500] |
| 0.05 | 0.1 | 188 | 0.155 | 183 | 0.087 [0.051,0.138] | PASS | 17 | 0.824 | 0.824 | +0.000 [+0.000,+0.000] |
| 0.1 | 0.05 | 108 | 0.265 | 113 | 0.035 [0.010,0.088] | PASS | 87 | 0.483 | 0.356 | +0.126 [+0.054,+0.208] |
| 0.1 | 0.1 | 108 | 0.3125 | 113 | 0.062 [0.025,0.123] | PASS | 87 | 0.586 | 0.552 | +0.034 [-0.033,+0.106] |
| 0.2 | 0.05 | 18 | -inf | 26 | 0.000 [0.000,0.132] | PASS | 174 | 0.000 | 0.000 | +0.000 [+0.000,+0.000] |
| 0.2 | 0.1 | 18 | 0.5475 | 26 | 0.077 [0.009,0.251] | PASS | 174 | 0.603 | 0.339 | +0.264 [+0.200,+0.331] |

P3 (shift, empirical) for eps = 0.1, alpha = 0.1, lambda = 0.3125:

| split | n | bad | false-accept rate among bad | good | acceptance among good |
|---|---|---|---|---|---|
| ood_E_high | 50 | 26 | 0.038 | 24 | 0.583 |
| ood_E_low | 50 | 41 | 0.000 | 9 | 0.556 |
| ood_force_x3 | 50 | 48 | 0.000 | 2 | 0.500 |
| ood_gravity_drop | 50 | 26 | 0.000 | 24 | 0.333 |
| test_id | 100 | 59 | 0.051 | 41 | 0.659 |

## base_seed_spread (base; score = seed_spread)

| eps | alpha | n bad (cal) | lambda | test bad | false-accept rate [95% CP] | P1 | test good | good accepted | seed-spread cert: good accepted | defect - seed spread [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 76 | 0.01263 | 96 | 0.010 [0.000,0.057] | PASS | 104 | 0.452 | 0.452 | +0.000 [+0.000,+0.000] |
| 0.05 | 0.1 | 76 | 0.015 | 96 | 0.083 [0.037,0.158] | PASS | 104 | 0.625 | 0.625 | +0.000 [+0.000,+0.000] |
| 0.1 | 0.05 | 25 | 0.01263 | 36 | 0.000 [0.000,0.097] | PASS | 164 | 0.293 | 0.293 | +0.000 [+0.000,+0.000] |
| 0.1 | 0.1 | 25 | 0.01577 | 36 | 0.056 [0.007,0.187] | PASS | 164 | 0.488 | 0.488 | +0.000 [+0.000,+0.000] |
| 0.2 | 0.05 | 8 | -inf | 14 | 0.000 [0.000,0.232] | PASS | 186 | 0.000 | 0.000 | +0.000 [+0.000,+0.000] |
| 0.2 | 0.1 | 8 | -inf | 14 | 0.000 [0.000,0.232] | PASS | 186 | 0.000 | 0.000 | +0.000 [+0.000,+0.000] |

P3 (shift, empirical) for eps = 0.1, alpha = 0.1, lambda = 0.01577:

| split | n | bad | false-accept rate among bad | good | acceptance among good |
|---|---|---|---|---|---|
| ood_E_high | 50 | 5 | 0.200 | 45 | 0.444 |
| ood_E_low | 50 | 32 | 0.000 | 18 | 0.333 |
| ood_force_x3 | 50 | 36 | 0.000 | 14 | 0.214 |
| ood_gravity_drop | 50 | 50 | 1.000 | 0 | nan |
| test_id | 100 | 22 | 0.091 | 78 | 0.449 |
