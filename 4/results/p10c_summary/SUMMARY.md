# Phase 10c: every pipeline re-scored against the reference at dt, dt/2 and dt/4 (pre-registered Oct 21, 2026)

Queries: 732; risk = P(error > 0.1), object-weighted; CIs 0.983 (Bonferroni over 3).

## in distribution (200 queries)

| pipeline | risk vs dt | risk vs dt/2 | risk vs dt/4 |
|---|---|---|---|
| PhysCtrl | 0.130 | 0.110 | 0.100 |
| coarse | 0.040 | 0.035 | 0.035 |
| M1 | 0.015 | 0.025 | 0.025 |
| M2 | 0.020 | 0.020 | 0.015 |
| M3 | 0.015 | 0.030 | 0.030 |
| reference (dt) | 0.000 | 0.005 | 0.020 |
| residual gate | 0.010 | 0.015 | 0.025 |
| seed spread | 0.000 | 0.005 | 0.020 |
| certify or simulate | 0.015 | 0.020 | 0.030 |
| ours | 0.020 | 0.025 | 0.030 |

Median distance between references: dt-dt/2 0.0055, dt-dt/4 0.0103, dt/2-dt/4 0.0050
Unstable reference rollouts (left out): dt 0, dt/2 0, dt/4 0

## shift (400 queries)

| pipeline | risk vs dt | risk vs dt/2 | risk vs dt/4 |
|---|---|---|---|
| PhysCtrl | 0.680 | 0.665 | 0.650 |
| coarse | 0.367 | 0.274 | 0.234 |
| M1 | 0.102 | 0.224 | 0.329 |
| M2 | 0.102 | 0.099 | 0.226 |
| M3 | 0.155 | 0.232 | 0.324 |
| reference (dt) | 0.000 | 0.113 | 0.226 |
| residual gate | 0.007 | 0.118 | 0.228 |
| seed spread | 0.247 | 0.342 | 0.419 |
| certify or simulate | 0.003 | 0.116 | 0.226 |
| ours | 0.013 | 0.108 | 0.216 |

Median distance between references: dt-dt/2 0.0226, dt-dt/4 0.0437, dt/2-dt/4 0.0218
Unstable reference rollouts (left out): dt 0, dt/2 1, dt/4 1

## contact (132 queries)

| pipeline | risk vs dt | risk vs dt/2 | risk vs dt/4 |
|---|---|---|---|
| PhysCtrl | 0.693 | 0.722 | 0.737 |
| M1 | 0.029 | 0.029 | 0.082 |
| M2 | 0.035 | 0.023 | 0.041 |
| M3 | 0.000 | 0.019 | 0.130 |
| reference (dt) | 0.000 | 0.000 | 0.037 |
| certify or simulate | 0.000 | 0.000 | 0.037 |
| ours | 0.012 | 0.012 | 0.062 |

Median distance between references: dt-dt/2 0.0161, dt-dt/4 0.0323, dt/2-dt/4 0.0164
Unstable reference rollouts (left out): dt 0, dt/2 0, dt/4 0

## ood3_E_high (100 queries)

| pipeline | risk vs dt | risk vs dt/2 | risk vs dt/4 |
|---|---|---|---|
| PhysCtrl | 0.060 | 0.060 | 0.040 |
| coarse | 0.020 | 0.020 | 0.020 |
| M1 | 0.010 | 0.010 | 0.010 |
| M2 | 0.010 | 0.010 | 0.010 |
| M3 | 0.000 | 0.000 | 0.000 |
| reference (dt) | 0.000 | 0.000 | 0.000 |
| residual gate | 0.000 | 0.000 | 0.000 |
| seed spread | 0.010 | 0.010 | 0.000 |
| certify or simulate | 0.010 | 0.010 | 0.000 |
| ours | 0.020 | 0.020 | 0.010 |

Median distance between references: dt-dt/2 0.0007, dt-dt/4 0.0013, dt/2-dt/4 0.0009
Unstable reference rollouts (left out): dt 0, dt/2 0, dt/4 0

## ood3_E_low (100 queries)

| pipeline | risk vs dt | risk vs dt/2 | risk vs dt/4 |
|---|---|---|---|
| PhysCtrl | 0.680 | 0.616 | 0.586 |
| coarse | 0.330 | 0.192 | 0.121 |
| M1 | 0.020 | 0.263 | 0.424 |
| M2 | 0.130 | 0.030 | 0.242 |
| M3 | 0.240 | 0.323 | 0.424 |
| reference (dt) | 0.000 | 0.202 | 0.303 |
| residual gate | 0.030 | 0.222 | 0.313 |
| seed spread | 0.010 | 0.202 | 0.303 |
| certify or simulate | 0.000 | 0.202 | 0.303 |
| ours | 0.020 | 0.152 | 0.253 |

Median distance between references: dt-dt/2 0.0297, dt-dt/4 0.0602, dt/2-dt/4 0.0308
Unstable reference rollouts (left out): dt 0, dt/2 1, dt/4 1

## ood3_drop (100 queries)

| pipeline | risk vs dt | risk vs dt/2 | risk vs dt/4 |
|---|---|---|---|
| PhysCtrl | 0.980 | 0.980 | 0.970 |
| coarse | 0.530 | 0.410 | 0.410 |
| M1 | 0.180 | 0.240 | 0.340 |
| M2 | 0.110 | 0.190 | 0.260 |
| M3 | 0.110 | 0.220 | 0.320 |
| reference (dt) | 0.000 | 0.070 | 0.190 |
| residual gate | 0.000 | 0.070 | 0.190 |
| seed spread | 0.970 | 0.970 | 0.960 |
| certify or simulate | 0.000 | 0.070 | 0.190 |
| ours | 0.000 | 0.070 | 0.190 |

Median distance between references: dt-dt/2 0.0264, dt-dt/4 0.0453, dt/2-dt/4 0.0217
Unstable reference rollouts (left out): dt 0, dt/2 0, dt/4 0

## ood3_force (100 queries)

| pipeline | risk vs dt | risk vs dt/2 | risk vs dt/4 |
|---|---|---|---|
| PhysCtrl | 1.000 | 1.000 | 1.000 |
| coarse | 0.590 | 0.470 | 0.380 |
| M1 | 0.200 | 0.380 | 0.540 |
| M2 | 0.160 | 0.160 | 0.390 |
| M3 | 0.270 | 0.380 | 0.550 |
| reference (dt) | 0.000 | 0.180 | 0.410 |
| residual gate | 0.000 | 0.180 | 0.410 |
| seed spread | 0.000 | 0.180 | 0.410 |
| certify or simulate | 0.000 | 0.180 | 0.410 |
| ours | 0.010 | 0.190 | 0.410 |

Median distance between references: dt-dt/2 0.0379, dt-dt/4 0.0718, dt/2-dt/4 0.0369
Unstable reference rollouts (left out): dt 0, dt/2 0, dt/4 0

## Tests

- T1 in distribution, vs dt: risk(PhysCtrl) - risk(ours) +0.110 [+0.050, +0.180]
- T1 in distribution, vs dt/2: risk(PhysCtrl) - risk(ours) +0.085 [+0.030, +0.150]
- T1 in distribution, vs dt/4: risk(PhysCtrl) - risk(ours) +0.070 [+0.020, +0.132]
- T1 shift, vs dt: risk(PhysCtrl) - risk(ours) +0.667 [+0.632, +0.701]
- T1 shift, vs dt/2: risk(PhysCtrl) - risk(ours) +0.557 [+0.516, +0.597]
- T1 shift, vs dt/4: risk(PhysCtrl) - risk(ours) +0.434 [+0.385, +0.476]
- T1 contact, vs dt: risk(PhysCtrl) - risk(ours) +0.681 [+0.578, +0.778]
- T1 contact, vs dt/2: risk(PhysCtrl) - risk(ours) +0.710 [+0.607, +0.805]
- T1 contact, vs dt/4: risk(PhysCtrl) - risk(ours) +0.675 [+0.565, +0.778]
- **T1 (PhysCtrl worse than ours everywhere, every reference): PASS**

- T2 in distribution, vs dt: risk(ours) 0.020
- T2 in distribution, vs dt/2: risk(ours) 0.025
- T2 in distribution, vs dt/4: risk(ours) 0.030
- T2 shift, vs dt: risk(ours) 0.013
- T2 shift, vs dt/2: risk(ours) 0.108
- T2 shift, vs dt/4: risk(ours) 0.216
- T2 contact, vs dt: risk(ours) 0.012
- T2 contact, vs dt/2: risk(ours) 0.012
- T2 contact, vs dt/4: risk(ours) 0.062
- **T2 (ours <= 0.03 everywhere, every reference): FAIL**

- T3 shift, M1 vs dt: risk(M1) - risk(ours) +0.090 [+0.058, +0.125]
- T3 shift, M1 vs dt/2: risk(M1) - risk(ours) +0.116 [+0.083, +0.150]
- T3 shift, M1 vs dt/4: risk(M1) - risk(ours) +0.113 [+0.073, +0.152]
- T3 shift, M2 vs dt: risk(M2) - risk(ours) +0.090 [+0.060, +0.128]
- T3 shift, M2 vs dt/2: risk(M2) - risk(ours) -0.009 [-0.045, +0.026]
- T3 shift, M2 vs dt/4: risk(M2) - risk(ours) +0.010 [-0.013, +0.035]
- T3 shift, M3 vs dt: risk(M3) - risk(ours) +0.142 [+0.102, +0.182]
- T3 shift, M3 vs dt/2: risk(M3) - risk(ours) +0.123 [+0.085, +0.165]
- T3 shift, M3 vs dt/4: risk(M3) - risk(ours) +0.108 [+0.072, +0.144]
- **T3 M1 (less safe than ours under shift against every reference): PASS**
- **T3 M2 (less safe than ours under shift against every reference): FAIL**
- **T3 M3 (less safe than ours under shift against every reference): PASS**
