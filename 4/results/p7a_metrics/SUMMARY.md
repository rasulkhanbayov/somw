# Phase 7a: reviewer metrics and literature baselines (pre-registered Oct 17, 2026)

Fresh pool: 598 queries (398 shifted, 200 in distribution). eps = 0.1. CIs: object-cluster bootstrap, 2000 resamples, 0.9875 two-sided (Bonferroni over 4).

Momentum-residual threshold lambda_R = 0.012569 (calibration pool, 1200 queries, 22 bad objects picked). Unit check: residual of the ground truth on 40 gravity-free calibration queries: median 1.04e-04, max 1.57e-01 (≈ 0 means the force model matches the simulator).

- **Q1** AURC(window) - AURC(residual), pooled: -0.0628 [-0.0863, -0.0402]: **PASS**
- **Q2** AURC(window) - AURC(seed spread), pooled: -0.3388 [-0.3600, -0.3124]: **PASS**
- **Q3** risk(residual gate) - risk(gated V2), shift: +0.0025 [-0.0200, +0.0250]: **FAIL**
- **Q4** risk(best-of-8 by residual) - risk(gated V2), shift: +0.6808 [+0.6442, +0.7142]: **PASS**

## Selective prediction (bad = sample-0 error > 0.10; lower AURC is better)

| set | score | AUROC | AURC |
|---|---|---|---|
| pooled | window (ours) | 0.977 | 0.1718 |
| pooled | seed spread | 0.671 | 0.5106 |
| pooled | momentum residual | 0.886 | 0.2345 |
| pooled | agreement (needs coarse rollout) | 0.973 | 0.1825 |
| in distribution | window (ours) | 0.895 | 0.0210 |
| in distribution | seed spread | 0.873 | 0.0242 |
| in distribution | momentum residual | 0.737 | 0.0479 |
| in distribution | agreement (needs coarse rollout) | 0.930 | 0.0164 |
| shift | window (ours) | 0.980 | 0.3588 |
| shift | seed spread | 0.657 | 0.6977 |
| shift | momentum residual | 0.881 | 0.4415 |
| shift | agreement (needs coarse rollout) | 0.975 | 0.3748 |

## Pipelines, in distribution

| pipeline | risk 0.05 | risk 0.10 | risk 0.20 | mean error | p90 | p99 | max | last-frame error | cost / always-simulate |
|---|---|---|---|---|---|---|---|---|---|
| PhysCtrl (generate only) | 0.440 | 0.115 | 0.060 | 0.0731 | 0.1122 | 0.6425 | 1.0696 | 0.1589 | 0.05 |
| coarse simulation only | 0.105 | 0.055 | 0.030 | 0.0305 | 0.0516 | 0.2877 | 0.5259 | 0.0693 | 0.12 |
| seed-spread deferral (8 samples) | 0.025 | 0.000 | 0.000 | 0.0091 | 0.0306 | 0.0561 | 0.0967 | 0.0198 | 1.08 |
| momentum-residual gate (ANCHOR-style) | 0.055 | 0.000 | 0.000 | 0.0099 | 0.0343 | 0.0902 | 0.0981 | 0.0207 | 0.79 |
| best-of-8, medoid | 0.395 | 0.120 | 0.060 | 0.0714 | 0.1104 | 0.6296 | 1.0453 | — | 0.40 |
| best-of-8, least residual | 0.410 | 0.125 | 0.055 | 0.0719 | 0.1121 | 0.6299 | 1.0342 | — | 0.40 |
| best-of-8, oracle (upper bound) | 0.340 | 0.110 | 0.050 | 0.0658 | 0.1077 | 0.6241 | 1.0340 | — | 0.40 |
| accept coarse if it agrees (XF) | 0.015 | 0.005 | 0.000 | 0.0074 | 0.0169 | 0.0711 | 0.1663 | 0.0151 | 0.66 |
| certify or simulate | 0.115 | 0.005 | 0.005 | 0.0214 | 0.0509 | 0.0902 | 0.2320 | 0.0457 | 0.56 |
| **ours: certify + repair (gated V2)** | 0.170 | 0.025 | 0.015 | 0.0360 | 0.0579 | 0.2321 | 0.4422 | 0.0670 | 0.44 |

## Pipelines, shift

| pipeline | risk 0.05 | risk 0.10 | risk 0.20 | mean error | p90 | p99 | max | last-frame error | cost / always-simulate |
|---|---|---|---|---|---|---|---|---|---|
| PhysCtrl (generate only) | 0.818 | 0.703 | 0.562 | 0.3470 | 0.8147 | 1.2328 | 1.7359 | 0.7085 | 0.05 |
| coarse simulation only | 0.517 | 0.374 | 0.194 | 0.1922 | 0.3021 | 1.1486 | 10.0000 | 0.3897 | 0.15 |
| seed-spread deferral (8 samples) | 0.257 | 0.242 | 0.222 | 0.1310 | 0.5469 | 1.0396 | 1.2570 | 0.1771 | 1.06 |
| momentum-residual gate (ANCHOR-style) | 0.028 | 0.018 | 0.005 | 0.0056 | 0.0000 | 0.1117 | 0.4260 | 0.0120 | 0.97 |
| best-of-8, medoid | 0.805 | 0.698 | 0.565 | 0.3463 | 0.8156 | 1.2197 | 1.7586 | — | 0.40 |
| best-of-8, least residual | 0.818 | 0.696 | 0.565 | 0.3465 | 0.8141 | 1.2142 | 1.7559 | — | 0.40 |
| best-of-8, oracle (upper bound) | 0.785 | 0.676 | 0.550 | 0.3369 | 0.8125 | 1.2104 | 1.7359 | — | 0.40 |
| accept coarse if it agrees (XF) | 0.015 | 0.007 | 0.003 | 0.0035 | 0.0094 | 0.0581 | 0.2632 | 0.0060 | 1.04 |
| certify or simulate | 0.038 | 0.010 | 0.003 | 0.0074 | 0.0311 | 0.0980 | 0.2478 | 0.0157 | 0.97 |
| **ours: certify + repair (gated V2)** | 0.053 | 0.015 | 0.007 | 0.0157 | 0.0392 | 0.1402 | 0.4272 | 0.0279 | 0.94 |

Notes: risk and mean error are object-weighted (as in 4m); percentiles, maximum and AURC are query-level. Oracle best-of-8 picks the sample with the least true error; it is an upper bound for any selection rule, not a method. Seed-spread deferral and best-of-8 are charged 8 generations.
