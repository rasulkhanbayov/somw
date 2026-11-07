# Phase 10a: is the reference converged? (pre-registered Oct 20, 2026)

d = distance to the stored reference (dt 1e-4, grid 125). Rule R1: 95th percentile of d_half <= 0.02.

| set | n | d_half median / p95 / max | d_quarter (subset) n, median / p95 / max | R1 |
|---|---|---|---|---|
| in distribution | 200 | 0.0055 / 0.0262 / 0.2076 | 50, 0.0122 / 0.0922 / 0.1688 | **NOT converged** |
| ood3_E_high | 100 | 0.0007 / 0.0086 / 0.0267 | 25, 0.0010 / 0.0157 / 0.0534 | **converged** |
| ood3_E_low | 100 | 0.0300 / 0.1995 / inf | 25, 0.0680 / 0.3606 / inf | **NOT converged** |
| ood3_drop | 100 | 0.0264 / 0.1150 / 0.3958 | 25, 0.0487 / 0.1912 / 0.2439 | **NOT converged** |
| ood3_force | 100 | 0.0379 / 0.4267 / 1.1823 | 25, 0.1126 / 0.3688 / 0.3877 | **NOT converged** |
| shift (pooled) | 400 | 0.0227 / 0.1669 / inf | 100, 0.0496 / 0.2714 / inf | **NOT converged** |
| contact | 132 | 0.0161 / 0.0444 / 0.0807 | 22, 0.0267 / 0.0585 / 0.0962 | **NOT converged** |

## R2: risk against a finer reference, bounds [lower, upper] (point = Phase 9)

| pipeline | in distribution | ood3_E_high | ood3_E_low | ood3_drop | ood3_force | shift (pooled) | contact |
|---|---|---|---|---|---|---|---|
| generate only | 0.100 / 0.130 / 0.180 | 0.050 / 0.060 / 0.060 | 0.580 / 0.680 / 0.750 | 0.980 / 0.980 / 0.980 | 0.960 / 1.000 / 1.000 | 0.642 / 0.680 / 0.698 | 0.619 / 0.693 / 0.770 |
| coarse only | 0.035 / 0.040 / 0.040 | 0.020 / 0.020 / 0.020 | 0.180 / 0.330 / 0.370 | 0.300 / 0.530 / 0.650 | 0.340 / 0.590 / 0.730 | 0.210 / 0.367 / 0.443 | — |
| always M1 | 0.015 / 0.015 / 0.030 | 0.010 / 0.010 / 0.010 | 0.000 / 0.020 / 0.300 | 0.060 / 0.180 / 0.340 | 0.010 / 0.200 / 0.490 | 0.020 / 0.102 / 0.285 | 0.023 / 0.029 / 0.041 |
| always M2 | 0.015 / 0.020 / 0.035 | 0.010 / 0.010 / 0.010 | 0.000 / 0.130 / 0.270 | 0.020 / 0.110 / 0.310 | 0.030 / 0.160 / 0.380 | 0.015 / 0.102 / 0.242 | 0.019 / 0.035 / 0.078 |
| always M3 | 0.010 / 0.015 / 0.030 | 0.000 / 0.000 / 0.000 | 0.020 / 0.240 / 0.340 | 0.020 / 0.110 / 0.230 | 0.090 / 0.270 / 0.430 | 0.033 / 0.155 / 0.250 | 0.000 / 0.000 / 0.019 |
| always reference | 0.000 / 0.000 / 0.005 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.210 | 0.000 / 0.000 / 0.070 | 0.000 / 0.000 / 0.180 | 0.000 / 0.000 / 0.115 | 0.000 / 0.000 / 0.000 |
| momentum-residual gate | 0.005 / 0.010 / 0.020 | 0.000 / 0.000 / 0.000 | 0.020 / 0.030 / 0.250 | 0.000 / 0.000 / 0.070 | 0.000 / 0.000 / 0.180 | 0.005 / 0.007 / 0.125 | — |
| seed-spread deferral | 0.000 / 0.000 / 0.005 | 0.010 / 0.010 / 0.010 | 0.000 / 0.010 / 0.220 | 0.970 / 0.970 / 0.970 | 0.000 / 0.000 / 0.180 | 0.245 / 0.247 / 0.345 | — |
| certify or simulate | 0.015 / 0.015 / 0.025 | 0.010 / 0.010 / 0.010 | 0.000 / 0.000 / 0.210 | 0.000 / 0.000 / 0.070 | 0.000 / 0.000 / 0.180 | 0.003 / 0.003 / 0.117 | 0.000 / 0.000 / 0.000 |
| ours | 0.020 / 0.020 / 0.035 | 0.020 / 0.020 / 0.020 | 0.000 / 0.020 / 0.260 | 0.000 / 0.000 / 0.070 | 0.010 / 0.010 / 0.200 | 0.007 / 0.013 / 0.138 | 0.012 / 0.012 / 0.031 |

- (a) ours' upper bound <= 0.03 in distribution, under shift and on contact: **FAIL**
- (b) under shift every tuned setting's lower bound > 0.03: **FAIL**
- **Claims robust to the reference's time-step error: NO**

## R3: free fall before the floor (ood3_drop ground truth)

78 queries with >= 2 free frames; centre-of-mass deviation from y0 + v0 t - g t^2/2: median 8.10e-05, max 1.22e-04: **PASS** (tolerance 0.01).

## Recorded: frame-time offset of the tuned settings (descriptive)

| setting | split | offset error median / p95 / max |
|---|---|---|
| M1 | id_new_22 | 0.0042 / 0.0190 / 0.0271 |
| M1 | id_new_23 | 0.0036 / 0.0113 / 0.0208 |
| M1 | ood3_E_high | 0.0035 / 0.0109 / 0.0144 |
| M1 | ood3_E_low | 0.0032 / 0.0056 / 0.0064 |
| M1 | ood3_drop | 0.0007 / 0.0026 / 0.0091 |
| M1 | ood3_force | 0.0087 / 0.0404 / 0.0645 |
| M2 | id_new_22 | 0.0035 / 0.0061 / 0.0066 |
| M2 | id_new_23 | 0.0032 / 0.0061 / 0.0063 |
| M2 | ood3_E_high | 0.0035 / 0.0105 / 0.0144 |
| M2 | ood3_E_low | 0.0032 / 0.0056 / 0.0064 |
| M2 | ood3_drop | 0.0006 / 0.0026 / 0.0052 |
| M2 | ood3_force | 0.0084 / 0.0108 / 0.0118 |
| M3 | id_new_22 | 0.0034 / 0.0061 / 0.0066 |
| M3 | id_new_23 | 0.0032 / 0.0061 / 0.0063 |
| M3 | ood3_E_high | 0.0030 / 0.0064 / 0.0086 |
| M3 | ood3_E_low | 0.0032 / 0.0056 / 0.0064 |
| M3 | ood3_drop | 0.0006 / 0.0026 / 0.0034 |
| M3 | ood3_force | 0.0084 / 0.0108 / 0.0118 |
