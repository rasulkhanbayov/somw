# Phase 9: ours-v2 confirmation on fresh single-object queries (pre-registered Oct 19, 2026)

Queries: 600 (200 in distribution, 400 shifted). tau = 0.02614; eps = 0.1. CIs: object-cluster bootstrap, 2000 resamples, 0.99 two-sided (Bonferroni over 5). Costs in A100 s (reference 4.666 s, generation 0.236 s).

- **C1** ID risk: ours-v2 0.020 vs certify-or-simulate 0.015 (+0.02) and <= 0.035: **PASS**
- **C2** ID cost: ours-v2 1.219 s [0.984, 1.465] = 0.26x reference; upper bound <= 0.35 x c_ref (1.633 s): **PASS**
- **C3** ID saving vs certify-or-simulate: +1.296 [+1.030, +1.553] s: **PASS**
- **C4** shift: ours-v2 risk 0.013 (<= 0.03); risk(always M1) - risk(ours-v2) +0.090 [+0.055, +0.128]: **PASS**
- **C5** shift saving vs certify-or-simulate: +0.756 [+0.617, +0.905] s: **PASS**
- **Single-object part: CONFIRMED**

## in distribution

| pipeline | mean error | risk | cost / always reference |
|---|---|---|---|
| generate only | 0.0739 | 0.130 | 0.05 |
| certify or simulate (one window) | 0.0220 | 0.015 | 0.54 |
| **ours-v2** | 0.0315 | 0.020 | 0.26 |
| coarse only | 0.0315 | 0.040 | 0.13 |
| always M1 | 0.0138 | 0.015 | 0.13 |
| always M2 | 0.0110 | 0.020 | 0.25 |
| always M3 | 0.0132 | 0.015 | 0.50 |
| always reference | 0.0000 | 0.000 | 1.00 |
| momentum-residual gate | 0.0108 | 0.010 | 0.77 |
| seed-spread deferral | 0.0084 | 0.000 | 1.11 |

## shift

| pipeline | mean error | risk | cost / always reference |
|---|---|---|---|
| generate only | 0.3507 | 0.680 | 0.05 |
| certify or simulate (one window) | 0.0078 | 0.003 | 0.87 |
| **ours-v2** | 0.0165 | 0.013 | 0.71 |
| coarse only | 0.1184 | 0.367 | 0.16 |
| always M1 | 0.0488 | 0.102 | 0.20 |
| always M2 | 0.0479 | 0.102 | 0.29 |
| always M3 | 0.0759 | 0.155 | 0.51 |
| always reference | 0.0000 | 0.000 | 1.00 |
| momentum-residual gate | 0.0039 | 0.007 | 0.97 |
| seed-spread deferral | 0.1265 | 0.247 | 1.06 |

## Contact scenes: tuned simulators (descriptive; query-level)

- always M1: risk 0.030, cost 0.13x
- always M2: risk 0.038, cost 0.25x
- always M3: risk 0.000, cost 0.50x
