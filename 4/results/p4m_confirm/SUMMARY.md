# Phase 4m: fresh shifted queries, strongest baselines, placement at matched budget (pre-registered Oct 14, 2026)

Fresh queries: 598 (398 shifted in 4 families, 200 in distribution). Calibration pool: 1200 queries. Costs in A100 s/query (reference 4.666, generation 0.236). Blown restarts: 7; crashed: 0.

Thresholds (replicate-0 calibration; bad objects picked): window 0.026143 (22), seed_spread 0.012935 (22), uniform 0.02732 (22), agreement 0.084902 (10). Window tau reproduces 0.02614: True.

- **M1 certificate under shift:** FAR(window) +0.012 [+0.003, +0.024] (<= 0.10); FAR(seed spread) - FAR(window) +0.341 [+0.315, +0.363]: **PASS**
- **M2 pipelines under shift:** risk(gated V2) +0.015 [+0.005, +0.028] (<= 0.10); risk(coarse) - risk(gated V2) +0.359 [+0.310, +0.405]: **PASS**
- **M3 agreement baseline:** XF does not dominate gated V2. shift: risk XF 0.007 vs gated 0.015, saving gated - XF -0.475 [-0.607, -0.337]; in distribution: risk XF 0.005 vs gated 0.025, saving gated - XF -1.009 [-1.312, -0.716]
- **M4 placement at matched budget (descriptive):** shift: cost uniform - spread +0.506 [+0.389, +0.626] s, risk -0.005 [-0.013, +0.000]; in distribution: cost uniform - spread +1.237 [+0.957, +1.540] s, risk -0.005 [-0.015, +0.000]

## Shifted (pooled)

| pipeline | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only | 0.3470 | 0.703 | 0.05 |
| coarse_cfl only | 0.1922 | 0.374 | 0.15 |
| certify or simulate | 0.0074 | 0.010 | 0.97 |
| certify or simulate, uniform windows | 0.0026 | 0.005 | 1.08 |
| **gated V2** | 0.0157 | 0.015 | 0.94 |
| XF: accept coarse if it agrees with the generation | 0.0035 | 0.007 | 1.04 |
| seed-spread deferral (8 samples) | 0.1310 | 0.242 | 1.06 |

## In distribution

| pipeline | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only | 0.0731 | 0.115 | 0.05 |
| coarse_cfl only | 0.0305 | 0.055 | 0.12 |
| certify or simulate | 0.0214 | 0.005 | 0.56 |
| certify or simulate, uniform windows | 0.0093 | 0.000 | 0.82 |
| **gated V2** | 0.0360 | 0.025 | 0.44 |
| XF: accept coarse if it agrees with the generation | 0.0074 | 0.005 | 0.66 |
| seed-spread deferral (8 samples) | 0.0091 | 0.000 | 1.08 |

## Per split

| split | n | gen risk | coarse risk | cert risk / cost | gated risk / cost | XF risk / cost | FAR window | FAR seed spread |
|---|---|---|---|---|---|---|---|---|
| id_new_20 | 100 | 0.09 | 0.04 | 0.00 / 0.49 | 0.02 / 0.41 | 0.00 / 0.66 | 0.00 | 0.00 |
| id_new_21 | 100 | 0.14 | 0.07 | 0.01 / 0.63 | 0.03 / 0.48 | 0.01 / 0.65 | 0.07 | 0.00 |
| ood2_E_high | 100 | 0.06 | 0.04 | 0.02 / 0.55 | 0.04 / 0.48 | 0.00 / 0.80 | 0.33 | 0.00 |
| ood2_E_low | 99 | 0.79 | 0.31 | 0.00 / 1.10 | 0.00 / 1.01 | 0.00 / 1.08 | 0.00 | 0.00 |
| ood2_drop | 100 | 0.97 | 0.49 | 0.02 / 1.11 | 0.02 / 1.13 | 0.03 / 1.10 | 0.02 | 0.99 |
| ood2_force | 99 | 1.00 | 0.66 | 0.00 / 1.14 | 0.00 / 1.14 | 0.00 / 1.17 | 0.00 | 0.00 |
