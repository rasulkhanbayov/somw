# Phase 4i: full-state suffix restarts (pre-registered Oct 13, 2026)

300 base queries (Phase 2c); 135 suffix queries change. tau = 0.02614. Costs in A100 s/query: reference 4.666, generation 0.236; suffix = c_ref x substeps / 20016; V2 shadow = measured 4d coarse_cfl A100 s/substep. Bonferroni alpha = 0.025.

Blow-up fallbacks: V1 0, V2 0; crashed units: 0.

## V1 stable F-lift: **FAIL**
- (1) cost saving cert - v1: +0.636 [+0.455, +0.804] s/query (97.5% CI): **PASS**
- (2) risk 0.057 vs cert 0.013 (limit +0.02): **FAIL**

## V2 coarse shadow: **PASS**
- (1) cost saving cert - v2: +0.505 [+0.335, +0.661] s/query (97.5% CI): **PASS**
- (2) risk 0.017 vs cert 0.013 (limit +0.02): **PASS**

| pipeline | mean error | risk | cost (s/query) | cost / always-reference |
|---|---|---|---|---|
| certify-or-simulate | 0.0114 | 0.013 | 3.937 | 0.84 |
| 4h suffix (F = I, generated positions) | 0.0403 | 0.070 | 3.301 | 0.71 |
| **V1 stable F-lift** | 0.0360 | 0.057 | 3.301 | 0.71 |
| **V2 coarse shadow** | 0.0235 | 0.017 | 3.433 | 0.74 |
| always reference | 0 | 0 | 4.666 | 1.00 |

## Secondary (on the 135 suffix queries)

| comparison | mean error difference (variant - 4h) | 95% CI |
|---|---|---|
| v1 vs 4h | -0.0097 | [-0.0148, -0.0056] |
| v2 vs 4h | -0.0375 | [-0.0436, -0.0320] |

- Suffix floor (ground-truth positions): V1 stable F 0.0070 vs 4h F = I 0.0334; V1 floor blow-ups 0.
- Position error at k_j: V2 shadow 0.0225 vs generated 0.0501.
- Share of singular values clamped by V1: mean 0.225 (bounds [0.8797011242508889, 1.3529454807043129]).

| split | n suffix | 4h error | V1 error | V2 error | 4h risk | V1 risk | V2 risk |
|---|---|---|---|---|---|---|---|
| ood_E_high | 22 | 0.0599 | 0.0598 | 0.0342 | 0.14 | 0.14 | 0.05 |
| ood_E_low | 36 | 0.0836 | 0.0545 | 0.0217 | 0.25 | 0.06 | 0.00 |
| ood_force_x3 | 23 | 0.0574 | 0.0564 | 0.0310 | 0.04 | 0.04 | 0.00 |
| test_id | 54 | 0.0604 | 0.0559 | 0.0297 | 0.07 | 0.13 | 0.00 |

| first failure j (k_j) | n | 4h error | V1 error | V2 error |
|---|---|---|---|---|
| 1 (k = 3) | 52 | 0.0726 | 0.0491 | 0.0155 |
| 2 (k = 7) | 26 | 0.0688 | 0.0626 | 0.0299 |
| 3 (k = 11) | 16 | 0.0742 | 0.0754 | 0.0378 |
| 4 (k = 15) | 12 | 0.0710 | 0.0748 | 0.0496 |
| 5 (k = 19) | 29 | 0.0450 | 0.0451 | 0.0369 |
