# Phase 4h: certified-prefix repair (pre-registered Oct 13, 2026)

300 base queries (Phase 2c). tau = lambda = 0.02614. Costs in A100 s/query; reference rollout 4.666 s; generation 0.236 s. Suffix cost: share of the reference rollout (c_ref x substeps / 20016).

- **(1) cost saving** cert - suffix: +0.636 [+0.480, +0.789] s/query: **PASS**
- **(2) risk** suffix 0.070 vs cert 0.013 (limit +0.02): **FAIL**
- **Verdict: FAIL**

| pipeline | mean error | risk | cost (s/query) | cost / always-reference |
|---|---|---|---|---|
| certify-or-simulate | 0.0114 | 0.013 | 3.937 | 0.84 |
| **certified-prefix repair** | 0.0403 | 0.070 | 3.301 | 0.71 |
| always reference | 0 | 0 | 4.666 | 1.00 |

Modes: suffix 135, accept 80, full 85
First failing window (0 = anchor, 1..5 = chain windows at k = 3, 7, 11, 15, 19): 0: 85, 1: 52, 2: 26, 3: 16, 4: 12, 5: 29
Suffix lift floor (restart from ground-truth positions with F = I): mean error 0.0334 vs suffix repair 0.0660 on the same queries

| split | n | error suffix | error cert | risk suffix | risk cert | cost suffix | cost cert |
|---|---|---|---|---|---|---|---|
| ood_E_high | 50 | 0.0484 | 0.0215 | 0.08 | 0.02 | 2.21 | 2.87 |
| ood_E_low | 50 | 0.0641 | 0.0039 | 0.18 | 0.00 | 4.02 | 4.72 |
| ood_force_x3 | 50 | 0.0282 | 0.0046 | 0.02 | 0.00 | 4.04 | 4.71 |
| ood_gravity_drop | 50 | 0.0000 | 0.0000 | 0.00 | 0.00 | 5.00 | 5.24 |
| test_id | 100 | 0.0506 | 0.0192 | 0.07 | 0.03 | 2.26 | 3.04 |
