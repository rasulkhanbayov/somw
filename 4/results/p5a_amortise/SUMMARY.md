# Phase 5a: audited amortisation of the certificate (pre-registered Oct 16, 2026)

Stream: 1698 single-object queries (1100 in distribution, 598 shifted; pools 4k, 4m, 2c). tau = 0.02614; audit rate 0.1; delta 0.05; 50 random orders. Values: mean [2.5%, 97.5%] over orders. Costs in A100 s (reference 4.666).

- **A1 amortisation:** probe rate first third 0.994 [0.975, 1.000] -> last third 0.848 [0.805, 0.891]; difference 0.146 [0.093, 0.195]: **PASS**
- **A2 risk:** amortised 0.011 [0.011, 0.012] vs always-probe 0.011 [0.011, 0.011]; difference 0.000 [0.000, 0.001] (mean <= 0.02): **PASS**
- **A3 cost:** saving 0.044 [0.027, 0.068] s/query (always-probe 3.492 [3.492, 3.492], amortised 3.447 [3.424, 3.465]): **PASS**
- **Confirmation: CONFIRMED**

Skip rate 0.089 [0.062, 0.123]; risk among skipped queries 0.012 [0.000, 0.032].

Probe rate by stream decile (mean over orders): 1.00, 1.00, 0.99, 0.93, 0.89, 0.88, 0.86, 0.85, 0.84, 0.84

**Drift order (secondary):** in-distribution part skip rate 0.095, risk 0.013; shifted part skip rate 0.132, risk 0.048 (always-probe risk there 0.008).

| split (seed-0 order) | n | skip rate | generation risk | certificate rejects |
|---|---|---|---|---|
| id_new_12 | 100 | 0.11 | 0.18 | 0.54 |
| id_new_13 | 100 | 0.11 | 0.10 | 0.47 |
| id_new_14 | 100 | 0.13 | 0.11 | 0.49 |
| id_new_15 | 100 | 0.12 | 0.14 | 0.45 |
| id_new_16 | 100 | 0.17 | 0.15 | 0.47 |
| id_new_17 | 100 | 0.07 | 0.13 | 0.58 |
| id_new_18 | 100 | 0.14 | 0.17 | 0.58 |
| id_new_19 | 100 | 0.10 | 0.13 | 0.51 |
| id_new_20 | 100 | 0.17 | 0.09 | 0.35 |
| id_new_21 | 100 | 0.15 | 0.14 | 0.49 |
| ood2_E_high | 100 | 0.20 | 0.06 | 0.36 |
| ood2_E_low | 99 | 0.00 | 0.79 | 0.97 |
| ood2_drop | 100 | 0.00 | 0.97 | 0.97 |
| ood2_force | 99 | 0.00 | 1.00 | 1.00 |
| ood_E_high | 50 | 0.16 | 0.10 | 0.42 |
| ood_E_low | 50 | 0.00 | 0.64 | 0.88 |
| ood_force_x3 | 50 | 0.02 | 0.72 | 0.88 |
| ood_gravity_drop | 50 | 0.00 | 1.00 | 1.00 |
| test_id | 100 | 0.17 | 0.22 | 0.52 |
