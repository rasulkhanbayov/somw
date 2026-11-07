# Phase 4j: held-out confirmation of V2 + shuffled-restart ablation (pre-registered Oct 13, 2026)

Base calibration pool: 1200 queries, 100 objects, 500 object splits (the Phase 4f seeds). Values: mean [2.5%, 97.5%] over splits, object-weighted. Reference 4.666 s, generation 0.236 s, probe fit 0.047 s/window + 0.2024 ms/substep (18898 A100 windows).

Blown shadow/suffix restarts (fallback to reference): 0 of 6000; crashed queries: 0; queries whose sample-0 error differs from the 4f error: 0.

Check: cert_ref risk 0.0228, cost ratio 0.5589 (Phase 4f: 0.022753333333333334, 0.5589031740648963).

- **J1 risk:** V2 0.032 [0.002, 0.068] (target <= 0.11); V2 - cert_ref 0.009 [0.000, 0.020] (target mean <= 0.02): **PASS**
- **J2 compute:** cost(V2) / cost(cert_ref) 0.888 [0.810, 1.034]: **FAIL**
- **J3 localisation:** error(shuffled) - error(V2) 0.003 [0.001, 0.006]: **PASS**
- **Confirmation: NOT CONFIRMED**; timing claim: **not supported**

| pipeline | mean error | risk | cost (s/query) | cost / always-reference |
|---|---|---|---|---|
| generate only | 0.071 [0.055, 0.087] | 0.131 [0.085, 0.172] | 0.236 [0.236, 0.236] | 0.05 |
| coarse_cfl only (no generation) | 0.029 [0.021, 0.037] | 0.042 [0.010, 0.072] | 0.577 [0.575, 0.579] | 0.12 |
| certify-or-simulate (4f) | 0.025 [0.008, 0.041] | 0.023 [0.000, 0.055] | 2.608 [1.644, 3.962] | 0.56 |
| **V2 (certified prefix + shadow suffix)** | 0.034 [0.017, 0.048] | 0.032 [0.002, 0.068] | 2.282 [1.678, 3.276] | 0.49 |
| V2, shuffled restarts (ablation) | 0.037 [0.020, 0.052] | 0.041 [0.007, 0.083] | 2.281 [1.678, 3.276] | 0.49 |
| V2, earliest restart k = 3 | 0.027 [0.010, 0.043] | 0.027 [0.000, 0.065] | 2.743 [1.907, 3.925] | 0.59 |
| always reference | 0 | 0 | 4.666 | 1.00 |

Chain decisions on the test halves: accept 0.532 [0.219, 0.757], anchor fails 0.041 [0.003, 0.163], suffix 0.426 [0.235, 0.630]; lambda 0.030 [0.016, 0.045].
Coarse-only mean error caps non-finite rollouts at 10 for the mean (they count as failures in the risk).
