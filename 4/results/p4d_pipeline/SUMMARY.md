# Phase 4d: certify-or-simulate (pre-registered Oct 12, 2026)

Costs in A100 seconds per query. Reference rollout (PhysCtrl setting, nosync): 4.666 s. Generation per sample (best batch / batch 1): base 0.236 / 0.300, large 4.520 / 4.696. eps = 0.1.


## base_defect (base; score = defect; lambda = 0.02614; n = 300; fallback rate 0.70)

| pipeline | mean error | risk (err > eps) | cost (s/query) |
|---|---|---|---|
| gen | 0.1901 | 0.483 | 0.236 |
| random | 0.1009 | 0.284 | 0.738 |
| cert | 0.0690 | 0.207 | 1.135 |
| oracle | 0.0675 | 0.207 | 0.713 |
| sim_fb | 0.0633 | 0.200 | 0.714 |
| sim_ref | 0.0000 | 0.000 | 4.666 |

T1 random - cert mean error: +0.0320 [+0.0262, +0.0381] (PASS)
T3 break-even simulator cost: cert is cheaper than always-simulate when one rollout costs more than 2.208 s; here sim_fb costs 0.714 s and sim_ref 4.666 s (per-query generation 0.236 s, probes 0.419 s).

T4 sweep (lambda at score quantiles): fallback rate / cost / mean error / risk
| quantile | fallback | cost | error cert | error random | risk cert | risk random |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 1.369 | 0.0633 | 0.0633 | 0.200 | 0.200 |
| 0.20 | 0.80 | 1.211 | 0.0665 | 0.0887 | 0.203 | 0.257 |
| 0.40 | 0.60 | 1.051 | 0.0723 | 0.1140 | 0.213 | 0.313 |
| 0.50 | 0.50 | 0.981 | 0.0779 | 0.1267 | 0.233 | 0.342 |
| 0.60 | 0.40 | 0.921 | 0.0850 | 0.1394 | 0.273 | 0.370 |
| 0.70 | 0.30 | 0.862 | 0.0947 | 0.1521 | 0.327 | 0.398 |
| 0.80 | 0.20 | 0.807 | 0.1128 | 0.1648 | 0.383 | 0.427 |
| 0.90 | 0.10 | 0.725 | 0.1387 | 0.1774 | 0.443 | 0.455 |
| 1.00 | 0.00 | 0.657 | 0.1885 | 0.1897 | 0.483 | 0.482 |

## base_seed_spread (base; score = seed_spread; lambda = 0.01577; n = 300; fallback rate 0.61)

| pipeline | mean error | risk (err > eps) | cost (s/query) |
|---|---|---|---|
| gen | 0.1901 | 0.483 | 1.889 |
| random | 0.1128 | 0.310 | 2.324 |
| cert | 0.1243 | 0.287 | 2.344 |
| oracle | 0.0703 | 0.207 | 2.303 |
| sim_fb | 0.0633 | 0.200 | 0.714 |
| sim_ref | 0.0000 | 0.000 | 4.666 |

T1 random - cert mean error: -0.0115 [-0.0238, +0.0002] (FAIL)
T3 break-even simulator cost: cert is cheaper than always-simulate when one rollout costs more than 4.843 s; here sim_fb costs 0.714 s and sim_ref 4.666 s (per-query generation 1.889 s, probes 0.000 s).

T4 sweep (lambda at score quantiles): fallback rate / cost / mean error / risk
| quantile | fallback | cost | error cert | error random | risk cert | risk random |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 2.603 | 0.0633 | 0.0633 | 0.200 | 0.200 |
| 0.20 | 0.80 | 2.484 | 0.1162 | 0.0887 | 0.277 | 0.257 |
| 0.40 | 0.60 | 2.332 | 0.1245 | 0.1140 | 0.287 | 0.313 |
| 0.50 | 0.50 | 2.252 | 0.1268 | 0.1267 | 0.287 | 0.342 |
| 0.60 | 0.40 | 2.179 | 0.1313 | 0.1394 | 0.307 | 0.370 |
| 0.70 | 0.30 | 2.109 | 0.1365 | 0.1521 | 0.323 | 0.398 |
| 0.80 | 0.20 | 2.048 | 0.1445 | 0.1648 | 0.367 | 0.427 |
| 0.90 | 0.10 | 1.989 | 0.1602 | 0.1774 | 0.447 | 0.455 |
| 1.00 | 0.00 | 1.891 | 0.1888 | 0.1897 | 0.483 | 0.482 |

## large_seed_spread (large; score = seed_spread; lambda = 0.02979; n = 300; fallback rate 0.72)

| pipeline | mean error | risk (err > eps) | cost (s/query) |
|---|---|---|---|
| gen | 0.1943 | 0.667 | 36.162 |
| random | 0.0995 | 0.329 | 36.678 |
| cert | 0.0757 | 0.230 | 36.691 |
| oracle | 0.0670 | 0.197 | 36.695 |
| sim_fb | 0.0633 | 0.200 | 0.714 |
| sim_ref | 0.0000 | 0.000 | 4.666 |

T1 random - cert mean error: +0.0238 [+0.0176, +0.0305] (PASS)
T3 break-even simulator cost: cert is cheaper than always-simulate when one rollout costs more than 130.705 s; here sim_fb costs 0.714 s and sim_ref 4.666 s (per-query generation 36.162 s, probes 0.000 s).

T4 sweep (lambda at score quantiles): fallback rate / cost / mean error / risk
| quantile | fallback | cost | error cert | error random | risk cert | risk random |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 36.876 | 0.0633 | 0.0633 | 0.200 | 0.200 |
| 0.20 | 0.80 | 36.743 | 0.0701 | 0.0895 | 0.203 | 0.293 |
| 0.40 | 0.60 | 36.602 | 0.0851 | 0.1157 | 0.297 | 0.387 |
| 0.50 | 0.50 | 36.525 | 0.0929 | 0.1288 | 0.350 | 0.433 |
| 0.60 | 0.40 | 36.456 | 0.1039 | 0.1419 | 0.420 | 0.480 |
| 0.70 | 0.30 | 36.382 | 0.1150 | 0.1550 | 0.503 | 0.527 |
| 0.80 | 0.20 | 36.320 | 0.1320 | 0.1681 | 0.577 | 0.573 |
| 0.90 | 0.10 | 36.246 | 0.1562 | 0.1812 | 0.630 | 0.620 |
| 1.00 | 0.00 | 36.163 | 0.1935 | 0.1938 | 0.667 | 0.665 |
