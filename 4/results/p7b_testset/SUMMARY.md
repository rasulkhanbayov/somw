# Phase 7b: PhysCtrl's own metrics on PhysCtrl's test set (descriptive; pre-registered Oct 17, 2026)

Queries: 100 (test_id; Phase 2c development data, so descriptive only). tau = 0.02614. Accepted 0.48; gated suffix 0.50; missing restarts charged as reference: 0. DeformLoss of the ground truth itself (PhysCtrl's reference value): 0.07438.

| pipeline | position MSE (PhysCtrl eval.py) | DeformLoss (PhysCtrl) | mean point error | risk (> 0.10) | cost / always-simulate |
|---|---|---|---|---|---|
| PhysCtrl (generate only) | 1.487e-03 | 0.1955 | 0.0702 | 0.220 | 0.05 |
| certify or simulate | 3.153e-04 | 0.1026 | 0.0192 | 0.030 | 0.66 |
| **ours: certify + repair** | 4.234e-04 | 0.1351 | 0.0343 | 0.030 | 0.52 |
| always simulate (exact) | 0.000e+00 | 0.07438 | 0.0000 | 0.000 | 1.00 |

Means over queries (eval.py sums per batch; the per-query mean is reported). Position MSE uses PhysCtrl's normalised coordinates, as eval.py's F.mse_loss(output, points_tgt). Lower is better for every column.
