# Pilot 8d: certified M1 (EXPLORATORY, development data; defined Oct 18, 2026)

Novelty threshold tau_M1 = 0.017641 (95th percentile rule over 400 in-distribution id_new_0..3 queries). Cost relative to one reference rollout.

| set | n | M1 alone: risk / cost | CM1: risk / cost | CM1 accept rate | AUROC (score vs M1 error > 0.10) |
|---|---|---|---|---|---|
| test_id | 100 | 0.000 / 0.13 | 0.000 / 0.39 | 0.98 | nan |
| shift | 200 | 0.085 / 0.20 | 0.065 / 0.70 | 0.57 | 0.836 |
| 4l contact | 255 | 0.020 / 0.13 | 0.008 / 0.56 | 0.64 | 0.889 |
| 4l control | 79 | 0.000 / 0.13 | 0.000 / 0.38 | 1.00 | nan |

## Same 2c queries, 8a pipelines (risk / mean error / cost)

- in distribution (test_id): gated V2, M2 suffix, M1 shadow (B2): 0.030 / 0.0324 / 0.30
- in distribution (test_id): always simulate M1: 0.000 / 0.0133 / 0.13
- in distribution (test_id): certify or simulate (reference): 0.030 / 0.0192 / 0.66
- shift (4 OOD splits): gated V2, M2 suffix, M1 shadow (B2): 0.010 / 0.0183 / 0.72
- shift (4 OOD splits): always simulate M1: 0.085 / 0.0402 / 0.20
- shift (4 OOD splits): always simulate M2: 0.055 / 0.0301 / 0.29
- shift (4 OOD splits): certify or simulate (reference): 0.005 / 0.0075 / 0.95

## Pre-stated rules

- shift: CM1 risk <= 0.02 at <= 0.50x: **no**
- in distribution: CM1 risk <= 0.01 at <= 0.45x (amended before data): **yes**
- contact: CM1 risk <= 0.02 at <= 0.60x: **yes**
- **Promising: NO**

Exploratory pilot on development data; the novelty threshold bounds false rejections in distribution, not false accepts. Nothing here is a confirmation.
