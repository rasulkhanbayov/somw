# Pilot E2: cheaper verification (EXPLORATORY, development data; defined Oct 18, 2026)

Thresholds: replicate-0 object rule on the calibration pool (variant d: id_new_0..3 only). Evaluation: the 2c pool.

| variant | cal queries | threshold | AUROC (pooled) | false accept (shift) | good accepted (test_id) | probe cost (A100 s) | promising |
|---|---|---|---|---|---|---|---|
| (a) event window only | 1200 | 0.03007 | 0.955 | 0.008 | 0.667 | 0.148 | YES |
| (b) event + middle | 1200 | 0.03007 | 0.959 | 0.008 | 0.667 | 0.295 | no |
| (c) baseline: 3 x m = 4 | 1200 | 0.03007 | 0.958 | 0.008 | 0.641 | 0.443 | — |
| (d) 3 x m = 2 | 400 | 0.01647 | 0.944 | 0.033 | 0.705 | 0.277 | no |

**E2 promising: YES** (rule: cost <= 0.5 x baseline, AUROC >= baseline - 0.02, shift false accept <= 0.10).

Exploratory pilot on development data.
