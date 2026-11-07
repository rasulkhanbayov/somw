# Pilot 8b: a fused certificate score (EXPLORATORY, development data; defined Oct 17, 2026)

Calibration pool 1200 queries (100 objects) split into thirds by object (fit / calibrate / evaluate); 2c shift splits 200 queries. eps = 0.1, alpha = 0.1.

| score | threshold | bad objects (cal) | good accepted (eval) | false accept (eval) | rejected (eval) | false accept (2c shift) | good accepted (2c shift) |
|---|---|---|---|---|---|---|---|
| window only | 0.02089 | 16 | 0.503 | 0.115 | 0.548 | 0.008 | 0.416 |
| fused | 0.002648 | 16 | 0.555 | 0.346 | 0.472 | 0.008 | 0.390 |

Good-acceptance gain +0.052; shift false-accept change +0.000. **Promising: NO** (rule: gain >= 0.10, eval FAR <= 0.10, shift FAR change <= +0.02).

Largest standardised coefficients: residual +1.29, win2 +1.26, max window +1.26, win0 +0.66, drag fraction +0.54, log10 E +0.47

Exploratory pilot on development data; one object split; nothing here is a confirmation.
