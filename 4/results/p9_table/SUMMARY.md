# Phase 9 main table (descriptive; object-weighted; fresh Phase 9 data)

Risk = P(error > 0.1); cost relative to one exact rollout (4.666 A100 s). Bold in the LaTeX table: the cheapest pipeline with risk <= 0.03 per benchmark: in distribution: always M1, shift: ours, contact: always M1.

## in distribution

| pipeline | risk (bad / n) | mean error | cost |
|---|---|---|---|
| generate only | 0.130 (26/200, 100 objects) | 0.0739 | 0.05 |
| coarse only | 0.040 (8/200, 100 objects) | 0.0315 | 0.13 |
| always M1 | 0.015 (3/200, 100 objects) | 0.0138 | 0.13 |
| always M2 | 0.020 (4/200, 100 objects) | 0.0110 | 0.25 |
| always M3 | 0.015 (3/200, 100 objects) | 0.0132 | 0.50 |
| seed-spread deferral | 0.000 (0/200, 100 objects) | 0.0084 | 1.11 |
| momentum-residual gate | 0.010 (2/200, 100 objects) | 0.0108 | 0.77 |
| certify or simulate | 0.015 (3/200, 100 objects) | 0.0220 | 0.54 |
| always reference | 0.000 (0/200, 100 objects) | 0.0000 | 1.00 |
| ours | 0.020 (4/200, 100 objects) | 0.0315 | 0.26 |

## shift

| pipeline | risk (bad / n) | mean error | cost |
|---|---|---|---|
| generate only | 0.680 (272/400, 100 objects) | 0.3507 | 0.05 |
| coarse only | 0.367 (147/400, 100 objects) | 0.1184 | 0.16 |
| always M1 | 0.102 (41/400, 100 objects) | 0.0488 | 0.20 |
| always M2 | 0.102 (41/400, 100 objects) | 0.0479 | 0.29 |
| always M3 | 0.155 (62/400, 100 objects) | 0.0759 | 0.51 |
| seed-spread deferral | 0.247 (99/400, 100 objects) | 0.1265 | 1.06 |
| momentum-residual gate | 0.007 (3/400, 100 objects) | 0.0039 | 0.97 |
| certify or simulate | 0.003 (1/400, 100 objects) | 0.0078 | 0.87 |
| always reference | 0.000 (0/400, 100 objects) | 0.0000 | 1.00 |
| ours | 0.013 (5/400, 100 objects) | 0.0165 | 0.71 |

## contact

| pipeline | risk (bad / n) | mean error | cost |
|---|---|---|---|
| generate only | 0.693 (87/132, 81 objects) | 0.1876 | 0.10 |
| certify or simulate | 0.000 (0/132, 81 objects) | 0.0003 | 1.18 |
| always reference | 0.000 (0/132, 81 objects) | 0.0000 | 1.00 |
| ours | 0.012 (1/132, 81 objects) | 0.0147 | 0.90 |
| chain-timed repair (4k timing) | 0.012 (1/132, 81 objects) | 0.0291 | 0.67 |
| always M1 | 0.029 (4/132, 81 objects) | 0.0238 | 0.13 |
| always M2 | 0.035 (5/132, 81 objects) | 0.0255 | 0.25 |
| always M3 | 0.000 (0/132, 81 objects) | 0.0204 | 0.50 |

## Certificate on the generations (descriptive)

| benchmark | bad accepted | good accepted |
|---|---|---|
| in distribution | 3/26 | 105/174 |
| shift | 1/272 | 84/128 |
| contact | 0/87 | 1/45 |

