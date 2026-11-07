# Phase 4e: anchored conditioning repair (pre-registered Oct 12, 2026)

Joined with Phase 4b on 600 pairs. lambda = 0.03007. Errors: mean per-point L2 to ground truth.

- **Primary** (L = 8): anchored repair beats replacement on rejected base trajectories: FAIL


## base, rejected by the certificate (PRIMARY) (n = 198); anchor's own error 0.0354; non-finite repaired trajectories (scored +inf): 0

| L | R0 original | R1 replace (4b) | R5 chosen | R6 chosen + anchor | R1 - R6 [95% CI] | frames >= L: R0 | R5 | R0 - R5 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.2671 | 0.2300 | 0.2679 | 0.2416 | -0.0116 [-0.0162,-0.0074] | 0.3505 | 0.3519 | -0.0014 [-0.0059,+0.0024] |
| 12 | 0.2671 | 0.2300 | 0.2655 | 0.2162 | +0.0138 [+0.0082,+0.0199] | 0.3984 | 0.3971 | +0.0013 [-0.0034,+0.0055] |

Most-chosen adjustments (L = 8): ood_E_high: 6: (+1, x1), 3: (+0.5, x1), 3: (+0, x1); ood_E_low: 18: (+0, x1), 14: (-0.5, x1), 8: (+0.5, x1); ood_force_x3: 20: (+1, x1), 10: (+0.5, x1), 7: (+0, x1); ood_gravity_drop: 17: (-1, x0.33), 16: (+1, x0.33), 7: (+0, x1); test_id: 17: (+1, x1), 15: (+0, x1), 8: (+0.5, x1)

## base, all (n = 300); anchor's own error 0.0259; non-finite repaired trajectories (scored +inf): 0

| L | R0 original | R1 replace (4b) | R5 chosen | R6 chosen + anchor | R1 - R6 [95% CI] | frames >= L: R0 | R5 | R0 - R5 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.1901 | 0.1647 | 0.1910 | 0.1728 | -0.0081 [-0.0113,-0.0052] | 0.2501 | 0.2517 | -0.0016 [-0.0044,+0.0012] |
| 12 | 0.1901 | 0.1647 | 0.1894 | 0.1552 | +0.0095 [+0.0054,+0.0135] | 0.2846 | 0.2845 | +0.0001 [-0.0029,+0.0031] |

Most-chosen adjustments (L = 8): ood_E_high: 25: (+1, x1), 10: (-0.5, x1), 6: (+0, x1); ood_E_low: 22: (+0, x1), 14: (-0.5, x1), 11: (+0.5, x1); ood_force_x3: 23: (+1, x1), 11: (+0.5, x1), 9: (+0, x1); ood_gravity_drop: 17: (-1, x0.33), 16: (+1, x0.33), 7: (+0, x1); test_id: 42: (+1, x1), 26: (+0, x1), 17: (+0.5, x1)

## large, all (n = 300); anchor's own error 0.0259; non-finite repaired trajectories (scored +inf): 0

| L | R0 original | R1 replace (4b) | R5 chosen | R6 chosen + anchor | R1 - R6 [95% CI] | frames >= L: R0 | R5 | R0 - R5 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.1943 | 0.1866 | 0.2077 | 0.1980 | -0.0114 [-0.0173,-0.0060] | 0.2694 | 0.2894 | -0.0200 [-0.0291,-0.0116] |
| 12 | 0.1943 | 0.1866 | 0.1991 | 0.1755 | +0.0111 [+0.0063,+0.0162] | 0.3154 | 0.3251 | -0.0097 [-0.0181,-0.0010] |

Most-chosen adjustments (L = 8): ood_E_high: 14: (+1, x1), 12: (-1, x1), 11: (+0, x1); ood_E_low: 18: (+0, x1), 15: (-0.5, x1), 7: (+0, x1.5); ood_force_x3: 13: (+0, x1), 10: (+1, x1), 8: (-1, x1); ood_gravity_drop: 8: (+0, x0.33), 7: (+0, x0.75), 6: (+1, x0.33); test_id: 25: (+0, x1), 23: (+1, x1), 14: (+0.5, x1)
