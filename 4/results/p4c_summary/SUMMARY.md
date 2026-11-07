# Phase 4c: conditioning repair (pre-registered Oct 11, 2026)

Joined with Phase 4b on 599 (query, model) pairs. lambda = 0.03007. Errors: mean per-point L2 to ground truth over 24 frames.

- **Primary**: conditioning repair beats replacement on rejected base trajectories: FAIL


## base, rejected by the certificate (PRIMARY) (n = 198)

| R0 original | R1 replace | R2 constrained | R4id re-sample | **R4 conditioning** |
|---|---|---|---|---|
| 0.2671 | 0.2300 | 0.2276 | 0.2674 | 0.5105 |

R1 - R4 -0.2804 [-0.3205,-0.2435]; R2 - R4 -0.2828 [-0.3229,-0.2461]; R0 - R4 -0.2433 [-0.2850,-0.2055]; R4id - R4 -0.2431 [-0.2852,-0.2049]

Held-out defect (uniform, 'ref', m = 4; never used in selection): median R0 0.1312 -> R4 0.0909; lower for R4 in 0.73 of trajectories.
Accepted at lambda after repair with the certificate's own signal (selection-biased): R4 0.12.

Most-chosen adjustments per split (count: dlog10E, force scale): ood_E_high: 4: (-1, x1), 3: (+1, x0.75), 2: (+1, x1); ood_E_low: 14: (+1, x0.33), 6: (+0.5, x0.33), 4: (+1, x0.75); ood_force_x3: 5: (+1, x0.75), 5: (+1, x0.5), 5: (+0, x0.75); ood_gravity_drop: 6: (+1, x0.33), 6: (+0.5, x0.33), 6: (-1, x0.33); test_id: 6: (+1, x0.75), 6: (+1, x0.5), 6: (+0.5, x0.75)

## base, all (n = 300)

| R0 original | R1 replace | R2 constrained | R4id re-sample | **R4 conditioning** |
|---|---|---|---|---|
| 0.1901 | 0.1647 | 0.1630 | 0.1908 | 0.3632 |

R1 - R4 -0.1985 [-0.2291,-0.1718]; R2 - R4 -0.2002 [-0.2310,-0.1734]; R0 - R4 -0.1731 [-0.2034,-0.1460]; R4id - R4 -0.1724 [-0.2030,-0.1453]

Held-out defect (uniform, 'ref', m = 4; never used in selection): median R0 0.0671 -> R4 0.0616; lower for R4 in 0.64 of trajectories.
Accepted at lambda after repair with the certificate's own signal (selection-biased): R4 0.41.

Most-chosen adjustments per split (count: dlog10E, force scale): ood_E_high: 9: (-1, x1), 8: (+0.5, x1), 6: (+1, x1); ood_E_low: 14: (+1, x0.33), 6: (+0.5, x0.33), 4: (+1, x0.75); ood_force_x3: 6: (+1, x0.75), 5: (+1, x0.5), 5: (+0, x0.75); ood_gravity_drop: 6: (+1, x0.33), 6: (+0.5, x0.33), 6: (-1, x0.33); test_id: 10: (+1, x1), 10: (+0.5, x1), 10: (+0.5, x0.75)

## large, all (n = 299)

| R0 original | R1 replace | R2 constrained | R4id re-sample | **R4 conditioning** |
|---|---|---|---|---|
| 0.1931 | 0.1855 | 0.1863 | 0.1988 | 0.4111 |

R1 - R4 -0.2256 [-0.2579,-0.1952]; R2 - R4 -0.2248 [-0.2572,-0.1943]; R0 - R4 -0.2179 [-0.2493,-0.1879]; R4id - R4 -0.2123 [-0.2431,-0.1830]

Held-out defect (uniform, 'ref', m = 4; never used in selection): median R0 0.0750 -> R4 0.0704; lower for R4 in 0.64 of trajectories.

Most-chosen adjustments per split (count: dlog10E, force scale): ood_E_high: 7: (-0.5, x0.75), 6: (+1, x0.75), 6: (+0.5, x1); ood_E_low: 13: (-1, x0.33), 10: (+1, x0.33), 4: (+1, x0.5); ood_force_x3: 7: (+1, x0.5), 7: (-1, x0.33), 4: (+1, x0.33); ood_gravity_drop: 5: (+0, x1.5), 4: (+0, x2), 3: (+1, x1); test_id: 18: (+1, x0.75), 14: (+0.5, x0.75), 11: (-0.5, x0.75)
