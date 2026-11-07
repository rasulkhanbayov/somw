# Phase 14a pilot: simulator distillation (pre-registered Oct 22, 2026; development data, exploratory)

| model | shift-dev risk | shift-dev mean error | ID-dev risk | ID-dev mean error | ood2_E_low risk | ood2_E_high risk | ood2_force risk |
|---|---|---|---|---|---|---|---|
| base | 0.613 | 0.2909 | 0.155 | 0.0751 | 0.788 | 0.060 | 1.000 |
| ACT_F | 0.853 | 0.3822 | 0.680 | 0.1483 | 0.939 | 0.620 | 1.000 |
| ACT_L | 0.823 | 0.3386 | 0.510 | 0.1318 | 0.919 | 0.550 | 1.000 |
| RND_F | 0.857 | 0.3635 | 0.675 | 0.1443 | 0.960 | 0.610 | 1.000 |
| RND_L | 0.800 | 0.3040 | 0.480 | 0.1266 | 0.889 | 0.510 | 1.000 |

- Kill rule: ACT_F fails, ACT_L fails (shift-dev risk <= 0.7 x base 0.613 = 0.429; ID-dev mean <= base 0.0751 + 0.005).
- **Verdict: KILL**
- ACT vs RND (descriptive): shift-dev risk F 0.853 vs 0.857; L 0.823 vs 0.800.
