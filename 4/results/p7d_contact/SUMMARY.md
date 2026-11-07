# Phase 7d: the missing baselines on the contact benchmark (pre-registered Oct 17, 2026)

Contact scenes: 270 (4o); controls: 76. Coarse rollouts crashed or blown (counted as failures for coarse only and rejected by XF): 0. Thresholds from the single-object calibration pool: agreement 0.084902, seed spread 0.012935, residual 0.012569; window tau as 4o. CIs 0.9833 two-sided (Bonferroni over 3); cluster bootstrap over the first object.

- **D1** risk(coarse only) - risk(SF): +0.306 [+0.240, +0.381]: **PASS**
- **D2** risk(seed spread) - risk(SF): +0.074 [+0.026, +0.128]: **PASS**
- **D3** risk(residual gate) - risk(SF): +0.278 [+0.209, +0.352]: **PASS**
- **XF rule:** saving SF - XF -0.837 [-1.000, -0.677] s/scene; XF does not dominate SF

| pipeline (contact scenes) | mean error | risk | cost / always-simulate | accept rate |
|---|---|---|---|---|
| PhysCtrl (two objects composed) | 0.1898 | 0.753 | 0.10 | — |
| coarse simulation only | 0.0982 | 0.318 | 0.12 | — |
| seed-spread deferral (8 samples) | 0.0197 | 0.086 | 1.66 | 0.14 |
| momentum-residual gate | 0.0724 | 0.290 | 0.69 | 0.43 |
| accept coarse if it agrees (XF) | 0.0000 | 0.000 | 1.22 | 0.00 |
| certify or simulate | 0.0002 | 0.000 | 1.18 | — |
| **ours: certify + repair, shadow-timed switch (SF)** | 0.0108 | 0.012 | 1.05 | — |
| always simulate | 0.0000 | 0.000 | 1.00 | — |

Controls (76 scenes, objects never touch; query-level means): PhysCtrl (two objects composed) risk 0.105, cost 0.10; coarse simulation only risk 0.026, cost 0.12; seed-spread deferral (8 samples) risk 0.000, cost 1.66; momentum-residual gate risk 0.026, cost 0.63; accept coarse if it agrees (XF) risk 0.000, cost 0.72; certify or simulate risk 0.000, cost 0.66; ours: certify + repair, shadow-timed switch (SF) risk 0.000, cost 0.64; always simulate risk 0.000, cost 1.00
