# Phase 14a diagnosis (Oct 24, 2026; EXPLORATORY, after the KILL verdict)

## Wall survey (reference within 0.5 of the domain boundary)

| set | n | reach a wall | median first model frame |
|---|---|---|---|
| 14a pool (all) | 2950 | 74.4% | 15.0 |
| 14a ACT arm | 2619 | 83.8% | 15.0 |
| replay (4g ID re-sims, all 3,000) | 3000 | 2.9% | 20.0 |
| Phase 9 ood3_E_low | 100 | 32.0% | 16.5 |
| Phase 9 ood3_E_high | 100 | 1.0% | 15.0 |
| Phase 9 ood3_force | 100 | 99.0% | 15.0 |
| Phase 9 ood3_drop | 100 | 0.0% | — |
| Phase 9 id_new_22 | 100 | 3.0% | 18.0 |
| Phase 9 id_new_23 | 100 | 4.0% | 20.5 |
| dev 4m ood2_E_low | 99 | 30.3% | 16.5 |
| dev 4m ood2_E_high | 100 | 3.0% | 23.0 |
| dev 4m ood2_force | 99 | 100.0% | 14.0 |

## Motion split: ID-dev (first 120)

| model | error | travel gen/ref | deformation gen/ref | centre-of-mass error | shape error | frame-1 error | acceleration gen/ref |
|---|---|---|---|---|---|---|---|
| base | 0.0819 | 1.00 | 0.97 | 0.0160 | 0.0775 | 0.0082 | 1.17 |
| ACT_F | 0.1558 | 0.94 | 1.00 | 0.0894 | 0.1188 | 0.0255 | 1.50 |
| ACT_L | 0.1398 | 0.95 | 0.97 | 0.0722 | 0.1146 | 0.0290 | 1.58 |
| RND_F | 0.1515 | 1.00 | 1.04 | 0.0880 | 0.1163 | 0.0396 | 1.54 |
| RND_L | 0.1365 | 0.99 | 1.01 | 0.0611 | 0.1167 | 0.0331 | 1.55 |

## Motion split: ood2_E_high

| model | error | travel gen/ref | deformation gen/ref | centre-of-mass error | shape error | frame-1 error | acceleration gen/ref |
|---|---|---|---|---|---|---|---|
| base | 0.0639 | 1.00 | 0.96 | 0.0152 | 0.0601 | 0.0084 | 1.18 |
| ACT_F | 0.1409 | 0.95 | 1.02 | 0.0829 | 0.1067 | 0.0247 | 1.45 |
| ACT_L | 0.1298 | 0.93 | 0.99 | 0.0745 | 0.1001 | 0.0290 | 1.56 |
| RND_F | 0.1379 | 0.98 | 1.04 | 0.0850 | 0.1029 | 0.0406 | 1.50 |
| RND_L | 0.1205 | 0.97 | 1.03 | 0.0580 | 0.1014 | 0.0324 | 1.54 |

## Motion split: ood2_E_low

| model | error | travel gen/ref | deformation gen/ref | centre-of-mass error | shape error | frame-1 error | acceleration gen/ref |
|---|---|---|---|---|---|---|---|
| base | 0.2508 | 1.00 | 0.88 | 0.0423 | 0.2358 | 0.0095 | 1.09 |
| ACT_F | 0.2619 | 0.92 | 0.78 | 0.1025 | 0.2367 | 0.0299 | 1.32 |
| ACT_L | 0.2503 | 0.92 | 0.76 | 0.1064 | 0.2258 | 0.0332 | 1.42 |
| RND_F | 0.2476 | 0.95 | 0.86 | 0.1143 | 0.2129 | 0.0414 | 1.38 |
| RND_L | 0.2398 | 0.94 | 0.85 | 0.1027 | 0.2140 | 0.0376 | 1.41 |
