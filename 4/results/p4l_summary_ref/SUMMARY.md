# Phase 4l: multi-object contact benchmark (pre-registered Oct 15, 2026)

Valid scenes: 334 (255 contact, 79 control); designed 400; excluded: pre-contact deviation 0.1638 > 0.02: 1, no valid placement: 19, pre-contact deviation 0.0239 > 0.02: 1, pre-contact deviation 0.1352 > 0.02: 1, pre-contact deviation 0.0293 > 0.02: 1, pre-contact deviation 0.0267 > 0.02: 1, pre-contact deviation 0.1696 > 0.02: 1, pre-contact deviation 0.1093 > 0.02: 1, pre-contact deviation 0.1482 > 0.02: 1, pre-contact deviation 0.0349 > 0.02: 1, pre-contact deviation 0.0230 > 0.02: 1, pre-contact deviation 0.2337 > 0.02: 1, pre-contact deviation 0.1385 > 0.02: 1, pre-contact deviation 0.0232 > 0.02: 1, pre-contact deviation 0.1194 > 0.02: 1, pre-contact deviation 0.0304 > 0.02: 1, pre-contact deviation 0.1135 > 0.02: 1, pre-contact deviation 0.0813 > 0.02: 1, pre-contact deviation 0.0704 > 0.02: 1, pre-contact deviation 0.0857 > 0.02: 1, pre-contact deviation 0.0861 > 0.02: 1, pre-contact deviation 0.0419 > 0.02: 1, pre-contact deviation 0.0223 > 0.02: 2, pre-contact deviation 0.0445 > 0.02: 1, pre-contact deviation 0.0263 > 0.02: 1, pre-contact deviation 0.0602 > 0.02: 1, pre-contact deviation 0.0210 > 0.02: 1, pre-contact deviation 0.0979 > 0.02: 1, pre-contact deviation 0.1008 > 0.02: 1, pre-contact deviation 0.0375 > 0.02: 1, pre-contact deviation 0.0397 > 0.02: 1, pre-contact deviation 0.1457 > 0.02: 1, pre-contact deviation 0.0334 > 0.02: 1, pre-contact deviation 0.0643 > 0.02: 1, pre-contact deviation 0.0420 > 0.02: 1, pre-contact deviation 0.0202 > 0.02: 1, pre-contact deviation 0.0526 > 0.02: 1, pre-contact deviation 0.2492 > 0.02: 1, pre-contact deviation 0.0228 > 0.02: 1, pre-contact deviation 0.0323 > 0.02: 1, pre-contact deviation 0.0205 > 0.02: 1, pre-contact deviation 0.5636 > 0.02: 1, pre-contact deviation 0.0389 > 0.02: 1, pre-contact deviation 0.4579 > 0.02: 1, pre-contact deviation 0.0627 > 0.02: 1, pre-contact deviation 0.3163 > 0.02: 1, pre-contact deviation 0.0742 > 0.02: 1. tau = 0.02894; eps = 0.1. Costs in A100 s (reference 4.666; generation 0.472 for two objects).

- **H0 lift floor:** median floor 0.0100 vs 0.25 x median base defect 0.0162 at the contact window (231 scenes with an approach): **PASS**
- **H1 placement (capture, base):** contact_spread - uniform -0.1114 [-0.1531, -0.0699]; contact_spread - spread_event -0.1165 [-0.1506, -0.0817]: **FAIL** (gt_super, secondary: vs uniform -0.0764 [-0.1225, -0.0311])
- **H2 certificate:** FAR(contact_spread) 0.024 (<= 0.15) over 188 bad scenes; FAR(uniform) - FAR(contact_spread) -0.021 [-0.050, -0.003]: **FAIL**; controls falsely rejected: 0.493
- **H3a event-gated V2 vs cert_ref:** risk 0.087 vs 0.016 (limit +0.02); saving +1.3897 [+1.2610, +1.5245] s: **FAIL**
- **H3b timing:** chain - event -0.0030 [-0.0098, +0.0037]; shuffled - event +0.0062 [+0.0025, +0.0100]: **FAIL**

| pipeline (contact scenes) | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only (two objects) | 0.2023 | 0.731 | 0.10 |
| certify or simulate | 0.0048 | 0.016 | 1.62 |
| event-gated V2 | 0.0453 | 0.087 | 1.32 |
| chain-gated V2 (4k timing) | 0.0423 | 0.082 | 1.79 |
| event-gated V2, shuffled timing | 0.0515 | 0.103 | 1.32 |

Mean capture (base): contact_spread 0.731, uniform 0.830, spread_event 0.839
Mean error: base 0.2046, gt_super 0.1739 (the ignored interaction alone).

## Restart curve (exploratory): mean error of V2 by restart frame relative to the generated approach k_g

| k - k_g | n | mean error |
|---|---|---|
| -10 | 89 | 0.0146 |
| -9 | 103 | 0.0161 |
| -8 | 118 | 0.0178 |
| -7 | 139 | 0.0193 |
| -6 | 156 | 0.0219 |
| -5 | 164 | 0.0265 |
| -4 | 178 | 0.0310 |
| -3 | 189 | 0.0358 |
| -2 | 192 | 0.0436 |
| -1 | 196 | 0.0516 |
| +0 | 199 | 0.0605 |
| +1 | 230 | 0.0603 |
| +2 | 230 | 0.0673 |
| +3 | 230 | 0.0738 |
| +4 | 227 | 0.0785 |
| +5 | 225 | 0.0856 |
| +6 | 210 | 0.0939 |
