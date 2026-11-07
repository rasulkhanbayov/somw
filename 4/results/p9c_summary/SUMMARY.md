# Phase 9 contact: three-window certificate, shadow-timed switch, M3 suffix on fresh scenes (pre-registered Oct 19, 2026; amended before data)

Valid scenes: 168 (132 contact, 36 control); designed 200, excluded 32. tau = 0.02614; eps = 0.1; d_s = 0.46875. Restarts missing (charged as reference): 0. EAF restart relative to the generated approach: median -4.0 frames.

- **CC1 risk:** SF 0.012 vs cert_ref 0.000 (limit +0.02): **PASS**
- **CC2 cost:** saving cert_ref - SF +1.3103 [+1.0286, +1.6011] s/query: **PASS**
- **CC3 timing:** error(chain) - error(SF) +0.0144 [+0.0079, +0.0199]: **PASS**
- **Confirmation: CONFIRMED**; fidelity timing beats 4k timing: **supported**

| pipeline (contact scenes) | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only (two objects) | 0.1876 | 0.693 | 0.10 |
| certify or simulate | 0.0003 | 0.000 | 1.18 |
| EAF repair (generation-timed) | 0.0154 | 0.025 | 0.90 |
| **SF repair (shadow-timed)** | 0.0147 | 0.012 | 0.90 |
| chain-gated V2 (4k timing) | 0.0291 | 0.012 | 0.67 |
| event-timed V2 (secondary) | 0.0510 | 0.150 | 0.62 |
| always simulate | 0.0000 | 0.000 | 1.00 |

Controls: 36; generation risk 0.167; falsely rejected among good 0.467.
