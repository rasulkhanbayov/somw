# Phase 4o: shadow-timed fidelity switch on fresh contact scenes (pre-registered Oct 16, 2026)

Valid scenes: 346 (270 contact, 76 control); designed 400, excluded 54. tau = 0.02614; eps = 0.1; d_s = 0.46875. Restarts missing (charged as reference): 0. EAF restart relative to the generated approach: median -4.0 frames.

- **O1 risk:** SF 0.012 vs cert_ref 0.000 (limit +0.02): **PASS**
- **O2 cost:** saving cert_ref - SF +0.6486 [+0.5191, +0.7769] s/query: **PASS**
- **O3 timing:** error(chain) - error(SF) +0.0155 [+0.0111, +0.0199]: **PASS**
- **Confirmation: CONFIRMED**; fidelity timing beats 4k timing: **supported**

| pipeline (contact scenes) | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only (two objects) | 0.1898 | 0.753 | 0.10 |
| certify or simulate | 0.0002 | 0.000 | 1.18 |
| EAF repair (generation-timed) | 0.0114 | 0.014 | 1.04 |
| **SF repair (shadow-timed)** | 0.0108 | 0.012 | 1.05 |
| chain-gated V2 (4k timing) | 0.0263 | 0.015 | 1.01 |
| event-timed V2 (secondary) | 0.0465 | 0.131 | 0.88 |
| always simulate | 0.0000 | 0.000 | 1.00 |

Controls: 76; generation risk 0.105; falsely rejected among good 0.412.
