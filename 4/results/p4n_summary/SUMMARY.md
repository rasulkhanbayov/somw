# Phase 4n: event-aware fidelity repair on fresh contact scenes (pre-registered Oct 16, 2026)

Valid scenes: 347 (271 contact, 76 control); designed 400, excluded 53. tau = 0.02614; eps = 0.1; d_s = 0.46875. Restarts missing (charged as reference): 0. EAF restart relative to the generated approach: median -5.0 frames.

- **N1 risk:** EAF 0.034 vs cert_ref 0.010 (limit +0.02): **FAIL**
- **N2 cost:** saving cert_ref - EAF +0.6944 [+0.5853, +0.8079] s/query: **PASS**
- **N3 timing:** error(chain) - error(EAF) +0.0103 [+0.0068, +0.0135]: **PASS**
- **Confirmation: NOT CONFIRMED**; fidelity timing beats 4k timing: **not supported**

| pipeline (contact scenes) | mean error | risk | cost / always-reference |
|---|---|---|---|
| generate only (two objects) | 0.1862 | 0.748 | 0.10 |
| certify or simulate | 0.0027 | 0.010 | 1.16 |
| **EAF repair** | 0.0173 | 0.034 | 1.01 |
| chain-gated V2 (4k timing) | 0.0276 | 0.026 | 0.99 |
| event-timed V2 (secondary) | 0.0453 | 0.103 | 0.87 |
| always simulate | 0.0000 | 0.000 | 1.00 |

Controls: 76; generation risk 0.145; falsely rejected among good 0.385.
