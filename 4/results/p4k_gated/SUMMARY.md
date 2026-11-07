# Phase 4k: cert-gated V2 on fresh queries (pre-registered Oct 14, 2026)

Fresh queries: 800 (100 objects; splits id_new_12, id_new_13, id_new_14, id_new_15, id_new_16, id_new_17, id_new_18, id_new_19). Fixed tau = 0.02614; tau_hi = 0.04501. Costs in A100 s/query: reference 4.666, generation 0.236; probe fit 0.047 s/window + 0.2024 ms/substep; shadow/coarse 0.2288 ms/substep (pooled median of 1200 A100 calibration rollouts). Object-cluster bootstrap, 2000 resamples.

Blown restarts (fallback to reference): 0; crashed queries: 0.

- **K1 risk** at tau: gated 0.025 vs cert_ref 0.011 (limit +0.02, and <= 0.11): **PASS**
- **K2 compute** saving cert_ref - gated (s/query): tau +0.535 [+0.479, +0.592]; tau_hi +0.256 [+0.208, +0.305]: **PASS**
- **K3 timing** error(shuffled) - error(gated) at tau: +0.002 [+0.001, +0.003]: **PASS**
- **Confirmation: CONFIRMED**; timing claim: **supported**

## tau = 0.01625: rejected 0.738, gated suffix 0.556

| pipeline | mean error | risk | cost (s/query) | cost / always-reference |
|---|---|---|---|---|
| generate only | 0.0693 | 0.139 | 0.236 | 0.05 |
| coarse_cfl only | 0.0304 | 0.046 | 0.577 | 0.12 |
| certify-or-simulate | 0.0074 | 0.005 | 4.074 | 0.87 |
| V2 ungated (4j) | 0.0166 | 0.009 | 3.380 | 0.72 |
| **gated V2** | 0.0170 | 0.011 | 3.515 | 0.75 |
| gated V2, shuffled restarts | 0.0192 | 0.014 | 3.515 | 0.75 |

## tau = 0.02614 (primary): rejected 0.511, gated suffix 0.476

| pipeline | mean error | risk | cost (s/query) | cost / always-reference |
|---|---|---|---|---|
| generate only | 0.0693 | 0.139 | 0.236 | 0.05 |
| coarse_cfl only | 0.0304 | 0.046 | 0.577 | 0.12 |
| certify-or-simulate | 0.0177 | 0.011 | 3.019 | 0.65 |
| V2 ungated (4j) | 0.0306 | 0.025 | 2.533 | 0.54 |
| **gated V2** | 0.0310 | 0.025 | 2.484 | 0.53 |
| gated V2, shuffled restarts | 0.0332 | 0.028 | 2.484 | 0.53 |

## tau = 0.02992: rejected 0.438, gated suffix 0.406

| pipeline | mean error | risk | cost (s/query) | cost / always-reference |
|---|---|---|---|---|
| generate only | 0.0693 | 0.139 | 0.236 | 0.05 |
| coarse_cfl only | 0.0304 | 0.046 | 0.577 | 0.12 |
| certify-or-simulate | 0.0230 | 0.020 | 2.675 | 0.57 |
| V2 ungated (4j) | 0.0355 | 0.030 | 2.286 | 0.49 |
| **gated V2** | 0.0345 | 0.028 | 2.203 | 0.47 |
| gated V2, shuffled restarts | 0.0370 | 0.035 | 2.203 | 0.47 |

## tau = 0.04501 (K2): rejected 0.251, gated suffix 0.241

| pipeline | mean error | risk | cost (s/query) | cost / always-reference |
|---|---|---|---|---|
| generate only | 0.0693 | 0.139 | 0.236 | 0.05 |
| coarse_cfl only | 0.0304 | 0.046 | 0.577 | 0.12 |
| certify-or-simulate | 0.0389 | 0.055 | 1.805 | 0.39 |
| V2 ungated (4j) | 0.0459 | 0.061 | 1.777 | 0.38 |
| **gated V2** | 0.0462 | 0.064 | 1.549 | 0.33 |
| gated V2, shuffled restarts | 0.0480 | 0.069 | 1.549 | 0.33 |

## Secondary: the 50 replicate-0 test-half objects (unseen geometry for the tau calibration), tau = 0.02614

| pipeline | mean error | risk | cost (s/query) |
|---|---|---|---|
| generate only | 0.0708 | 0.158 | 0.236 |
| coarse_cfl only | 0.0317 | 0.062 | 0.577 |
| certify-or-simulate | 0.0185 | 0.015 | 3.071 |
| V2 ungated (4j) | 0.0306 | 0.028 | 2.594 |
| **gated V2** | 0.0311 | 0.028 | 2.547 |
| gated V2, shuffled restarts | 0.0340 | 0.030 | 2.547 |
