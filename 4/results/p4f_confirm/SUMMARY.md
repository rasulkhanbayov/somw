# Phase 4f: certify-or-simulate with a reference fallback (pre-registered Oct 12, 2026)

Base model; 1200 calibration-pool trajectories over 100 held-out objects; 500 object splits. Values: mean [2.5%, 97.5%] over splits, object-weighted. Reference rollout 4.666 s; generation 0.236 s (batch 1: 0.300 s); probe cost 0.047 s/window + 0.2024 ms/substep (fit on 18898 A100 windows).

- **F1 risk** P(err > 0.1) of cert_ref: 0.023 [0.000, 0.055]; target <= 0.11: **PASS**
- **F2 compute** cost(cert_ref) / cost(sim_ref): 0.559 [0.352, 0.849]: **PASS** (with batch-1 generation: 0.573 [0.366, 0.863])
- **F3 value of verification** risk(random_ref) - risk(cert_ref): 0.053 [0.029, 0.072]: **PASS**

| quantity | mean [2.5%, 97.5%] |
|---|---|
| fallback rate | 0.423 [0.217, 0.713] |
| mean error: generate only | 0.071 [0.055, 0.087] |
| mean error: cert_ref | 0.025 [0.008, 0.041] |
| risk: generate only | 0.131 [0.085, 0.172] |
| risk: random_ref | 0.076 [0.032, 0.120] |
| cost cert_ref (s/query) | 2.608 [1.644, 3.962] |
