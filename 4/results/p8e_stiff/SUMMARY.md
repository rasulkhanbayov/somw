# Pilot E1: the stiff-material regime (EXPLORATORY, development pilot set; defined Oct 18, 2026)

Queries: 150 (log10 E 7.52-9.00); E >= 1e8: 109. tau = 0.02614. Accepted by the certificate: 0.76. Costs in A100 s (reference at PhysCtrl's setting = 4.666 s).

| pipeline | log10 E [7.5, 8.0) risk / cost | log10 E [8.0, 8.5) risk / cost | log10 E [8.5, 9.0) risk / cost | E >= 1e8: risk / cost |
|---|---|---|---|---|
| generate only | 0.049 / 0.24 s | 0.057 / 0.24 s | 0.089 / 0.24 s | 0.073 / 0.24 s |
| ours (gated, M2 repair) | 0.024 / 2.04 s | 0.038 / 3.87 s | 0.036 / 5.35 s | 0.037 / 4.63 s |
| stiff reference | 0.000 / 5.92 s | 0.000 / 10.40 s | 0.000 / 18.22 s | 0.000 / 14.41 s |
| always M1 | 0.024 / 2.68 s | 0.019 / 4.79 s | 0.036 / 8.39 s | 0.028 / 6.64 s |
| always M2 | 0.024 / 2.68 s | 0.019 / 4.79 s | 0.036 / 8.39 s | 0.028 / 6.64 s |
| always M3 | 0.000 / 3.50 s | 0.000 / 6.24 s | 0.000 / 10.93 s | 0.000 / 8.65 s |
| ours, one-window certificate (secondary) | 0.024 / 1.29 s | 0.038 / 2.85 s | 0.036 / 3.64 s | 0.037 / 3.26 s |

S* (cheapest of M1/M2/M3 with risk <= 0.02 on E >= 1e8) = **always M3**.

## Pre-stated rules (E >= 1e8)

- (i) cost(S*) 8.65 s >= 0.5 x stiff reference 14.41 s: **yes**
- (ii) ours risk 0.037 <= 0.02 and cost 4.63 s <= 0.5 x S* (4.32 s): **no**
- **E1 promising: NO**
- E3 (idealised latency): median 0.98 s vs 0.5 x S* serial 4.32 s: **promising**

Exploratory pilot; the stiff split is a pilot set, not a confirmation; latency assumes concurrent windows.
