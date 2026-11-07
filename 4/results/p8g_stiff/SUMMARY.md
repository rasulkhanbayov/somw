# Pilot E1b: stiff regime, round 2 (EXPLORATORY, fresh pilot draws; defined Oct 19, 2026)

Queries: 150 (log10 E 7.50-9.00); E >= 1e8: 101. tau = 0.02614. Accepted by the certificate: 0.75. Costs in A100 s (reference at PhysCtrl's setting = 4.666 s).

| pipeline | log10 E [7.5, 8.0) risk / cost | log10 E [8.0, 8.5) risk / cost | log10 E [8.5, 9.0) risk / cost | E >= 1e8: risk / cost |
|---|---|---|---|---|
| generate only | 0.061 / 0.24 s | 0.085 / 0.24 s | 0.074 / 0.24 s | 0.079 / 0.24 s |
| ours (three windows, M3 repair; secondary) | 0.020 / 2.43 s | 0.021 / 3.68 s | 0.074 / 5.97 s | 0.050 / 4.90 s |
| stiff reference | 0.000 / 5.52 s | 0.000 / 9.98 s | 0.000 / 18.28 s | 0.000 / 14.42 s |
| always M1 | 0.000 / 2.43 s | 0.000 / 4.60 s | 0.074 / 8.42 s | 0.040 / 6.64 s |
| always M2 | 0.000 / 2.43 s | 0.000 / 4.60 s | 0.074 / 8.42 s | 0.040 / 6.64 s |
| always M3 | 0.000 / 3.19 s | 0.000 / 5.98 s | 0.000 / 10.96 s | 0.000 / 8.65 s |
| ours (one window, M3 repair) | 0.020 / 1.87 s | 0.043 / 2.23 s | 0.074 / 4.10 s | 0.059 / 3.23 s |

S* (cheapest of M1/M2/M3 with risk <= 0.02 on E >= 1e8) = **always M3**.

## Pre-stated rules (E >= 1e8)

- (i) cost(S*) 8.65 s >= 0.5 x stiff reference 14.42 s: **yes**
- (ii) ours risk 0.059 <= 0.02 and cost 3.23 s <= 0.5 x S* (4.32 s): **no**
- **E1b promising: NO**
- E3 (idealised latency): median 1.04 s vs 0.5 x S* serial 4.32 s: **promising**

Exploratory pilot; the stiff split is a pilot set, not a confirmation; latency assumes concurrent windows.
