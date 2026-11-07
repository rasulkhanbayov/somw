# Phase 12 gate S0 (development queries only; pre-registered Oct 22, 2026)

- dt rule (10 queries): change_4_8 0.0003 -> n_div = 4
- S0a free fall (5 queries): max deviation 1.91e-07 (limit 1e-3): **PASS**
- S0b small strain (20 queries, log E >= 6): median MPM-S2 distance over frames 1-8 0.0047 (limit 0.02; per query max 0.0560): **PASS**
- S0c push (same 20): max relative error of the centre-of-mass displacement at frame 1 0.0638 (limit 0.02): **FAIL**
- Runtime per full query at n_div 4: median 4.8 min (8 threads)
- **S0: FAIL**
