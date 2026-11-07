# Notes on `VERIFY.md` (run of Nov 3, 2026; traced Nov 7)

Every stored verdict computed on a dev server was re-run from the archived data: 46 of 50 `SUMMARY.md` files came back
byte-identical on the first pass, and both paper tables are identical. The other four, each traced to its cause:

| Verdict | What differed | Cause | Effect on any reported number |
|---|---|---|---|
| `p12_s0_diagnose` | FAILED in the scratch copy | `p12_s0.py` located the data relative to the code, so the scratch copy found no queries (harness artefact). Fixed: it now uses `DATA_ROOT`. Re-run on Nov 7: IDENTICAL | none |
| `p14a_acquire` | one blank line added | the `--wall-free` option (Oct 24) added an empty optional line to the report | none (arms and counts identical) |
| `p4g_summary` | one line added: `Non-degradation (\|R7u - R0\| <= 0.005 on test_id): +0.0329 -> FAIL` | the summary script was extended for 4g-v2 after this verdict was written, and now prints the check for v1 too | none; the line agrees with the recorded v1 result (forgetting) |
| `p10a_convergence` | two table cells in the 4th decimal: `ood3_E_low` p95 of d_half 0.1995 → 0.1994; contact p95 of d_quarter 0.0585 → 0.0584 | job 96 ran twice (430617, then 430786 about 4 hours later, not recorded in the log at the time). The second run overwrote the rows, and the stored verdict predates it. GPU MPM is not bit-reproducible, so the rows differ slightly | none: `summary.json` (R1, R2, R3 and all bounds) is identical, neither cell is cited in the paper or `PLAN.md` (which use 0.20 and 0.044) |

Not re-run: the Phase 12 S0 gate (`results/p12_s0`, hours of PolyFEM; job 102). Its record and trajectories are archived.

The stored files were left unchanged, so `results/` shows what was reported at the time. GPU runs are not re-executed by
this check: their stored rows and trajectories are what the verdicts read.
