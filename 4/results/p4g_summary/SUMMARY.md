# Phase 4g: frame-conditioned repair (pre-registered Oct 12, 2026)

300 base trajectories joined with Phase 4b. Constraints and seeds are identical to Phase 4b.

- **Primary**: the frame-conditioned generator propagates verified windows (R1 - R7 > 0 on unconstrained AND all frames): FAIL


## rejected by the certificate (PRIMARY) (n = 198)

| region | R0 | R1 replace | R2 (4b) | **R7 FC** | R7u FC unconstr. | R8 FC anchor | R1 - R7 [95% CI] | R2 - R7 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.2671 | 0.2300 | 0.2276 | 0.2433 | 0.3133 | 0.3044 | -0.0133 [-0.0201,-0.0074] | -0.0157 [-0.0225,-0.0098] |
| free | 0.3139 | 0.3139 | 0.3072 | 0.3450 | 0.3725 | 0.3722 | -0.0311 [-0.0463,-0.0178] | -0.0377 [-0.0533,-0.0243] |
| before | 0.1394 | 0.1394 | 0.1378 | 0.1422 | 0.1489 | 0.1477 | -0.0028 [-0.0044,-0.0012] | -0.0044 [-0.0057,-0.0031] |

Extrapolation from the anchor (frames >= 4): R0 0.3101 -> R8 0.3647; R0 - R8 -0.0545 [-0.0679,-0.0418]
Fine-tuning alone (unconstrained): R0 0.2671 vs R7u 0.3133; R0 - R7u -0.0461 [-0.0573,-0.0355]
Certificate acceptance after repair (lambda 0.02614): R7 0.01

## all base (n = 300)

| region | R0 | R1 replace | R2 (4b) | **R7 FC** | R7u FC unconstr. | R8 FC anchor | R1 - R7 [95% CI] | R2 - R7 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.1901 | 0.1647 | 0.1630 | 0.1752 | 0.2302 | 0.2236 | -0.0105 [-0.0152,-0.0067] | -0.0123 [-0.0168,-0.0084] |
| free | 0.2216 | 0.2216 | 0.2167 | 0.2470 | 0.2707 | 0.2704 | -0.0254 [-0.0357,-0.0167] | -0.0303 [-0.0407,-0.0216] |
| before | 0.0939 | 0.0939 | 0.0927 | 0.0982 | 0.1034 | 0.1025 | -0.0043 [-0.0054,-0.0032] | -0.0055 [-0.0063,-0.0046] |

Extrapolation from the anchor (frames >= 4): R0 0.2208 -> R8 0.2679; R0 - R8 -0.0471 [-0.0567,-0.0393]
Fine-tuning alone (unconstrained): R0 0.1901 vs R7u 0.2302; R0 - R7u -0.0400 [-0.0481,-0.0335]
Certificate acceptance after repair (lambda 0.02614): R7 0.02

## test_id only (n = 100)

| region | R0 | R1 replace | R2 (4b) | **R7 FC** | R7u FC unconstr. | R8 FC anchor | R1 - R7 [95% CI] | R2 - R7 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.0702 | 0.0668 | 0.0657 | 0.0713 | 0.1031 | 0.1006 | -0.0045 [-0.0059,-0.0033] | -0.0056 [-0.0069,-0.0044] |
| free | 0.0753 | 0.0753 | 0.0724 | 0.0874 | 0.1108 | 0.1105 | -0.0120 [-0.0153,-0.0091] | -0.0150 [-0.0179,-0.0122] |
| before | 0.0223 | 0.0223 | 0.0213 | 0.0282 | 0.0321 | 0.0316 | -0.0058 [-0.0067,-0.0050] | -0.0069 [-0.0076,-0.0061] |

Extrapolation from the anchor (frames >= 4): R0 0.0823 -> R8 0.1206; R0 - R8 -0.0382 [-0.0449,-0.0319]
Fine-tuning alone (unconstrained): R0 0.0702 vs R7u 0.1031; R0 - R7u -0.0329 [-0.0384,-0.0276]
Certificate acceptance after repair (lambda 0.02614): R7 0.01
