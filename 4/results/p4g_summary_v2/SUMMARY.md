# Phase 4g: frame-conditioned repair (pre-registered Oct 12, 2026)

300 base trajectories joined with Phase 4b. Constraints and seeds are identical to Phase 4b.

- **Primary**: the frame-conditioned generator propagates verified windows (R1 - R7 > 0 on unconstrained AND all frames): FAIL


## rejected by the certificate (PRIMARY) (n = 198)

| region | R0 | R1 replace | R2 (4b) | **R7 FC** | R7u FC unconstr. | R8 FC anchor | R1 - R7 [95% CI] | R2 - R7 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.2671 | 0.2300 | 0.2276 | 0.2303 | 0.2700 | 0.2660 | -0.0002 [-0.0018,+0.0013] | -0.0026 [-0.0039,-0.0013] |
| free | 0.3139 | 0.3139 | 0.3072 | 0.3143 | 0.3169 | 0.3215 | -0.0004 [-0.0041,+0.0030] | -0.0071 [-0.0100,-0.0040] |
| before | 0.1394 | 0.1394 | 0.1378 | 0.1412 | 0.1414 | 0.1423 | -0.0018 [-0.0029,-0.0009] | -0.0034 [-0.0041,-0.0027] |

Extrapolation from the anchor (frames >= 4): R0 0.3101 -> R8 0.3186; R0 - R8 -0.0084 [-0.0116,-0.0055]
Fine-tuning alone (unconstrained): R0 0.2671 vs R7u 0.2700; R0 - R7u -0.0028 [-0.0054,-0.0005]
Certificate acceptance after repair (lambda 0.02614): R7 0.02

## all base (n = 300)

| region | R0 | R1 replace | R2 (4b) | **R7 FC** | R7u FC unconstr. | R8 FC anchor | R1 - R7 [95% CI] | R2 - R7 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.1901 | 0.1647 | 0.1630 | 0.1655 | 0.1932 | 0.1911 | -0.0008 [-0.0019,+0.0002] | -0.0026 [-0.0035,-0.0017] |
| free | 0.2216 | 0.2216 | 0.2167 | 0.2237 | 0.2249 | 0.2290 | -0.0021 [-0.0045,+0.0005] | -0.0070 [-0.0091,-0.0049] |
| before | 0.0939 | 0.0939 | 0.0927 | 0.0966 | 0.0962 | 0.0971 | -0.0026 [-0.0034,-0.0020] | -0.0038 [-0.0043,-0.0033] |

Extrapolation from the anchor (frames >= 4): R0 0.2208 -> R8 0.2290; R0 - R8 -0.0082 [-0.0105,-0.0059]
Fine-tuning alone (unconstrained): R0 0.1901 vs R7u 0.1932; R0 - R7u -0.0031 [-0.0048,-0.0013]
Certificate acceptance after repair (lambda 0.02614): R7 0.12

## test_id only (n = 100)

| region | R0 | R1 replace | R2 (4b) | **R7 FC** | R7u FC unconstr. | R8 FC anchor | R1 - R7 [95% CI] | R2 - R7 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| all | 0.0702 | 0.0668 | 0.0657 | 0.0681 | 0.0748 | 0.0768 | -0.0014 [-0.0022,-0.0005] | -0.0024 [-0.0030,-0.0019] |
| free | 0.0753 | 0.0753 | 0.0724 | 0.0792 | 0.0798 | 0.0835 | -0.0038 [-0.0061,-0.0017] | -0.0068 [-0.0084,-0.0052] |
| before | 0.0223 | 0.0223 | 0.0213 | 0.0258 | 0.0251 | 0.0262 | -0.0034 [-0.0041,-0.0027] | -0.0045 [-0.0050,-0.0039] |

Extrapolation from the anchor (frames >= 4): R0 0.0823 -> R8 0.0920; R0 - R8 -0.0097 [-0.0131,-0.0064]
Fine-tuning alone (unconstrained): R0 0.0702 vs R7u 0.0748; R0 - R7u -0.0046 [-0.0073,-0.0019]
Certificate acceptance after repair (lambda 0.02614): R7 0.14
Non-degradation (|R7u - R0| <= 0.005 on test_id): +0.0046 -> PASS
