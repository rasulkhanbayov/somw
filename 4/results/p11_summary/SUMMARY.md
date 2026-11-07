# Phase 11: video-level evaluation (pre-registered Oct 20, 2026)

Cases: 58 (48 Phase 11, 10 PhysCtrl examples); excluded (preprocessing, empty drag box or unstable reference): 2 ['p11_00_rubber_duck__coeff_0.4', 'p11_18_rubber_giraffe_teether__coeff_0.4']. Judge: Qwen/Qwen2-VL-7B-Instruct.

## rejected shifted (28 cases)

| variant | SA | PC | VQ | trajectory error |
|---|---|---|---|---|
| PhysCtrl | 4.21 (n 28) | 4.21 (n 28) | 3.21 (n 28) | 0.5072 |
| ours | 4.29 (n 28) | 4.29 (n 28) | 3.29 (n 28) | 0.0045 |
| reference | 4.29 (n 28) | 4.29 (n 28) | 3.29 (n 28) | 0.0000 |

## accepted shifted (0 cases)

| variant | SA | PC | VQ | trajectory error |
|---|---|---|---|---|
| PhysCtrl | — | — | — | — |
| ours | — | — | — | — |
| reference | — | — | — | — |

## in range (30 cases)

| variant | SA | PC | VQ | trajectory error |
|---|---|---|---|---|
| PhysCtrl | 4.60 (n 30) | 4.60 (n 30) | 3.67 (n 30) | 0.0535 |
| ours | 4.60 (n 30) | 4.60 (n 30) | 3.70 (n 30) | 0.0342 |
| reference | 4.63 (n 30) | 4.63 (n 30) | 3.67 (n 30) | 0.0000 |

## Tests (rejected shifted cases, PC = physical commonsense)

- **V0 judge validity:** PC(reference) - PC(PhysCtrl) +0.07 [-0.07, +0.21] (n 28): **FAIL**
- **V1:** PC(ours) - PC(PhysCtrl) +0.07 [-0.07, +0.21] (n 28): **FAIL** (uninformative: the judge does not separate the reference from PhysCtrl)
- V2 (human preference): see p11_study.py.

Figure case (median PhysCtrl error among rejected shifted Phase 11 cases): p11_10_rubber_boot__coeff_0.4
