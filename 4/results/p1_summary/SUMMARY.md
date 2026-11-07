# Phase 1 summary: **GO**

Rule: simulation >= 10x generation (per sample) in some realistic regime (physctrl, dense10x, multiobj2, multiobj3) on every hardware measured (A100, H200).

Passing: dense10x with base

## Generation cost (seconds per trajectory sample)

| hardware | model | best batch | batch 1 |
|---|---|---|---|
| A100 | base | 0.2344 | 0.2999 |
| A100 | large | 4.5192 | 4.6959 |
| H200 | base | 0.1399 | 0.2324 |
| H200 | large | 2.5259 | 2.7959 |

## Simulation cost (seconds per 48-frame rollout, median over queries)

| hardware | regime | best mode | best (s) | as_is (s) |
|---|---|---|---|---|
| A100 | dense10x | graph | 5.247 | 9.168 |
| A100 | dense10x_fine | graph | 22.802 | 26.481 |
| A100 | multiobj2 | graph | 4.301 | 8.224 |
| A100 | multiobj3 | graph | 4.293 | 8.233 |
| A100 | physctrl | graph | 4.314 | 8.329 |
| H200 | dense10x | graph | 3.790 | 9.236 |
| H200 | dense10x_fine | graph | 15.444 | 19.521 |
| H200 | multiobj2 | graph | 3.085 | 8.239 |
| H200 | multiobj3 | graph | 3.124 | 6.011 |
| H200 | physctrl | graph | 3.117 | 8.463 |

## Simulator fidelity sweep (A100)

Rows where the rollout blew up and lost the CUDA context: 12 (counted as unstable).

| substep_dt | n_grid | stable | median err | median rollout (s) |
|---|---|---|---|---|
| 0.0001 | 64 | 1.00 | 0.0180 | 4.633 |
| 0.0001 | 96 | 1.00 | 0.0073 | 4.643 |
| 0.0001 | 125 | 1.00 | 0.0000 | 4.667 |
| 0.0002 | 64 | 1.00 | 0.0189 | 2.335 |
| 0.0002 | 96 | 1.00 | 0.0075 | 2.320 |
| 0.0002 | 125 | 1.00 | 0.0056 | 2.332 |
| 0.0004 | 64 | 1.00 | 0.0171 | 1.174 |
| 0.0004 | 96 | 1.00 | 0.0068 | 1.166 |
| 0.0004 | 125 | 1.00 | 0.0087 | 1.175 |
| 0.0008 | 64 | 1.00 | 0.0163 | 0.592 |
| 0.0008 | 96 | 0.97 | 0.0072 | 0.591 |
| 0.0008 | 125 | 0.90 | 0.0131 | 0.593 |
| 0.0016 | 64 | 0.95 | 0.0156 | 0.299 |
| 0.0016 | 96 | 0.70 | 0.0081 | 0.299 |
| 0.0016 | 125 | 0.55 | 0.0428 | 0.299 |

Generator **base** median error 0.0523. Matched-accuracy simulator: (0.0008, 64) at 0.592 s.

Generator **large** median error 0.1193. Matched-accuracy simulator: (0.0008, 64) at 0.592 s.

## Cost ratio: simulation / generation (per sample, best batch)

| hardware | regime | model | sim (s) | gen (s) | ratio | note |
|---|---|---|---|---|---|---|
| A100 | dense10x | base | 5.247 | 0.2344 | 22.4 | gen at 2,048 pts; upsampling not counted |
| A100 | dense10x | large | 5.247 | 4.5192 | 1.2 | gen at 2,048 pts; upsampling not counted |
| A100 | dense10x_fine | base | 22.802 | 0.2344 | 97.3 | gen at 2,048 pts; upsampling not counted |
| A100 | dense10x_fine | large | 22.802 | 4.5192 | 5.0 | gen at 2,048 pts; upsampling not counted |
| A100 | multiobj2 | base | 4.301 | 0.4687 | 9.2 | gen = 2 x single object (lower bound) |
| A100 | multiobj2 | large | 4.301 | 9.0383 | 0.5 | gen = 2 x single object (lower bound) |
| A100 | multiobj3 | base | 4.293 | 0.7031 | 6.1 | gen = 3 x single object (lower bound) |
| A100 | multiobj3 | large | 4.293 | 13.5575 | 0.3 | gen = 3 x single object (lower bound) |
| A100 | physctrl | base | 0.547 | 0.2344 | 2.3 | matched-accuracy sim (x0.13 of reference) |
| A100 | physctrl | large | 0.547 | 4.5192 | 0.1 | matched-accuracy sim (x0.13 of reference) |
| H200 | dense10x | base | 3.790 | 0.1399 | 27.1 | gen at 2,048 pts; upsampling not counted |
| H200 | dense10x | large | 3.790 | 2.5259 | 1.5 | gen at 2,048 pts; upsampling not counted |
| H200 | dense10x_fine | base | 15.444 | 0.1399 | 110.4 | gen at 2,048 pts; upsampling not counted |
| H200 | dense10x_fine | large | 15.444 | 2.5259 | 6.1 | gen at 2,048 pts; upsampling not counted |
| H200 | multiobj2 | base | 3.085 | 0.2797 | 11.0 | gen = 2 x single object (lower bound) |
| H200 | multiobj2 | large | 3.085 | 5.0518 | 0.6 | gen = 2 x single object (lower bound) |
| H200 | multiobj3 | base | 3.124 | 0.4196 | 7.4 | gen = 3 x single object (lower bound) |
| H200 | multiobj3 | large | 3.124 | 7.5777 | 0.4 | gen = 3 x single object (lower bound) |
| H200 | physctrl | base | 0.395 | 0.1399 | 2.8 | matched-accuracy sim (x0.13 of reference) |
| H200 | physctrl | large | 0.395 | 2.5259 | 0.2 | matched-accuracy sim (x0.13 of reference) |

## Probe cost (one window, best mode)

| hardware | window (model frames) | seconds | fraction of best full rollout |
|---|---|---|---|
| A100 | 1 | 0.307 | 0.071 |
| A100 | 2 | 0.484 | 0.112 |
| H200 | 1 | 0.291 | 0.093 |
| H200 | 2 | 0.425 | 0.136 |