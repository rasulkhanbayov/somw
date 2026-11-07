#!/bin/bash
# Phase 4m on 4 x H200 (pre-registered Oct 14, 2026): fresh shifted queries, the strongest
# baselines, placement at matched budget.
#     bash slurm/submit_phase4m.sh        (from the project root, on panther-login)
# build (58) -> generate + errors (59) -> scan + all-k (60); calibration coarse rollouts (61) in
# parallel. About 9-11 GPU-hours in total. When it has finished, tell Claude: it runs
# code/p4m_confirm.py here.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j58=$(sbatch --parsable slurm/58_p4m_build_H200.sh)
j59=$(sbatch --parsable --dependency=afterok:$j58 slurm/59_p4m_generate_H200.sh)
j60=$(sbatch --parsable --dependency=afterok:$j59 slurm/60_p4m_scan_allk_H200.sh)
j61=$(sbatch --parsable slurm/61_p4m_calib_coarse_H200.sh)
echo "submitted: build=$j58 -> generate+errors=$j59 -> scan+allk=$j60; calibration coarse=$j61 (H200)"
