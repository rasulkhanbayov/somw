#!/bin/bash
# Phase 10 (pre-registered Oct 20, 2026; PLAN.md "Phase 10") on H200: 96 = reference convergence (4 GPUs, ~3.8 GPU-h),
# 97 = PhysCtrl's metrics with the final pipeline (4 GPUs, minutes). Independent.
#     bash slurm/submit_phase10.sh        (from the project root, on panther-login)
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2
    exit 1
fi
j96=$(sbatch --parsable slurm/96_p10a_convergence_H200.sh)
j97=$(sbatch --parsable slurm/97_p10b_testset_H200.sh)
echo "submitted: convergence=$j96, PhysCtrl metrics=$j97 (H200)"
