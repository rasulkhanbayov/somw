#!/bin/bash
# Phase 10c (pre-registered Oct 21, 2026; PLAN.md "Phase 10c") on 4 x H200: exact re-scoring against three reference
# time steps. About 8-9 GPU-h.      bash slurm/submit_phase10c.sh        (from the project root, on panther-login)
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2
    exit 1
fi
j=$(sbatch --parsable slurm/101_p10c_rescore_H200.sh)
echo "submitted: p10c rescore=$j (H200)"
