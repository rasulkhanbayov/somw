#!/bin/bash
# Pilot E1b on 4 x H200 (exploratory; PLAN.md "Pilot E1b"): the stiff-material regime.
#     bash slurm/submit_pilot_e1.sh        (from the project root, on panther-login)
# build (84) -> generations (85) -> ladder + scan + restarts (86). About 4 GPU-hours.
# When it has finished, tell Claude: it runs the probe errors and code/p8e_stiff.py --config configs/p8g_stiff.yaml on a dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2
    exit 1
fi
j84=$(sbatch --parsable slurm/84_p8g_build_H200.sh)
j85=$(sbatch --parsable --dependency=afterok:$j84 slurm/85_p8g_generate_H200.sh)
j86=$(sbatch --parsable --dependency=afterok:$j85 slurm/86_p8g_sims_H200.sh)
echo "submitted: build=$j84 -> generate=$j85 -> sims=$j86 (H200)"
