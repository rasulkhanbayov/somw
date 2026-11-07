#!/bin/bash
# Pilot E1 on 4 x H200 (exploratory; PLAN.md "Pilots E1-E3"): the stiff-material regime.
#     bash slurm/submit_pilot_e1.sh        (from the project root, on panther-login)
# build (81) -> generations (82) -> ladder + scan + restarts (83). About 4 GPU-hours.
# When it has finished, tell Claude: it runs the probe errors and code/p8e_stiff.py on a dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2
    exit 1
fi
j81=$(sbatch --parsable slurm/81_p8e_build_H200.sh)
j82=$(sbatch --parsable --dependency=afterok:$j81 slurm/82_p8e_generate_H200.sh)
j83=$(sbatch --parsable --dependency=afterok:$j82 slurm/83_p8e_sims_H200.sh)
echo "submitted: build=$j81 -> generate=$j82 -> sims=$j83 (H200)"
