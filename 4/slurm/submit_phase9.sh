#!/bin/bash
# Phase 9 on 4 x H200 (pre-registered Oct 19, 2026; PLAN.md "Phase 9"): confirmation of ours-v2 on fresh data.
#     bash slurm/submit_phase9.sh        (from the project root, on panther-login)
# Two independent chains: single objects 87 -> 88 -> 89, contact 90 -> 91 -> 92 -> 93. About 7-9 GPU-hours.
# When both have finished, tell Claude: it runs the probe errors and the verdicts on a dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2
    exit 1
fi
j87=$(sbatch --parsable slurm/87_p9_build_H200.sh)
j88=$(sbatch --parsable --dependency=afterok:$j87 slurm/88_p9_generate_H200.sh)
j89=$(sbatch --parsable --dependency=afterok:$j88 slurm/89_p9_sims_H200.sh)
j90=$(sbatch --parsable slurm/90_p9c_comp_build_H200.sh)
j91=$(sbatch --parsable --dependency=afterok:$j90 slurm/91_p9c_comp_generate_H200.sh)
j92=$(sbatch --parsable --dependency=afterok:$j91 slurm/92_p9c_scene_build_H200.sh)
j93=$(sbatch --parsable --dependency=afterok:$j92 slurm/93_p9c_sims_H200.sh)
echo "submitted: single objects build=$j87 -> generate=$j88 -> sims=$j89; contact components=$j90 -> generate=$j91 -> scenes=$j92 -> sims=$j93 (H200)"
