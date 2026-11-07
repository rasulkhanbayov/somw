#!/bin/bash
# Phase 4o on 4 x H200 (pre-registered Oct 16, 2026): shadow-timed fidelity switch on fresh scenes.
#     bash slurm/submit_phase4n.sh        (from the project root, on panther-login)
# components (72) -> generations (73) -> scenes (74) -> scan + restarts (75). About 9-10 GPU-hours.
# When it has finished, tell Claude: it runs code/p4o_summarize.py on a dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j72=$(sbatch --parsable slurm/72_p4o_comp_build_H200.sh)
j73=$(sbatch --parsable --dependency=afterok:$j72 slurm/73_p4o_comp_generate_H200.sh)
j74=$(sbatch --parsable --dependency=afterok:$j73 slurm/74_p4o_scene_build_H200.sh)
j75=$(sbatch --parsable --dependency=afterok:$j74 slurm/75_p4o_scan_allk_H200.sh)
echo "submitted: components=$j72 -> generate=$j73 -> scenes=$j74 -> scan+restarts=$j75 (H200)"
