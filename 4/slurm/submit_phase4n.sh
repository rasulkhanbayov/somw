#!/bin/bash
# Phase 4n on 4 x H200 (pre-registered Oct 16, 2026): event-aware fidelity repair on fresh scenes.
#     bash slurm/submit_phase4n.sh        (from the project root, on panther-login)
# components (68) -> generations (69) -> scenes (70) -> scan + restarts (71). About 9-10 GPU-hours.
# When it has finished, tell Claude: it runs code/p4n_summarize.py on a dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j68=$(sbatch --parsable slurm/68_p4n_comp_build_H200.sh)
j69=$(sbatch --parsable --dependency=afterok:$j68 slurm/69_p4n_comp_generate_H200.sh)
j70=$(sbatch --parsable --dependency=afterok:$j69 slurm/70_p4n_scene_build_H200.sh)
j71=$(sbatch --parsable --dependency=afterok:$j70 slurm/71_p4n_scan_allk_H200.sh)
echo "submitted: components=$j68 -> generate=$j69 -> scenes=$j70 -> scan+restarts=$j71 (H200)"
