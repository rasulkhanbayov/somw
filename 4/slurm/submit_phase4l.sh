#!/bin/bash
# Phase 4l on 4 x H200 (pre-registered Oct 15, 2026): the multi-object contact benchmark.
#     bash slurm/submit_phase4l.sh        (from the project root, on panther-login)
# components (62) -> component generations (63) -> scenes (64) -> errors + scan + restarts (65).
# About 20 GPU-hours in total; large outputs go to $BIG_ROOT. When it has finished, tell Claude:
# it runs code/p4l_summarize.py here.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j62=$(sbatch --parsable slurm/62_p4l_comp_build_H200.sh)
j63=$(sbatch --parsable --dependency=afterok:$j62 slurm/63_p4l_comp_generate_H200.sh)
j64=$(sbatch --parsable --dependency=afterok:$j63 slurm/64_p4l_scene_build_H200.sh)
j65=$(sbatch --parsable --dependency=afterok:$j64 slurm/65_p4l_scan_allk_H200.sh)
echo "submitted: components=$j62 -> generate=$j63 -> scenes=$j64 -> errors+scan+restarts=$j65 (H200)"
