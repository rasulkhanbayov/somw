#!/bin/bash
# Phase 11 (pre-registered Oct 20, 2026; PLAN.md "Phase 11"), on H200: 98 images -> 99 renders (4 GPUs).
# Before: slurm/00c_download_p11_models.sh once on a dev server. After 99: tell Claude (frames are extracted on a dev
# server), then submit the judge: sbatch slurm/100_p11_judge_H200.sh
#     bash slurm/submit_phase11.sh        (from the project root, on panther-login)
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2
    exit 1
fi
j98=$(sbatch --parsable slurm/98_p11_images_H200.sh)
j99=$(sbatch --parsable --dependency=afterok:$j98 slurm/99_p11_render_H200.sh)
echo "submitted: images=$j98 -> renders=$j99 (H200)"
