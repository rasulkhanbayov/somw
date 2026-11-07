#!/bin/bash
# Paper qualitative figures (Oct 19, 2026), on 1-2 H200: 94 recomputes the point-cloud cases, 95 re-renders the videos
# with the Phase 9 pipeline. Independent; about 1-1.5 GPU-hours together.
#     bash slurm/submit_paper_figs.sh        (from the project root, on panther-login)
# When both have finished, tell Claude: it draws the figures on a dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2
    exit 1
fi
j94=$(sbatch --parsable slurm/94_fig_paper_traj_H200.sh)
j95=$(sbatch --parsable slurm/95_p6_video_v2_H200.sh)
echo "submitted: point-cloud cases=$j94, video re-render=$j95 (H200)"
