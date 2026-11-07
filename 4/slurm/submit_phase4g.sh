#!/bin/bash
# Phase 4g chain: training data (4 x A100) -> fine-tune (1 x H200) -> repair evaluation (4 x H200).
#     bash slurm/submit_phase4g.sh       (on panther-login; any directory works)
# When it has finished, tell Claude: it runs p4g_summarize.py here.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node." >&2; exit 1
fi
j46=$(sbatch --parsable slurm/46_p4g_build_train_A100.sh)
j47=$(sbatch --parsable --dependency=afterok:$j46 slurm/47_p4g_train_H200.sh)
j48=$(sbatch --parsable --dependency=afterok:$j47 slurm/48_p4g_repair_H200.sh)
echo "submitted: train data=$j46 -> fine-tune=$j47 -> repair eval=$j48"
