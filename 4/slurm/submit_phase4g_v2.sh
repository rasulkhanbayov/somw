#!/bin/bash
# Phase 4g-v2: fine-tune (1 x H200) -> repair evaluation (4 x H200). Reuses the 4g training data.
#     bash slurm/submit_phase4g_v2.sh     (on panther-login; any directory works)
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then echo "ERROR: run on the Slurm login node." >&2; exit 1; fi
j49=$(sbatch --parsable slurm/49_p4g_train_v2_H200.sh)
j50=$(sbatch --parsable --dependency=afterok:$j49 slurm/50_p4g_repair_v2_H200.sh)
echo "submitted: v2 fine-tune=$j49 -> v2 repair eval=$j50"
