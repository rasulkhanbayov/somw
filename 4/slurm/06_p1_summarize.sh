#!/bin/bash
# PHASE 1 verdict. No GPU needed: run directly on a shared server, not via sbatch:
#     bash slurm/06_p1_summarize.sh
# Needs: 03, 04 and all four 05 tasks finished.
# Output: results/p1_summary/{SUMMARY.md, summary.json}. Send SUMMARY.md back.
set -euo pipefail
source "$(dirname "$0")/cluster.env"
# conda's activate hooks may read unset variables; suspend -u around them.
set +u
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
cd "$PROJECT_ROOT"
python code/p1_summarize.py --config configs/p1_summary.yaml
echo "== Phase 1 summary: $SOMW_RESULTS/p1_summary/SUMMARY.md =="
