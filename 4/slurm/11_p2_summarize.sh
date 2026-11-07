#!/bin/bash
# PHASE 2 verdicts (G2a lift floor, G2b verifier). No GPU: run directly on a shared server:
#     bash slurm/11_p2_summarize.sh
# Needs: 08 and 10 finished.
# Output: results/p2_summary/{SUMMARY.md, summary.json}. Send SUMMARY.md back.
set -euo pipefail
source "$(dirname "$0")/cluster.env"
# conda's activate hooks may read unset variables; suspend -u around them.
set +u
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
cd "$PROJECT_ROOT"
python code/p2_summarize.py --config configs/p2_summary.yaml
echo "== Phase 2 summary: $SOMW_RESULTS/p2_summary/SUMMARY.md =="
