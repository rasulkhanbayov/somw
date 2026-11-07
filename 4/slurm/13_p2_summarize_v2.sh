#!/bin/bash
# PHASE 2-v2 verdicts (lift_noF probes; same rules as v1). No GPU: run directly on a shared server:
#     bash slurm/11_p2_summarize.sh
# Needs: 08 and 12 finished.
# Output: results/p2_summary/{SUMMARY.md, summary.json}. Send SUMMARY.md back.
set -euo pipefail
source "$(dirname "$0")/cluster.env"
# conda's activate hooks may read unset variables; suspend -u around them.
set +u
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
cd "$PROJECT_ROOT"
python code/p2_summarize.py --config configs/p2_summary_v2.yaml
echo "== Phase 2 summary: $SOMW_RESULTS/p2_summary_v2/SUMMARY.md =="
