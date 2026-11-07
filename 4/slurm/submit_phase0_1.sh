#!/bin/bash
# Submit Phase 0 and Phase 1 with dependencies, from the project root:
#     bash slurm/submit_phase0_1.sh
# 01 (A100) and 02 (H200) run in parallel; 03, 04 and the 05 array start only if 01
# succeeded. Then run `bash slurm/06_p1_summarize.sh` when they have all finished.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j01=$(sbatch --parsable slurm/01_p0_check_A100.sh)
j02=$(sbatch --parsable slurm/02_p0_check_H200.sh)
j03=$(sbatch --parsable --dependency=afterok:$j01 slurm/03_p1_cost_A100.sh)
j04=$(sbatch --parsable --dependency=afterok:$j01:$j02 slurm/04_p1_cost_H200.sh)
j05=$(sbatch --parsable --dependency=afterok:$j01 slurm/05_p1_fidelity_A100.sh)
echo "submitted: p0 A100=$j01 p0 H200=$j02 | p1 cost A100=$j03 H200=$j04 | fidelity array=$j05"
echo "when all are done:  bash slurm/06_p1_summarize.sh"
