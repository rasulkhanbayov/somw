#!/bin/bash
# Re-submit Phase 1 only (03, 04, 05), when Phase 0 already passed, from the project root:
#     bash slurm/submit_phase1.sh
# No dependencies: they read results/p0_check_A100/testset_manifest.json, which must exist.
# Then run `bash slurm/06_p1_summarize.sh` when all have finished.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
if [ ! -f results/p0_check_A100/testset_manifest.json ]; then
    echo "ERROR: results/p0_check_A100/testset_manifest.json missing: run Phase 0 (submit_phase0_1.sh)." >&2
    exit 1
fi
j03=$(sbatch --parsable slurm/03_p1_cost_A100.sh)
j04=$(sbatch --parsable slurm/04_p1_cost_H200.sh)
j05=$(sbatch --parsable slurm/05_p1_fidelity_A100.sh)
echo "submitted: p1 cost A100=$j03 H200=$j04 | fidelity array=$j05"
echo "when all are done:  bash slurm/06_p1_summarize.sh"
