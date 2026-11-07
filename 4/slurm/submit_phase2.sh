#!/bin/bash
# Submit Phase 2 with dependencies, from the project root, AFTER the Phase 1 verdict:
#     bash slurm/submit_phase2.sh
# 07 (OOD build, 4 x A100) -> 08 (lift floor, 4 x A100) and 09 (generation, 2 x H200)
# -> 10 (probes, 4 x A100, after 09). Then run `bash slurm/11_p2_summarize.sh`.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j07=$(sbatch --parsable slurm/07_p2_build_ood_A100.sh)
j08=$(sbatch --parsable --dependency=afterok:$j07 slurm/08_p2a_lift_floor_A100.sh)
j09=$(sbatch --parsable --dependency=afterok:$j07 slurm/09_p2b_generate_H200.sh)
j10=$(sbatch --parsable --dependency=afterok:$j09 slurm/10_p2b_probe_A100.sh)
echo "submitted: ood=$j07 lift_floor=$j08 generate=$j09 probe=$j10"
echo "when 08 and 10 are done:  bash slurm/11_p2_summarize.sh"
