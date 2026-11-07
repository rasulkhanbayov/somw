#!/bin/bash
# Submit Phase 2c (exhaustive scan, 4 x A100) and the Phase 4a calibration data chain
# (4 x H200: build -> generate -> errors + scan) at the same time, from the project root:
#     bash slurm/submit_phase2c_4a.sh
# When everything has finished, tell Claude: it runs slurm/15_p2c_summarize.sh here.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j14=$(sbatch --parsable slurm/14_p2c_scan_A100.sh)
j32=$(sbatch --parsable slurm/32_p4_build_calib_H200.sh)
j33=$(sbatch --parsable --dependency=afterok:$j32 slurm/33_p4_generate_calib_H200.sh)
j34=$(sbatch --parsable --dependency=afterok:$j33 slurm/34_p4_scan_calib_H200.sh)
echo "submitted: p2c scan (A100)=$j14 | calib build=$j32 -> generate=$j33 -> errors+scan=$j34 (H200)"
