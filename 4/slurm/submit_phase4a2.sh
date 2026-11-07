#!/bin/bash
# Phase 4a extension on 4 x H200: build -> generate -> errors + scan for id_new_4..11.
#     bash slurm/submit_phase4a2.sh        (from the project root, on panther-login)
# When it has finished, tell Claude: it runs the object-level certificate here.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j36=$(sbatch --parsable slurm/36_p4_build_calib2_H200.sh)
j37=$(sbatch --parsable --dependency=afterok:$j36 slurm/37_p4_generate_calib2_H200.sh)
j38=$(sbatch --parsable --dependency=afterok:$j37 slurm/38_p4_scan_calib2_H200.sh)
echo "submitted: build=$j36 -> generate=$j37 -> errors+scan=$j38 (H200)"
