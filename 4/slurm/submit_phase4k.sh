#!/bin/bash
# Phase 4k on 4 x H200 (pre-registered Oct 14, 2026): fresh queries -> gated V2 test.
#     bash slurm/submit_phase4k.sh        (from the project root, on panther-login)
# build (55) -> generate + errors (56) -> scan + all-k V2 (57). About 8-10 GPU-hours in total.
# When it has finished, tell Claude: it runs code/p4k_gated.py here.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on $(hostname). Submit from the Slurm login node:" >&2
    echo "  ssh panther-login && cd $(pwd) && bash slurm/$(basename "$0")" >&2
    exit 1
fi
j55=$(sbatch --parsable slurm/55_p4k_build_H200.sh)
j56=$(sbatch --parsable --dependency=afterok:$j55 slurm/56_p4k_generate_H200.sh)
j57=$(sbatch --parsable --dependency=afterok:$j56 slurm/57_p4k_scan_allk_H200.sh)
echo "submitted: build=$j55 -> generate+errors=$j56 -> scan+allk=$j57 (H200)"
