#!/bin/bash
# Phase 14a (pre-registered Oct 22, 2026). Stage 1 = pool -> generations -> certificate scan, chained with dependencies.
# Stage 2 (after Claude runs code/p14a_acquire.py on a dev server): bash slurm/submit_phase14a.sh stage2
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-stage1}" = stage1 ]; then
  a=$(sbatch --parsable slurm/104_p14a_build_A100.sh)
  b=$(sbatch --parsable --dependency=afterok:$a slurm/105_p14a_generate_H200.sh)
  c=$(sbatch --parsable --dependency=afterok:$b slurm/106_p14a_scan_H200.sh)
  echo "stage 1 submitted: build $a -> generate $b -> scan $c"
else
  d=$(sbatch --parsable slurm/107_p14a_train_H200.sh)
  e=$(sbatch --parsable --dependency=afterok:$d slurm/108_p14a_eval_H200.sh)
  echo "stage 2 submitted: train $d -> eval $e"
fi
