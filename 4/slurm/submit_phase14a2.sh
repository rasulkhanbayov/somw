#!/bin/bash
# Pilot 14a-v2 (pre-registered Oct 24, 2026; PLAN.md "Phase 14"). Needs results/p14a2_acquire (Claude ran it on a dev
# server). Training -> evaluation, chained; the verdict runs on a dev server afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f results/p14a2_acquire/arm_ACT.json ] || { echo "results/p14a2_acquire missing"; exit 1; }
d=$(sbatch --parsable slurm/109_p14a2_train_H200.sh)
e=$(sbatch --parsable --dependency=afterok:$d slurm/110_p14a2_eval_H200.sh)
echo "14a-v2 submitted: train $d -> eval $e"
