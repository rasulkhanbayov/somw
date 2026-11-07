#!/bin/bash -l
#SBATCH -J somw_p5_online
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p5_online_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p5_online_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 48:00:00
#SBATCH --array=0-3

# STUB. PHASE 5a: online learning loop (probe replay -> LoRA updates -> fewer probes per query).
# Design depends on the Phase 1-2 verdicts, so this script refuses to run.
# TODO before enabling:
#   - Replay buffer of verified windows; periodic LoRA updates of the generator;
#   - probes-per-query curve at stable error; equal-simulator-budget comparison against
#   - learning from full rollouts.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
echo "STUB: 40_p5_online_TODO_H200.sh is not implemented yet. See PLAN.md." >&2
exit 1
