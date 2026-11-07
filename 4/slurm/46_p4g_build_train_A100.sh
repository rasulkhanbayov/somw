#!/bin/bash -l
#SBATCH -J somw_p4g_build
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_build_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_build_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3


# PHASE 4g data on 4 x A100: the frame-conditioned generator's training set. PhysCtrl's 100
# sample/ objects (no overlap with the evaluation objects) x 30 in-distribution physics draws,
# simulated at the reference setting. Pre-registered: PLAN.md "Phase 4g".
# Estimate (not measured): 3,000 rollouts x ~5 s = about 4 GPU-h, about 1 h wall on 4 A100.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p4g_train_build.yaml
run_resumable python code/p2_build_ood.py --config configs/p4g_train_build.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4g train-data shard $SLURM_ARRAY_TASK_ID done. =="
