#!/bin/bash -l
#SBATCH -J somw_p4g_repair
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_repair_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_repair_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 10:00:00
#SBATCH --array=0-3

# PHASE 4g evaluation on 4 x H200: repair with the fine-tuned frame-conditioned generator
# (R7 all constraints, R7u none, R8 anchor only), same constraints and seeds as Phase 4b.
# Needs 47 (final.pt). Pre-registered: PLAN.md "Phase 4g". Estimate: about 1 H200-h.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4g_repair.yaml
run_resumable python code/p4g_repair_fc.py --config configs/p4g_repair.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4g repair shard $SLURM_ARRAY_TASK_ID done. =="
