#!/bin/bash -l
#SBATCH -J somw_p4e_repair
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4e_repair_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4e_repair_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 10:00:00
#SBATCH --array=0-3

# PHASE 4e on 4 x H200: anchored conditioning repair. The same 30 candidates as Phase 4c,
# selected by agreement with a rollout from the TRUE initial state over the first L = 8 / 12
# model frames (the window defect is not used). Pre-registered:
# code/p4e_repair_anchored.py, PLAN.md "Phase 4e". Estimate (not measured): about 8 GPU-h, like 4c.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4e_repair.yaml
run_resumable python code/p4e_repair_anchored.py --config configs/p4e_repair.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4e anchored repair shard $SLURM_ARRAY_TASK_ID done. =="
