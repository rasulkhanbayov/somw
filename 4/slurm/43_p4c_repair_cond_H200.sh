#!/bin/bash -l
#SBATCH -J somw_p4c_repair
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4c_repair_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4c_repair_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 10:00:00
#SBATCH --array=0-3

# PHASE 4c on 4 x H200: conditioning repair. For each Phase 2c query and model, 30 conditioning
# adjustments (dlog10E x force scale) are generated in one batch from shared noise; the exact
# simulator picks the most consistent one (certificate defect + anchor mismatch). Starts by
# checking that identity conditioning reproduces PhysCtrl's pipeline. Pre-registered:
# code/p4c_repair_conditioning.py, PLAN.md "Phase 4c". Needs the Phase 4b rows (job 419956).
# Estimate (not measured): about 8 GPU-h, dominated by large (30 x 25-step samples per query).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4c_repair.yaml
run_resumable python code/p4c_repair_conditioning.py --config configs/p4c_repair.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4c conditioning repair shard $SLURM_ARRAY_TASK_ID done. =="
