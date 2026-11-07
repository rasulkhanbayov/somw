#!/bin/bash -l
#SBATCH -J somw_58_p4m_build_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/58_p4m_build_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/58_p4m_build_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 06:00:00
#SBATCH --array=0-3

# PHASE 4m (pre-registered Oct 14, 2026; PLAN.md "Phase 4m"). Ground truth for 600 fresh queries (4 random shift families + id_new_20/21). Output: $DATA_ROOT/fresh4m/, results/p4m_build/.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4m_build.yaml
run_resumable python code/p2_build_ood.py --config configs/p4m_build.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== 58_p4m_build_H200 shard $SLURM_ARRAY_TASK_ID done. =="
