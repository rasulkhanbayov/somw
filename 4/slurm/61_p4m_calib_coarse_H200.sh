#!/bin/bash -l
#SBATCH -J somw_61_p4m_calib_coarse_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/61_p4m_calib_coarse_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/61_p4m_calib_coarse_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 06:00:00
#SBATCH --array=0-3

# PHASE 4m (pre-registered Oct 14, 2026; PLAN.md "Phase 4m"). coarse_cfl rollouts with positions for the 1,200 calibration queries (threshold calibration of the agreement baseline). Independent of 58-60.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4m_calib_coarse.yaml
run_resumable python code/p4j_allk.py --config configs/p4m_calib_coarse.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== 61_p4m_calib_coarse_H200 shard $SLURM_ARRAY_TASK_ID done. =="
