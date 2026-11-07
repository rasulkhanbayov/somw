#!/bin/bash -l
#SBATCH -J somw_60_p4m_scan_allk_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/60_p4m_scan_allk_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/60_p4m_scan_allk_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 12:00:00
#SBATCH --array=0-3

# PHASE 4m (pre-registered Oct 14, 2026; PLAN.md "Phase 4m"). coarse_cfl m = 4 scan, then V2 at every restart point + coarse rollout with saved positions. Needs 59.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4m_scan.yaml
log_config configs/p4m_allk.yaml
run_resumable python code/p2c_scan.py --config configs/p4m_scan.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
run_resumable python code/p4j_allk.py --config configs/p4m_allk.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== 60_p4m_scan_allk_H200 shard $SLURM_ARRAY_TASK_ID done. =="
