#!/bin/bash -l
#SBATCH -J somw_106_p14a_scan_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/106_p14a_scan_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/106_p14a_scan_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 10:00:00
#SBATCH --array=0-3

# PHASE 14a (pre-registered Oct 22, 2026; PLAN.md "Phase 14"): simulator-distillation pilot. Step 3: coarse_cfl m = 4 window scan of the pool (the certificate). Needs 105. Then on a dev server: python code/p14a_acquire.py
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p14a_scan.yaml
run_two_shards python code/p2c_scan.py --config configs/p14a_scan.yaml
echo "== 106_p14a_scan_H200 task $SLURM_ARRAY_TASK_ID done. =="
