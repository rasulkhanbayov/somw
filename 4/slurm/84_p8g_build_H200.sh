#!/bin/bash -l
#SBATCH -J somw_84_p8g_build_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/84_p8g_build_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/84_p8g_build_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-3

# PILOT E1b (exploratory; PLAN.md "Pilot E1b"): the stiff-material regime. Step 1: ground truth with the CFL-safe reference.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p8g_build.yaml
run_two_shards python code/p2_build_ood.py --config configs/p8g_build.yaml
echo "== 84 p8g build pair $SLURM_ARRAY_TASK_ID done. =="
