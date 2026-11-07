#!/bin/bash -l
#SBATCH -J somw_90_p9c_comp_build_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/90_p9c_comp_build_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/90_p9c_comp_build_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-3

# PHASE 9 (pre-registered Oct 19, 2026; PLAN.md "Phase 9"): confirmation of ours-v2 on fresh data. Contact, step 1: components of draws 24-27.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p9c_comp_build.yaml
run_two_shards python code/p2_build_ood.py --config configs/p9c_comp_build.yaml
echo "== 90 p9c components pair $SLURM_ARRAY_TASK_ID done. =="
