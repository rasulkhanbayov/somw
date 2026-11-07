#!/bin/bash -l
#SBATCH -J somw_109_p14a2_train_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/109_p14a2_train_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/109_p14a2_train_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PILOT 14a-v2 (pre-registered Oct 24, 2026; PLAN.md "Phase 14"): wall-free pool + replay distillation. One arm x variant per task (0 ACT_Ld, 1 ACT_Ltd, 2 RND_Ld, 3 RND_Ltd). Needs results/p14a2_acquire (done on the dev server).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
V=(ACT_Ld ACT_Ltd RND_Ld RND_Ltd); C=configs/p14a2_train_${V[$SLURM_ARRAY_TASK_ID]}.yaml
log_config "$C"
python code/p14_train.py --config "$C" --resume
echo "== 109_p14a2_train_H200 task $SLURM_ARRAY_TASK_ID done. =="
