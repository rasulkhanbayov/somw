#!/bin/bash -l
#SBATCH -J somw_107_p14a_train_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/107_p14a_train_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/107_p14a_train_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 14a (pre-registered Oct 22, 2026; PLAN.md "Phase 14"): simulator-distillation pilot. Step 4: fine-tune base, one arm x variant per task (0 ACT_F, 1 ACT_L, 2 RND_F, 3 RND_L). Needs results/p14a_acquire (dev server).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
V=(ACT_F ACT_L RND_F RND_L); C=configs/p14a_train_${V[$SLURM_ARRAY_TASK_ID]}.yaml
log_config "$C"
python code/p14_train.py --config "$C" --resume
echo "== 107_p14a_train_H200 task $SLURM_ARRAY_TASK_ID done. =="
