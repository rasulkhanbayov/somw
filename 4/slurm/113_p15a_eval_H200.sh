#!/bin/bash -l
#SBATCH -J somw_113_p15a_eval_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/113_p15a_eval_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/113_p15a_eval_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 02:00:00
#SBATCH --array=0-1

# PHASE 15a (pre-registered Oct 22, 2026; PLAN.md "Phase 15"): contact-aware generator pilot. Step 3: sample 0 of composed PhysCtrl, base on the joint input and the fine-tuned model on the 200 development scenes. Needs 112. Verdict on a dev server: python code/p15a_summary.py
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p15a_eval.yaml
python code/p15a_eval.py --config configs/p15a_eval.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 2
echo "== 113_p15a_eval_H200 task $SLURM_ARRAY_TASK_ID done. =="
