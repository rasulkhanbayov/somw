#!/bin/bash -l
#SBATCH -J somw_110_p14a2_eval_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/110_p14a2_eval_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/110_p14a2_eval_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 02:00:00
#SBATCH --array=0-1

# PILOT 14a-v2 (pre-registered Oct 24, 2026; PLAN.md "Phase 14"): sample 0 of the four 14a-v2 checkpoints on the development sets. Needs 109. Verdict on a dev server: python code/p14a_summary.py --config configs/p14a2_eval.yaml
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p14a2_eval.yaml
python code/p14_eval_generate.py --config configs/p14a2_eval.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 2
echo "== 110_p14a2_eval_H200 task $SLURM_ARRAY_TASK_ID done. =="
