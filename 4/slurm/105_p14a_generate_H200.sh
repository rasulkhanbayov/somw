#!/bin/bash -l
#SBATCH -J somw_105_p14a_generate_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/105_p14a_generate_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/105_p14a_generate_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-1

# PHASE 14a (pre-registered Oct 22, 2026; PLAN.md "Phase 14"): simulator-distillation pilot. Step 2: PhysCtrl base, 8 samples per candidate. Needs 104.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p14a_gen.yaml
python code/p2b_generate.py --config configs/p14a_gen.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 2
echo "== 105_p14a_generate_H200 task $SLURM_ARRAY_TASK_ID done. =="
