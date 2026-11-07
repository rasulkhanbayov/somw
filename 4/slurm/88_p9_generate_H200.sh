#!/bin/bash -l
#SBATCH -J somw_88_p9_generate_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/88_p9_generate_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/88_p9_generate_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 01:30:00
#SBATCH --array=0-1

# PHASE 9 (pre-registered Oct 19, 2026; PLAN.md "Phase 9"): confirmation of ours-v2 on fresh data. Single objects, step 2: generations (8 samples). Needs 87.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p9_gen.yaml
python code/p2b_generate.py --config configs/p9_gen.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 2
echo "== 88 p9 generate task $SLURM_ARRAY_TASK_ID done. =="
