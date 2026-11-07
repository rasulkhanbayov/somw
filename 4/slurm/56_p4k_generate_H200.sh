#!/bin/bash -l
#SBATCH -J somw_p4k_generate
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4k_generate_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4k_generate_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 4k step 2 on 4 x H200: 8 base samples per fresh query, then true error and seed spread.
# Needs: 55. Output: $DATA_ROOT/generations/base/id_new_1[2-9]/, results/p4k_errors/
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4k_gen.yaml
python code/p2b_generate.py --config configs/p4k_gen.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
run_resumable python code/p2b_probe.py --config configs/p4k_gen.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4k generate + errors shard $SLURM_ARRAY_TASK_ID done. =="
