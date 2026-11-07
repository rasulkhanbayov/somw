#!/bin/bash -l
#SBATCH -J somw_p4h_suffix
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4h_suffix_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4h_suffix_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 10:00:00
#SBATCH --array=0-3

# PHASE 4h on 4 x H200: identical to 51_p4h_suffix_A100.sh (the A100 queue was ~1.5 days on
# Oct 13). The simulator is GPU-agnostic; the verdict costs the suffix as its share of the
# measured A100 reference rollout (configs/p4h_suffix.yaml: suffix_cost: ref_fraction), so the
# hardware does not change the result.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4h_suffix.yaml
run_resumable python code/p4h_suffix_repair.py --config configs/p4h_suffix.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4h suffix repair shard $SLURM_ARRAY_TASK_ID done. =="
