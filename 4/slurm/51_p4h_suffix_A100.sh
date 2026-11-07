#!/bin/bash -l
#SBATCH -J somw_p4h_suffix
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4h_suffix_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4h_suffix_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3


# PHASE 4h on 4 x A100: certified-prefix repair. For each Phase 2c base query, find the first
# failing window in the temporal chain (defects read from the Phase 2c scan) and re-simulate
# only the suffix from the lifted state at the reference setting, plus the same restart from
# ground truth (the suffix lift floor). Pre-registered: PLAN.md "Phase 4h".
# Estimate (not measured): at most 600 suffix rollouts x ~4 s, under 1 GPU-h.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p4h_suffix.yaml
run_resumable python code/p4h_suffix_repair.py --config configs/p4h_suffix.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4h suffix repair shard $SLURM_ARRAY_TASK_ID done. =="
