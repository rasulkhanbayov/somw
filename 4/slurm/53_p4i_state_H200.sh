#!/bin/bash -l
#SBATCH -J somw_p4i_state
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4i_state_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4i_state_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 10:00:00
#SBATCH --array=0-3

# PHASE 4i on 4 x H200 (pre-registered Oct 13, 2026): full-state restarts for the 135 Phase 4h
# suffix queries. V1 stable F-lift (+ its ground-truth floor), V2 coarse-shadow state. Costs are
# shares of measured A100 rollouts (configs/p4i_state.yaml), so the GPU type does not change the
# verdict. Needs results/p4i_f_bounds/bounds.json (computed on CPU before submission).
# Expected: under 1 GPU-hour in total (4h took ~10 min per shard with two suffix rollouts per query).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4i_state.yaml
run_resumable python code/p4i_state_repair.py --config configs/p4i_state.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4i state repair shard $SLURM_ARRAY_TASK_ID done. =="
