#!/bin/bash -l
#SBATCH -J somw_p4j_allk
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4j_allk_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4j_allk_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 12:00:00
#SBATCH --array=0-3

# PHASE 4j on 4 x H200 (pre-registered Oct 13, 2026): V2 suffix repair at every chain restart
# point k in {3, 7, 11, 15, 19} for the 1,200 base calibration queries (one coarse shadow + five
# reference suffixes per query). Expected ~7 GPU-hours in total, ~2 h wall-clock. Costs in the
# verdict are shares of measured A100 rollouts, so the GPU type does not change it.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4j_confirm.yaml
run_resumable python code/p4j_allk.py --config configs/p4j_confirm.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4j all-k shard $SLURM_ARRAY_TASK_ID done. =="
