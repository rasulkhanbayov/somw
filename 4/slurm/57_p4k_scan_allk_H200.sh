#!/bin/bash -l
#SBATCH -J somw_p4k_scan_allk
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4k_scan_allk_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4k_scan_allk_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 12:00:00
#SBATCH --array=0-3

# PHASE 4k step 3 on 4 x H200: the coarse_cfl m = 4 window scan (base), then V2 at every restart
# point with the shadow continued to the end (coarse-only baseline). Needs: 56.
# Output: results/p4k_scan/, results/p4k_allk/. Costs in the verdict are A100-based, not timed here.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4k_scan.yaml
log_config configs/p4k_gated.yaml
run_resumable python code/p2c_scan.py --config configs/p4k_scan.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
run_resumable python code/p4j_allk.py --config configs/p4k_gated.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4k scan + all-k shard $SLURM_ARRAY_TASK_ID done. =="
