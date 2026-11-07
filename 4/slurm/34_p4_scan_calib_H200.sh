#!/bin/bash -l
#SBATCH -J somw_p4_scan_calib
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_scan_calib_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_scan_calib_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 12:00:00
#SBATCH --array=0-3

# PHASE 4a data, step 3 on 4 x H200: the true error and seed spread of every calibration
# generation, then the same exhaustive window scan as Phase 2c (base and large, ref and
# coarse, m = 1, 2, 4). Seconds are recorded but cost claims use substep counts, which do
# not depend on the GPU. Needs: 33.
# Output: results/p4_calib_errors/shard_*/probes.jsonl, results/p4_calib_scan/shard_*/scan.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4_calib_gen.yaml
log_config configs/p4_calib_scan.yaml
run_resumable python code/p2b_probe.py --config configs/p4_calib_gen.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
run_resumable python code/p2c_scan.py --config configs/p4_calib_scan.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4 calib errors + scan shard $SLURM_ARRAY_TASK_ID done. =="
