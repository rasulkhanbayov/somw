#!/bin/bash -l
#SBATCH -J somw_p4_scan_calib_cfl
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_scan_calib_cfl_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_scan_calib_cfl_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 06:00:00
#SBATCH --array=0-3

# PHASE 4a data, step 3b on 4 x H200: add the CFL-adaptive fidelity 'coarse_cfl' to the
# finished calibration scan (job 419396). --resume keeps every existing row.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4_calib_scan.yaml
export MAX_CUDA_RESTARTS=300
run_resumable python code/p2c_scan.py --config configs/p4_calib_scan.yaml --resume \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4 calib scan (coarse_cfl) shard $SLURM_ARRAY_TASK_ID done. =="
