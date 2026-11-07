#!/bin/bash -l
#SBATCH -J somw_p4_build_calib2
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_build_calib2_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_build_calib2_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 4a EXTENSION (id_new_4..11), step 1 on 4 x H200: simulate the calibration set, 100 test objects x 4
# fresh in-distribution physics draws (id_new_0..3), with the same Warp MPM as all ground
# truth. Runs on H200s so the A100s stay free for the Phase 2c scan; no timing is measured.
# Output: $DATA_ROOT/calib/id_new_*/*.h5, results/p4_calib_build/shard_*/ood_manifest.json
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4_calib2_build.yaml
run_resumable python code/p2_build_ood.py --config configs/p4_calib2_build.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4 calib build shard $SLURM_ARRAY_TASK_ID done. =="
