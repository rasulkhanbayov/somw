#!/bin/bash -l
#SBATCH -J somw_p4k_build
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4k_build_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4k_build_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 4k step 1 on 4 x H200 (pre-registered Oct 14, 2026): ground truth for the fresh queries,
# 100 test objects x 8 new in-distribution physics draws (id_new_12..19), reference setting.
# Output: $DATA_ROOT/calib/id_new_1[2-9]/*.h5, results/p4k_build/shard_*/ood_manifest.json
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4k_build.yaml
run_resumable python code/p2_build_ood.py --config configs/p4k_build.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4k build shard $SLURM_ARRAY_TASK_ID done. =="
