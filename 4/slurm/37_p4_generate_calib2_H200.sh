#!/bin/bash -l
#SBATCH -J somw_p4_generate_calib2
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_generate_calib2_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_generate_calib2_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 4a EXTENSION (id_new_4..11), step 2 on 4 x H200: 8 samples per calibration query from base and large.
# Needs: 32. Output: $DATA_ROOT/generations/<model>/id_new_*/<qid>.npz
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4_calib2_gen.yaml
python code/p2b_generate.py --config configs/p4_calib2_gen.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4 calib generate shard $SLURM_ARRAY_TASK_ID done. =="
