#!/bin/bash -l
#SBATCH -J somw_p2b_generate
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2b_generate_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2b_generate_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-1

# PHASE 2b part 1 on 2 x H200: 8 samples per query from base and large, for 100
# in-distribution + 4 x 50 OOD queries. Generation is cheap; two tasks are plenty.
# Resumable: existing .npz files are skipped.
# Needs: 07 finished (otherwise only in-distribution queries; rerun after 07 to add OOD).
# Output: $DATA_ROOT/generations/<model>/<split>/<qid>.npz
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p2b.yaml
python code/p2b_generate.py --config configs/p2b.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 2
echo "== P2b generate shard $SLURM_ARRAY_TASK_ID done. =="
