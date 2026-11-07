#!/bin/bash -l
#SBATCH -J somw_p2a_lift_floor
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2a_lift_floor_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2a_lift_floor_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00
#SBATCH --array=0-3

# PHASE 2a (GO/NO-GO) on 4 x A100: the lift floor. Window defects on GROUND-TRUTH
# trajectories for 7 variants (oracle state, lift from odd frames, dense-time lift,
# no C, no F, k=6, k=24), 6 window starts x 2 lengths, 4 perturbed restarts each.
# Needs: 07 (adds 10 OOD queries per split; runs in-distribution only if 07 is absent).
# Output: results/p2a_lift_floor/shard_*/lift_floor.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p2a_lift_floor.yaml
run_resumable python code/p2a_lift_floor.py --config configs/p2a_lift_floor.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P2a lift floor shard $SLURM_ARRAY_TASK_ID done. =="
