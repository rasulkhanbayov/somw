#!/bin/bash -l
#SBATCH -J somw_p2_build_ood
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2_build_ood_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2_build_ood_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 2 prep on 4 x A100: re-simulate 100 elastic test objects under 4 OOD physics
# settings (E 10^3.5, E 10^7.5, force x3, gravity drop onto a floor) with the SAME Warp
# MPM that produced the training data. Writes PhysCtrl-format HDF5 to $DATA_ROOT/ood/.
# Submit after the Phase 1 verdict is GO (or after you decide to proceed regardless).
# Output: results/p2_build_ood/shard_*/{ood_manifest.json, failures.json}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p2_ood.yaml
run_resumable python code/p2_build_ood.py --config configs/p2_ood.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P2 OOD shard $SLURM_ARRAY_TASK_ID done. =="
