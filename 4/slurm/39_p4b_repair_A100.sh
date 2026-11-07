#!/bin/bash -l
#SBATCH -J somw_p4b_repair
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4b_repair_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4b_repair_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00
#SBATCH --array=0-3


# PHASE 4b on 4 x A100 (idle while the H200s run the calibration extension; the 80 GB A100 holds both generators): training-free repair by constrained re-sampling of the Phase 2c
# generations (base and large). Constraints are the anchor + certificate windows, simulated at
# the reference setting. Compares replace (R1) against constrained re-sampling (R2), with
# re-sampling-alone (R0p) and anchor-only (R3) controls. Starts by checking that our sampler
# reproduces PhysCtrl's pipeline. Pre-registered: code/p4b_repair.py, PLAN.md "Phase 4b".
# Estimate (not measured): about 2 GPU-h, dominated by large (25 DDIM steps x 3 samples).
# Output: results/p4b_repair/shard_*/repair.jsonl; then Claude runs p4b_summarize.py here.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p4b_repair.yaml
run_resumable python code/p4b_repair.py --config configs/p4b_repair.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4b repair shard $SLURM_ARRAY_TASK_ID done. =="
