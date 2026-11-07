#!/bin/bash -l
#SBATCH -J somw_p2c_scan
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2c_scan_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2c_scan_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 12:00:00
#SBATCH --array=0-3

# PHASE 2c on 4 x A100: exhaustive window scan. Every candidate window plus the anchor,
# for base, large and the ground truth, at the reference and the matched-accuracy simulator
# setting, for window lengths m = 1, 2, 4. Reuses the Phase 2 generations. The questions and
# pass rules (fidelity, window length + fusion, placement on contacts, single-threshold
# certificate) are pre-registered in code/p2c_summarize.py and PLAN.md.
# Estimate (not measured): about 10 GPU-h, so about 3 h wall on 4 A100.
# Output: results/p2c_scan/shard_*/scan.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p2c_scan.yaml
run_resumable python code/p2c_scan.py --config configs/p2c_scan.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P2c scan shard $SLURM_ARRAY_TASK_ID done. =="
