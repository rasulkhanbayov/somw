#!/bin/bash -l
#SBATCH -J somw_p2b_probe
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2b_probe_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2b_probe_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00
#SBATCH --array=0-3

# PHASE 2b part 2 (GO/NO-GO) on 4 x A100: for each generated trajectory, true error vs
# ground truth, seed-spread and motion baselines, and window defects under 6 placement
# strategies (uniform, random, event, disagree, proposal, and the oracle ceiling).
# Needs: 09 finished. Resumable: skips (query, model) pairs already written.
# Output: results/p2b_probe/shard_*/probes.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p2b.yaml
run_resumable python code/p2b_probe.py --config configs/p2b.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P2b probe shard $SLURM_ARRAY_TASK_ID done. =="
