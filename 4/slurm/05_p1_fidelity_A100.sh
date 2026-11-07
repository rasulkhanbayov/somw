#!/bin/bash -l
#SBATCH -J somw_p1_fidelity
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p1_fidelity_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p1_fidelity_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00
#SBATCH --array=0-3

# PHASE 1b on 4 x A100 (array): simulator fidelity sweep (5 substep sizes x 3 grids) on
# 40 elastic test queries (10 per task), plus generator error on the same queries.
# Answers "is PhysCtrl's simulator setting overkill?" by finding the cheapest simulator
# that is as accurate as the generator (the matched-accuracy cost).
# Needs: 01. Independent of 03/04; can run at the same time on other GPUs.
# Output: results/p1_fidelity_A100/shard_{0..3}/fidelity.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p1_fidelity.yaml
run_resumable python code/p1_fidelity.py --config configs/p1_fidelity.yaml --tag A100 \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P1 fidelity shard $SLURM_ARRAY_TASK_ID done. =="
