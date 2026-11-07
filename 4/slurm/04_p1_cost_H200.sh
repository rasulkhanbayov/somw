#!/bin/bash -l
#SBATCH -J somw_p1_cost_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p1_cost_H200_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p1_cost_H200_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00

# PHASE 1a on H200: identical to 03, on the other hardware. Ratios are compared only
# within one GPU type, and the verdict requires a regime to pass on both.
# Needs: 01 (or 02) finished.
# Output: results/p1_cost_H200/timings.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p1_cost.yaml
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv || true
run_resumable python code/p1_cost.py --config configs/p1_cost.yaml --tag H200
echo "== P1 cost (H200) done. =="
