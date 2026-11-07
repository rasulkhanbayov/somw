#!/bin/bash -l
#SBATCH -J somw_p1_cost_A100
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p1_cost_A100_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p1_cost_A100_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00

# PHASE 1a (GO/NO-GO input) on A100: wall-clock of PhysCtrl generation (base: 4 DDIM
# steps, large: 25; batch 1/8/32) against Warp MPM rollouts in 5 regimes x 3 timing
# modes (as_is / nosync / graph), plus the cost of one probe window.
# Needs: 01 finished (test-set manifest). Timings assume the GPU is not shared.
# Output: results/p1_cost_A100/timings.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p1_cost.yaml
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv || true
run_resumable python code/p1_cost.py --config configs/p1_cost.yaml --tag A100
echo "== P1 cost (A100) done. =="
