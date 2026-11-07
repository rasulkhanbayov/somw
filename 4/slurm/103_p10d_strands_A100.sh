#!/bin/bash -l
#SBATCH -J somw_103_p10d
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/103_p10d_strands_A100_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/103_p10d_strands_A100_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00

# PHASE 10d (pre-registered Oct 22, 2026; PLAN.md "Phase 10d"): are the reference's strands numerical? The 23 shifted
# Phase 11 items at dt/2 and dt/4 (stored body) and on a body-preserving filled body at grid 125 and 250.
# About 1 A100-hour (estimate). Finished runs are skipped on restart. Verdict on a dev server:
#   python code/p10d_strands.py --summarize
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p10d_strands.yaml
run_resumable python code/p10d_strands.py --config configs/p10d_strands.yaml
echo "== 103 p10d done. =="
