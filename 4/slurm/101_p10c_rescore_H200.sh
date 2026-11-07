#!/bin/bash -l
#SBATCH -J somw_101_p10c
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/101_p10c_rescore_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/101_p10c_rescore_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 05:00:00
#SBATCH --array=0-3

# PHASE 10c (pre-registered Oct 21, 2026; PLAN.md "Phase 10c"): positions of every Phase 9 pipeline and of the reference
# at dt, dt/2, dt/4 on all 732 Phase 9 queries -> $BIG_ROOT/p10c_traj. Two shards per GPU; about 8-9 GPU-h in total.
# Needs results/p10c_decisions (written on a dev server). Verdict: code/p10c_summary.py on a dev server.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p10c_rescore.yaml
run_two_shards python code/p10c_rescore.py --config configs/p10c_rescore.yaml --resume   # finished queries are skipped
echo "== 101 p10c pair $SLURM_ARRAY_TASK_ID done. =="
