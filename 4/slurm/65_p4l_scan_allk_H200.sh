#!/bin/bash -l
#SBATCH -J somw_65_p4l_scan_allk_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/65_p4l_scan_allk_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/65_p4l_scan_allk_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 16:00:00
#SBATCH --array=0-3

# PHASE 4l, multi-object contact benchmark (pre-registered Oct 15, 2026; PLAN.md "Phase 4l"). Step 4: true error / seed spread, exhaustive coarse_cfl m = 4 scan (base, gt_super, gt), V2 restarts at every frame 1..22. Needs 64.
# Large outputs go to $BIG_ROOT (the hbku-research share).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4l_errors.yaml
log_config configs/p4l_scan.yaml
log_config configs/p4l_allk.yaml
run_resumable python code/p2b_probe.py --config configs/p4l_errors.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
run_resumable python code/p2c_scan.py --config configs/p4l_scan.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
run_resumable python code/p4j_allk.py --config configs/p4l_allk.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== 65_p4l_scan_allk_H200 shard $SLURM_ARRAY_TASK_ID done. =="
