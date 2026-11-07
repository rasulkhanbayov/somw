#!/bin/bash -l
#SBATCH -J somw_75_p4o_scan_allk_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/75_p4o_scan_allk_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/75_p4o_scan_allk_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 06:00:00
#SBATCH --array=0-3

# PHASE 4o, shadow-timed fidelity switch on fresh contact scenes (pre-registered Oct 16, 2026;
# PLAN.md "Phase 4o"). Step 4: coarse scan (base) and V2 restarts (chain, event, generation- and shadow-timed points); two shards per GPU. Needs 70. Large outputs go to $BIG_ROOT.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4o_scan.yaml
log_config configs/p4o_allk.yaml
run_two_shards python code/p2c_scan.py --config configs/p4o_scan.yaml
run_two_shards python code/p4j_allk.py --config configs/p4o_allk.yaml
echo "== 71_p4o_scan_allk_H200 task $SLURM_ARRAY_TASK_ID done. =="
