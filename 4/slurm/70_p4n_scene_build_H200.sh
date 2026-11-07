#!/bin/bash -l
#SBATCH -J somw_70_p4n_scene_build_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/70_p4n_scene_build_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/70_p4n_scene_build_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-3

# PHASE 4n, event-aware fidelity repair on fresh contact scenes (pre-registered Oct 16, 2026;
# PLAN.md "Phase 4n"). Step 3: scene design, joint ground truth, composed generations; two shards per GPU. Needs 69. Large outputs go to $BIG_ROOT.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4n_scene_build.yaml
run_two_shards python code/p4l_mo_build.py --config configs/p4n_scene_build.yaml
echo "== 70_p4n_scene_build_H200 task $SLURM_ARRAY_TASK_ID done. =="
