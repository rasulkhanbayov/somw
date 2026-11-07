#!/bin/bash -l
#SBATCH -J somw_p5_real
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p5_real_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p5_real_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 12:00:00
#SBATCH --array=0-0

# STUB. PHASE 5c: real-world checks: gs_physics collisions and the Physics-IQ solid subset.
# Design depends on the Phase 1-2 verdicts, so this script refuses to run.
# TODO before enabling:
#   - Needs the rendering stack (DOWNLOAD_RENDER=1 in 00_setup_env.sh).
#   - VLM judge and user study protocol for video-level results (Phase 6 figures).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
echo "STUB: 42_p5_realworld_TODO_H200.sh is not implemented yet. See PLAN.md." >&2
exit 1
