#!/bin/bash -l
#SBATCH -J somw_p5_rigid
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p5_rigid_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p5_rigid_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 24:00:00
#SBATCH --array=0-3

# STUB. PHASE 5b: rigid-contact track on RigidBench (ground-truth contact events).
# Design depends on the Phase 1-2 verdicts, so this script refuses to run.
# TODO before enabling:
#   - Render the 5,000-example training manifest (Blender 4.5.9) or use eval set only;
#   - probe with the SAME Bullet build Blender uses, or report the simulator mismatch;
#   - Kabsch lift for rigid state; ATE-3D metric.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
echo "STUB: 41_p5_rigidbench_TODO_A100.sh is not implemented yet. See PLAN.md." >&2
exit 1
