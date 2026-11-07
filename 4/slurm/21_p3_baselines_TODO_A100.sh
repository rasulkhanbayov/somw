#!/bin/bash -l
#SBATCH -J somw_p3_baselines
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p3_baselines_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p3_baselines_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 24:00:00
#SBATCH --array=0-3

# STUB. PHASE 3b: all baselines at matched total compute (generator + simulator + verifier).
# Design depends on the Phase 1-2 verdicts, so this script refuses to run.
# TODO before enabling:
#   - always-generate (PhysCtrl base/large), always-simulate (upper bound),
#   - uniform / random windows at equal budget, ANCHOR-style threshold trigger,
#   - Hybrid-NWM-style whole-trajectory deferral, best-of-N selection, oracle placement.
#   - Write code/p3_pareto.py producing the error-vs-total-compute Pareto curve and its AUC.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
echo "STUB: 21_p3_baselines_TODO_A100.sh is not implemented yet. See PLAN.md." >&2
exit 1
