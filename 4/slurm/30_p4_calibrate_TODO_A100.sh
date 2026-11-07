#!/bin/bash -l
#SBATCH -J somw_p4_calib
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_calib_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_calib_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 08:00:00
#SBATCH --array=0-0

# STUB. PHASE 4a: conformal calibration of the accept threshold and coverage audit.
# Design depends on the Phase 1-2 verdicts, so this script refuses to run.
# TODO before enabling:
#   - Split-conformal threshold on the calibration slice (score = best signal from Phase 2).
#   - Report empirical coverage vs alpha in-distribution; report OOD coverage as an
#   - empirical finding only (exchangeability does not hold there).
#   - Risk-coverage curves; recalibration after every model update.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
echo "STUB: 30_p4_calibrate_TODO_A100.sh is not implemented yet. See PLAN.md." >&2
exit 1
