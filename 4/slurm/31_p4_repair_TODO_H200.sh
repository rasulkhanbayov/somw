#!/bin/bash -l
#SBATCH -J somw_p4_repair
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_repair_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4_repair_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 24:00:00
#SBATCH --array=0-3

# STUB. PHASE 4b: repair by constrained re-sampling with verified windows fixed.
# Design depends on the Phase 1-2 verdicts, so this script refuses to run.
# TODO before enabling:
#   - Check whether PhysCtrl's DiT supports frame inpainting; if not, fine-tune with
#   -   random frame-conditioning masks (LoRA) on the Phase 3 data.
#   - Compare against simple replacement of probed frames; show backward propagation
#   -   (a fix at a collision changes earlier frames).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
echo "STUB: 31_p4_repair_TODO_H200.sh is not implemented yet. See PLAN.md." >&2
exit 1
