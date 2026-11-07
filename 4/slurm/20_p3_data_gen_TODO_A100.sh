#!/bin/bash -l
#SBATCH -J somw_p3_data
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p3_data_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p3_data_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 24:00:00
#SBATCH --array=0-3

# STUB. PHASE 3a: generate the query stream and fine-tuning data with PhysCtrl's own
# generate_mpm_data.py (elastic) and generate_rigid_data.py (Genesis, rigid),
# including a multi-object contact split (extend their scripts).
# Design depends on the Phase 1-2 verdicts, so this script refuses to run.
# TODO before enabling:
#   - Decide the main regime from results/p1_summary/SUMMARY.md (the one that passed).
#   - Download Objaverse meshes for held-out categories (dataset_toolkits/download.py).
#   - Write code/p3_stream.py: splits ID / param-OOD / geometry-OOD / force-OOD / multi-object;
#   -   per query store full GT simulation, 4 perturbed simulations, contact-event labels,
#   -   calibration-vs-test flag (see proposal 'Query stream design').
#   - Replay-determinism check for Genesis (rigid) like p0 does for MPM.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
echo "STUB: 20_p3_data_gen_TODO_A100.sh is not implemented yet. See PLAN.md." >&2
exit 1
