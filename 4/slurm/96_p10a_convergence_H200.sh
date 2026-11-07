#!/bin/bash -l
#SBATCH -J somw_96_p10a
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/96_p10a_convergence_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/96_p10a_convergence_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 10a (pre-registered Oct 20, 2026; PLAN.md "Phase 10"): is the reference converged? The reference setting at half
# its time step on every Phase 9 query (600 single objects, 168 contact scenes) and at a quarter on a fixed subset.
# Two shards per GPU; about 3.8 GPU-h in total. Analysis on a dev server: p10a_convergence.py --config configs/p10a_summary.yaml
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
for c in p10a_half p10a_quarter p10ac_half p10ac_quarter; do log_config configs/$c.yaml; done
for c in p10a_half p10ac_half p10a_quarter p10ac_quarter; do
    run_two_shards python code/p4d_fullsim.py --config configs/$c.yaml
done
echo "== 96 p10a pair $SLURM_ARRAY_TASK_ID done. =="
