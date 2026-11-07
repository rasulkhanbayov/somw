#!/bin/bash -l
#SBATCH -J somw_93_p9c_sims_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/93_p9c_sims_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/93_p9c_sims_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 06:00:00
#SBATCH --array=0-3

# PHASE 9 (pre-registered Oct 19, 2026; PLAN.md "Phase 9"): confirmation of ours-v2 on fresh data. Contact, step 4: coarse scan, restarts (M3 suffix), baseline rollouts. Needs 92.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
for c in p9c_scan p9c_allk p9c_full; do log_config configs/$c.yaml; done
run_two_shards python code/p2c_scan.py --config configs/p9c_scan.yaml
run_two_shards python code/p4j_allk.py --config configs/p9c_allk.yaml
run_two_shards python code/p4d_fullsim.py --config configs/p9c_full.yaml
echo "== 93 p9c sims pair $SLURM_ARRAY_TASK_ID done. =="
