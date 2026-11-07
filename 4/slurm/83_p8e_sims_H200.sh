#!/bin/bash -l
#SBATCH -J somw_83_p8e_sims_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/83_p8e_sims_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/83_p8e_sims_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 05:00:00
#SBATCH --array=0-3

# PILOT E1 (exploratory; PLAN.md "Pilots E1-E3"): the stiff-material regime. Step 3: simulator ladder, window scan, gated-V2 restarts. Needs 82.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
for c in p8e_full p8e_scan p8e_allk; do log_config configs/$c.yaml; done
run_two_shards python code/p4d_fullsim.py --config configs/p8e_full.yaml
run_two_shards python code/p2c_scan.py --config configs/p8e_scan.yaml
run_two_shards python code/p4j_allk.py --config configs/p8e_allk.yaml
echo "== 83 p8e sims pair $SLURM_ARRAY_TASK_ID done. =="
