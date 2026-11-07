#!/bin/bash -l
#SBATCH -J somw_86_p8g_sims_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/86_p8g_sims_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/86_p8g_sims_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 05:00:00
#SBATCH --array=0-3

# PILOT E1b (exploratory; PLAN.md "Pilot E1b"): the stiff-material regime. Step 3: simulator ladder, window scan, gated-V2 restarts. Needs 82.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
for c in p8g_full p8g_scan p8g_allk; do log_config configs/$c.yaml; done
run_two_shards python code/p4d_fullsim.py --config configs/p8g_full.yaml
run_two_shards python code/p2c_scan.py --config configs/p8g_scan.yaml
run_two_shards python code/p4j_allk.py --config configs/p8g_allk.yaml
echo "== 86 p8g sims pair $SLURM_ARRAY_TASK_ID done. =="
