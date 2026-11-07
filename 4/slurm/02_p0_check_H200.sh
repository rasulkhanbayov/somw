#!/bin/bash -l
#SBATCH -J somw_p0_check_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p0_check_H200_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p0_check_H200_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 01:00:00

# PHASE 0 on H200: the same checks, proving the env and the Warp kernels work on the
# H200 nodes before any H200 job is trusted. Can run in parallel with 01.
# Output: results/p0_check_H200/p0_report.json
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p0_check.yaml
python code/selftest.py
python code/p0_check.py --config configs/p0_check.yaml --tag H200
echo "== P0 (H200) done. =="
