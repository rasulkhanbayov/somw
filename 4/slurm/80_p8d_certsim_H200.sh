#!/bin/bash -l
#SBATCH -J somw_80_p8d_certsim
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/80_p8d_certsim_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/80_p8d_certsim_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PILOT 8d (exploratory, development data) on 4 x H200, two shards per GPU: M1 proposal rollouts, M3 check windows
# from M1's own state at k = 3, 11, 19, and reference suffixes (not for the id_new_0..3 threshold queries).
# Output: results/p8d_certsim/. Analysis: code/p8d_pilot.py on a dev server.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p8d_certsim.yaml
run_two_shards python code/p8d_certsim.py --config configs/p8d_certsim.yaml
echo "== 80 p8d shard pair $SLURM_ARRAY_TASK_ID done. =="
