#!/bin/bash -l
#SBATCH -J somw_78_p7d_coarse
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/78_p7d_contact_coarse_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/78_p7d_contact_coarse_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-3

# PHASE 7d on 4 x H200 (two shards per GPU): coarse_cfl rollout of every 4o contact/control scene from its
# true initial state to the end, positions saved to $BIG_ROOT/coarse_traj_4o. Verdict: p7d_contact.py (CPU).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p7d_contact_coarse.yaml
run_two_shards python code/p4j_allk.py --config configs/p7d_contact_coarse.yaml
echo "== 78 p7d shard pair $SLURM_ARRAY_TASK_ID done. =="
