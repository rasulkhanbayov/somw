#!/bin/bash -l
#SBATCH -J somw_79_p8a_fidelity
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/79_p8a_fidelity_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/79_p8a_fidelity_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PILOT 8a (exploratory, development data) on 4 x H200, two shards per GPU:
#   A  full rollouts at M1/M2/M3 on the 2c pool + 4l scenes        -> results/p8a_full/
#   B1 gated-V2 restarts, coarse shadow, M2 suffix, 2c pool          -> results/p8a_restart_b1/
#   B2 gated-V2 restarts, M1 shadow, M2 suffix, 2c pool              -> results/p8a_restart_b2/
# Analysis: code/p8a_pilot.py on a dev server.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
for c in p8a_full p8a_restart_b1 p8a_restart_b2; do log_config configs/$c.yaml; done
run_two_shards python code/p4d_fullsim.py --config configs/p8a_full.yaml
run_two_shards python code/p4j_allk.py --config configs/p8a_restart_b1.yaml
run_two_shards python code/p4j_allk.py --config configs/p8a_restart_b2.yaml
echo "== 79 p8a shard pair $SLURM_ARRAY_TASK_ID done. =="
