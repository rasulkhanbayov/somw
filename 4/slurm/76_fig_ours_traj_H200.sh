#!/bin/bash -l
#SBATCH -J somw_76_fig_ours_traj
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/76_fig_ours_traj_H200_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/76_fig_ours_traj_H200_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 00:30:00

# Deck figures (Oct 16, 2026): recompute the positions of OUR output for the 5 rule-selected qualitative
# cases (results/fig_qualitative/deck_selection.json). Two MPM restarts; a few minutes on one GPU.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
python code/fig_ours_traj.py
echo "== 76 fig_ours_traj done. =="
