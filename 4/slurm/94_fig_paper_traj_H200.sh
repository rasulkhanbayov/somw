#!/bin/bash -l
#SBATCH -J somw_94_fig_paper_traj
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/94_fig_paper_traj_H200_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/94_fig_paper_traj_H200_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 00:30:00

# Paper figures (Oct 19, 2026): recompute our output (Phase 9 pipeline) and an always-M1 rollout for the 5 cases in
# results/fig_qualitative/paper_selection.json (fixed by code/fig_paper_select.py before any picture). A few MPM
# rollouts; minutes on one GPU. Each result is checked against the stored Phase 9 errors (raises on mismatch).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
python code/fig_ours_traj.py --selection results/fig_qualitative/paper_selection.json --prefix paper_
echo "== 94 fig_paper_traj done. =="
