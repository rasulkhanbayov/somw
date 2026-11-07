#!/bin/bash -l
#SBATCH -J somw_77_p7b_testset
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/77_p7b_testset_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/77_p7b_testset_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-3

# PHASE 7b on 4 x H200 (two shards per GPU): PhysCtrl's test set (100 queries). Coarse shadow + the five
# chain restarts per query, spliced outputs saved to $BIG_ROOT/p7b_traj, PhysCtrl's DeformLoss on GPU for
# sample 0, the ground truth and each restart. Output: results/p7b_allk/. Verdict: p7b_summarize.py (CPU).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p7b_testset.yaml
run_two_shards python code/p4j_allk.py --config configs/p7b_testset.yaml
echo "== 77 p7b shard pair $SLURM_ARRAY_TASK_ID done. =="
