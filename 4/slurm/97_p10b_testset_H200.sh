#!/bin/bash -l
#SBATCH -J somw_97_p10b
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/97_p10b_testset_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/97_p10b_testset_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 01:00:00
#SBATCH --array=0-3

# PHASE 10b (pre-registered Oct 20, 2026; PLAN.md "Phase 10"): PhysCtrl's own metrics on its test set with the final
# pipeline (M1 shadow + the five M2 chain restarts per query; spliced outputs to $BIG_ROOT/p10b_traj; DeformLoss on GPU).
# Two shards per GPU. Summary on a dev server: p7b_summarize.py --config configs/p10b_testset.yaml
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p10b_testset.yaml
run_two_shards python code/p4j_allk.py --config configs/p10b_testset.yaml
echo "== 97 p10b shard pair $SLURM_ARRAY_TASK_ID done. =="
