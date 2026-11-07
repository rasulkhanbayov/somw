#!/bin/bash -l
#SBATCH -J somw_111_p15a_build_A100
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/111_p15a_build_A100_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/111_p15a_build_A100_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-3

# PHASE 15a (pre-registered Oct 22, 2026; clarified Oct 24; PLAN.md "Phase 15"): contact-aware generator pilot. Step 1: joint two-object rollouts (4,096 particles, reference setting) of the 1,700 designed scenes + model files -> $BIG_ROOT/p15; two shards per GPU. Needs results/p15a_scenes (dev server).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p15a_build.yaml
run_two_shards python code/p15_build.py --config configs/p15a_build.yaml
echo "== 111_p15a_build_A100 task $SLURM_ARRAY_TASK_ID done. =="
