#!/bin/bash -l
#SBATCH -J somw_104_p14a_build_A100
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/104_p14a_build_A100_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/104_p14a_build_A100_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 14a (pre-registered Oct 22, 2026; PLAN.md "Phase 14"): simulator-distillation pilot. Step 1: the 3,000-candidate pool (sample/ objects x 30 aug_ draws) at the reference setting; two shards per GPU.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p14a_build.yaml
run_two_shards python code/p2_build_ood.py --config configs/p14a_build.yaml
echo "== 104_p14a_build_A100 task $SLURM_ARRAY_TASK_ID done. =="
