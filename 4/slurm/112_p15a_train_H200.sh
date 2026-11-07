#!/bin/bash -l
#SBATCH -J somw_112_p15a_train_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/112_p15a_train_H200_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/112_p15a_train_H200_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 05:00:00

# PHASE 15a (pre-registered Oct 22, 2026; PLAN.md "Phase 15"): contact-aware generator pilot. Step 2: full fine-tune of base with object-ID embeddings, 6,000 steps, 50% two-object scenes + 50% single-object replay. Needs 111.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p15a_train.yaml
python code/p14_train.py --config configs/p15a_train.yaml --resume
echo "== 112_p15a_train_H200 done. =="
