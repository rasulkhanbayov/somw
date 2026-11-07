#!/bin/bash -l
#SBATCH -J somw_p4g_train_v2
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_train_v2_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_train_v2_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 12:00:00

# PHASE 4g-v2 on 1 x H200: base frozen; temporal-attention LoRA + known-frame embedding,
# distillation to the frozen base (frozen config configs/p4g_train_v2.yaml: 10,000 steps,
# batch 8, lr 1e-4). Starts with an
# identity check (the wrapper equals PhysCtrl at init). Resumable: resubmit to continue from
# the latest checkpoint. Estimate (not measured): about 3 H200-h (two forward passes per step).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4g_train_v2.yaml
if [ -f "$DATA_ROOT/checkpoints/fc_base_v2/latest.txt" ]; then RES=--resume; else RES=; fi
python code/p4g_train_fc.py --config configs/p4g_train_v2.yaml $RES
echo "== P4g-v2 training done. =="
