#!/bin/bash -l
#SBATCH -J somw_p4g_train
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_train_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4g_train_%j.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 12:00:00

# PHASE 4g on 1 x H200: fine-tune PhysCtrl base into the frame-conditioned generator
# (frozen config configs/p4g_train.yaml: 20,000 steps, batch 8, lr 5e-5). Starts with an
# identity check (the wrapper equals PhysCtrl at init). Resumable: resubmit to continue from
# the latest checkpoint. Estimate (not measured): 2-4 H200-h.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4g_train.yaml
if [ -f "$DATA_ROOT/checkpoints/fc_base_v1/latest.txt" ]; then RES=--resume; else RES=; fi
python code/p4g_train_fc.py --config configs/p4g_train.yaml $RES
echo "== P4g training done. =="
