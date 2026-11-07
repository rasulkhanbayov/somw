#!/bin/bash -l
#SBATCH -J somw_82_p8e_generate_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/82_p8e_generate_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/82_p8e_generate_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 01:00:00
#SBATCH --array=0-3

# PILOT E1 (exploratory; PLAN.md "Pilots E1-E3"): the stiff-material regime. Step 2: PhysCtrl base generations (8 samples). Needs 81.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p8e_gen.yaml
python code/p2b_generate.py --config configs/p8e_gen.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== 82 p8e generate task $SLURM_ARRAY_TASK_ID done. =="
