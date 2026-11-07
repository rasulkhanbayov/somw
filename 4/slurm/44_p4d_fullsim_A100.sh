#!/bin/bash -l
#SBATCH -J somw_p4d_fullsim
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4d_fullsim_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p4d_fullsim_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3


# PHASE 4d data on 4 x A100: one full rollout per query from the true initial state at the
# fallback fidelity (coarse_cfl), for the Phase 2c queries and the calibration set, with A100
# timing. This is the fallback of certify-or-simulate. Pre-registered: code/p4d_pipeline.py,
# PLAN.md "Phase 4d". Estimate (not measured): about 1,500 rollouts x ~1 s, under 1 GPU-h.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p4d_fullsim.yaml
run_resumable python code/p4d_fullsim.py --config configs/p4d_fullsim.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P4d fullsim shard $SLURM_ARRAY_TASK_ID done. =="
