#!/bin/bash -l
#SBATCH -J somw_p2c_scan_resume
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2c_scan_resume_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2c_scan_resume_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00
#SBATCH --array=0-3

# PHASE 2c, RESUME of 14 (job 419393, which stopped at the 50-restart cap while crash-
# looping on ood_E_high at the fixed coarse dt: a CFL violation). Keeps every row already
# written (--resume from the first start), skips windows with CFL > cfl_gate (scored as
# blown, the +inf rule), and adds the CFL-adaptive fidelity 'coarse_cfl' for question A'.
# Must use the same 4 shards as 14.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p2c_scan.yaml
export MAX_CUDA_RESTARTS=300
run_resumable python code/p2c_scan.py --config configs/p2c_scan.yaml --resume \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P2c scan (resume) shard $SLURM_ARRAY_TASK_ID done. =="
