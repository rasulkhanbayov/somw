#!/bin/bash -l
#SBATCH -J somw_p2b_probe_v2
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2b_probe_v2_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p2b_probe_v2_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 16:00:00
#SBATCH --array=0-3

# PHASE 2b-v2 on 4 x A100: the probe rerun with lift_noF (F = identity on restart),
# chosen from ground-truth evidence after v1's F-lift blew up the simulator on 36-58% of
# ground-truth windows. Reuses v1's generations (no 09 needed); thresholds and the +inf
# blow-up rule are unchanged. Falsifier: same G2a/G2b as v1 (PLAN.md, Phase 2).
# Needs: results of 07-09 (already done). Output: results/p2b_probe_v2/shard_*/probes.jsonl
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p2b_v2.yaml
run_resumable python code/p2b_probe.py --config configs/p2b_v2.yaml \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P2b-v2 probe shard $SLURM_ARRAY_TASK_ID done. =="
