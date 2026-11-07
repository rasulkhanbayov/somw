#!/bin/bash -l
#SBATCH -J somw_67_p4l_scan_ref
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/67_p4l_scan_ref_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/67_p4l_scan_ref_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 06:00:00
#SBATCH --array=0-3

# PHASE 4l-ref (PLAN.md "Phase 4l-ref"): the pre-registered H0 follow-up. The exhaustive m = 4 scan of
# the 334 contact/control scenes at PhysCtrl's REFERENCE setting (sources base, gt_super, gt).
# Two shards run side by side on each GPU (8 shards over 4 GPUs): one MPM process used only ~0.7 GB and
# ~50-70% of an H200 in Phase 4l, so pairing them keeps the GPU busy (Panther guidelines).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
log_config configs/p4l_scan_ref.yaml
T="$SLURM_ARRAY_TASK_ID"
run_resumable python code/p2c_scan.py --config configs/p4l_scan_ref.yaml --shard $((2 * T)) --num-shards 8 &
p1=$!
run_resumable python code/p2c_scan.py --config configs/p4l_scan_ref.yaml --shard $((2 * T + 1)) --num-shards 8 &
p2=$!
rc=0
wait "$p1" || rc=$?
wait "$p2" || rc=$?
echo "== 67 p4l ref scan task $T (shards $((2 * T)), $((2 * T + 1))) done, exit $rc. =="
exit "$rc"
