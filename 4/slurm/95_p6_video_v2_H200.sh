#!/bin/bash -l
#SBATCH -J somw_95_p6_video_v2
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/95_p6_video_v2_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/95_p6_video_v2_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0-1

# Paper re-render (Oct 19, 2026): the 10 Phase 6a cases with the Phase 9 pipeline (configs/p6_video_v2.yaml).
# One new variant per case (certified_v2); accepted or exact-fallback cases copy the existing render. The Phase 6a
# preprocessing (_prep) is reused; Phase 6a records are not overwritten. About 1 GPU-hour in total.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
export HF_HOME="$BIG_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Job 429595 shard 1 died on crirdchpxd006: the node's /tmp was full ("No space left on device"), so the C compiler
# behind torch.compile (DAS transformer) aborted. Keep every temporary and compile cache on the share instead.
export TMPDIR="$BIG_ROOT/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
log_config configs/p6_video_v2.yaml
python code/p6_video.py --config configs/p6_video_v2.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 2
rm -rf "$TMPDIR"          # not a trap: _common.sh owns the EXIT trap (exit code + GPU log)
echo "== 95 P6 video v2 shard $SLURM_ARRAY_TASK_ID done. =="
