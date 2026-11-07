#!/bin/bash -l
#SBATCH -J somw_99_p11_render_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/99_p11_render_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/99_p11_render_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 08:00:00
#SBATCH --array=0-3

# PHASE 11 step 2 (pre-registered Oct 20, 2026): PhysCtrl / ours (Phase 9 pipeline) / reference renders of the 25 images
# at force_coeff 0.06 and 0.40 (configs/p11_render.yaml), sharded by object. About 15 GPU-h in total. Needs 98.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
export HF_HOME="$BIG_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$BIG_ROOT/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"   # node /tmp can be full (job 429595)
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
log_config configs/p11_render.yaml
python code/p6_video.py --config configs/p11_render.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
rm -rf "$TMPDIR"
echo "== 99 p11 render shard $SLURM_ARRAY_TASK_ID done. =="
