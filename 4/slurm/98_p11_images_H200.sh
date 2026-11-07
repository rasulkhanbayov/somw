#!/bin/bash -l
#SBATCH -J somw_98_p11_images_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/98_p11_images_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/98_p11_images_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 01:00:00
#SBATCH --array=0

# PHASE 11 step 1 (pre-registered Oct 20, 2026; PLAN.md "Phase 11"): 25 synthetic object images (FLUX.1-schnell) and their
# configs -> $BIG_ROOT/p11_examples. Minutes. Needs slurm/00c_download_p11_models.sh run once.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
export HF_HOME="$BIG_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$BIG_ROOT/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"   # node /tmp can be full (job 429595)
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
python code/p11_images.py
rm -rf "$TMPDIR"
echo "== 98 p11 images done. =="
