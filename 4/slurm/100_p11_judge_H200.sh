#!/bin/bash -l
#SBATCH -J somw_100_p11_judge_H200
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/100_p11_judge_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/100_p11_judge_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 03:00:00
#SBATCH --array=0

# PHASE 11 step 3 (pre-registered Oct 20, 2026): Qwen2-VL-7B judge with PhysCtrl's prompt on every rendered video.
# BEFORE submitting: on a dev server, python code/p11_judge.py --config configs/p11_judge.yaml --extract-only
# (saves the frames with ffmpeg). Resumable. Verdict: code/p11_summary.py on a dev server.
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
export HF_HOME="$BIG_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$BIG_ROOT/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"   # node /tmp can be full (job 429595)
mkdir -p "$TMPDIR"
export TMP="$TMPDIR" TEMP="$TMPDIR" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
python code/p11_judge.py --config configs/p11_judge.yaml
rm -rf "$TMPDIR"
echo "== 100 p11 judge done. =="
