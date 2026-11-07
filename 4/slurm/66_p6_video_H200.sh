#!/bin/bash -l
#SBATCH -J somw_p6_video
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/66_p6_video_H200_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/66_p6_video_H200_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80000MB
#SBATCH -A h200_hbku
#SBATCH -q h200_hbku_qos
#SBATCH -t 04:00:00
#SBATCH --array=0-3

# PHASE 6a (fixed Oct 15, 2026; PLAN.md "Phase 6a"): qualitative videos. 10 cases (5 elastic PhysCtrl
# examples x force 1 and 3), 3 renders each (generation / certified / ground truth) with PhysCtrl's
# own SAM -> SV3D -> LGM -> tracking -> DAS pipeline. GPU-heavy (SV3D, CogVideoX-5B).
# Needs: slurm/00b_download_render_models.sh run once (SV3D offline cache). Outputs: $BIG_ROOT/renders/
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" H200
export HF_HOME="$BIG_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
log_config configs/p6_video.yaml
python code/p6_video.py --config configs/p6_video.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
echo "== P6 video shard $SLURM_ARRAY_TASK_ID done. =="
