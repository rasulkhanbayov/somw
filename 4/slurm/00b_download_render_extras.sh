#!/bin/bash
# One-time, on a setup/dev server with internet (e.g. crimv3mgpu015), NOT on the login node:
#     cd ~/cvpr_physics_plausibility/4 && bash slurm/00b_download_render_extras.sh
# PhysCtrl's image-to-video pipeline downloads SV3D (chenguolin/sv3d-diffusers) at run time.
# GPU nodes may have no internet, so it is fetched here into $BIG_ROOT/hf_cache (the share).
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/cluster.env
set +u; source "$CONDA_DIR/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"; set -u
export HF_HOME="$BIG_ROOT/hf_cache"
mkdir -p "$HF_HOME"
huggingface-cli download chenguolin/sv3d-diffusers
echo "== SV3D in $HF_HOME =="; du -sh "$HF_HOME"
