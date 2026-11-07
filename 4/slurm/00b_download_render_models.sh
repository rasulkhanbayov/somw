#!/bin/bash
# One-time, on a dev server with internet (10.4.64.25-28, e.g. crimv3mgpu015), NOT the login node:
#     cd ~/cvpr_physics_plausibility/4 && bash slurm/00b_download_render_models.sh
# Fixes two package versions the video pipeline needs, then pre-downloads the one model PhysCtrl's
# video pipeline fetches at run time (SV3D, multi-view images for the LGM reconstruction) into the
# hbku-research share, so render jobs run offline.
# (VGGT and ZoeDepth are only used for non-elastic materials / tracking from video: not needed.)
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/cluster.env
set +u; source "$CONDA_DIR/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"; set -u
# Two pins the video pipeline needs (found Oct 15, 2026 by importing PhysCtrl's inference.py on CPU):
#   kiui 0.3.5 ships an empty kiui/typing.py (66 bytes in its own RECORD), so every kiui import fails;
#   setuptools >= 81 no longer provides pkg_resources, which PhysCtrl's track visualizer imports.
pip install "kiui==0.3.4" "setuptools==75.8.0"
python -c "import kiui.op, pkg_resources; print('kiui + pkg_resources OK')"
export HF_HOME="$BIG_ROOT/hf_cache"
mkdir -p "$HF_HOME"
huggingface-cli download chenguolin/sv3d-diffusers
echo "== SV3D cached under $HF_HOME =="
du -sh "$HF_HOME"
