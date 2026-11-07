#!/bin/bash
# Run ONCE, interactively, on a shared server with internet access, NOT on the
# login node (it kills long-running user processes) and not as an sbatch job
# (compute nodes may lack internet). Same rule as the previous project's
# 00_setup_env.sh. From the project root:
#     bash slurm/00_setup_env.sh
#
# Safe to re-run: every step is skipped if already done.
#
# What it does:
#   1. Miniconda (if missing) and a per-project env "$ENV_NAME" (python 3.10).
#   2. PyTorch 2.5.1. PhysCtrl's README pins torch 2.5.1 on cu118; we use the cu121
#      wheels of the same version, which run on both A100 (CUDA 12.1 module) and
#      H200 (CUDA 12.6 module) nodes since the wheel bundles its own runtime.
#      torch-cluster and xformers are pinned to builds for torch 2.5.1 + cu121.
#   3. Clones PhysCtrl into $PHYSCTRL_ROOT and installs its requirements.txt
#      (warp-lang 1.5.0, taichi 1.7.3, diffusers 0.33.0, ...), minus genesis-world,
#      whose pins conflict with the rest (see step 3 below).
#   4. Downloads ONLY physctrl_base and physctrl_large checkpoints (~2 GB) plus the
#      released test set and sample set. The 5B video model (cogshader5B) is needed
#      only for rendering videos (Phase 6): set DOWNLOAD_RENDER=1 to fetch it (into $BIG_ROOT,
#      the hbku-research share, not the home) and
#      to build diff-gaussian-rasterization (needs nvcc).
#   5. Runs the CPU self-test and the synthetic dry run of Phases 0-2.
set -euo pipefail

if [ ! -f slurm/cluster.env ]; then
    echo "ERROR: run from the project root (the 4/ folder)." >&2; exit 1
fi
source slurm/cluster.env
for v in PROJECT_ROOT DATA_ROOT; do
    if [[ "${!v}" == *"<"* ]]; then echo "ERROR: set $v in slurm/cluster.env first." >&2; exit 1; fi
done
DOWNLOAD_RENDER="${DOWNLOAD_RENDER:-0}"
PHYSCTRL_REPO="https://github.com/cwchenwang/physctrl.git"
mkdir -p "$DATA_ROOT" "$WARP_CACHE_DIR" "$HF_HOME" logs results

# ---- 1. conda ---------------------------------------------------------------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "== installing Miniconda to $CONDA_DIR =="
    curl -fsSL -o "$HOME/miniconda_installer.sh" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$HOME/miniconda_installer.sh" -b -p "$CONDA_DIR"
    rm -f "$HOME/miniconda_installer.sh"
fi
# conda's shell functions and activate hooks may read unset variables; suspend -u.
set +u
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
set -u
pip install --upgrade pip

# ---- 2. torch stack ------------------------------------------------------------
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.5.1') else 1)" 2>/dev/null; then
    pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cu121
fi
pip install "xformers==0.0.28.post3" --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html --no-build-isolation

# ---- 3. PhysCtrl -----------------------------------------------------------------
if [ ! -d "$PHYSCTRL_ROOT/.git" ]; then
    git clone "$PHYSCTRL_REPO" "$PHYSCTRL_ROOT"
fi
echo "PhysCtrl commit: $(git -C "$PHYSCTRL_ROOT" rev-parse HEAD)" | tee "$DATA_ROOT/physctrl_commit.txt"
# PhysCtrl's requirements.txt does not resolve as written: genesis-world 0.2.1 pins
# pydantic==2.7.1, numpy==1.26.4 and moviepy>=2, against their pydantic==2.8.2,
# numpy==1.26.1 and moviepy==1.0.3 (which their renderer, libs/das, needs). Genesis is
# only used by generate_rigid_data.py (rigid track, Phase 3/5b), so it is left out here
# and gets its own env when that phase starts. A constraints file stops any requirement
# from upgrading torch (unconstrained, pip pulls the newest torch).
REQ="$DATA_ROOT/physctrl_requirements_somw.txt"
CONS="$DATA_ROOT/torch_constraints.txt"
grep -v '^genesis-world' "$PHYSCTRL_ROOT/requirements.txt" > "$REQ"
printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n' > "$CONS"
pip install -r "$REQ" -c "$CONS" --extra-index-url https://download.pytorch.org/whl/cu121
# Oct 15, 2026: kiui 0.3.5 ships an empty kiui/typing.py and setuptools >= 81 drops pkg_resources;
# both break PhysCtrl's video pipeline (inference.py). Pin working versions.
pip install "kiui==0.3.4" "setuptools==75.8.0"
python -c "import torch; assert torch.__version__.startswith('2.5.1'), torch.__version__"
python -c "import torch, numpy; assert '/.local/' not in torch.__file__ + numpy.__file__, 'packages load from ~/.local: is PYTHONNOUSERSITE=1 set?'"
if [ "$DOWNLOAD_RENDER" = "1" ]; then
    module load "$A100_CUDA_MODULE" 2>/dev/null || true
    pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git --no-build-isolation
fi

# ---- 4. checkpoints and data -------------------------------------------------------
pip install -U "huggingface_hub[cli]<1.0"
CKPT="$DATA_ROOT/checkpoints"
if [ ! -f "$CKPT/checkpoints/physctrl_large.safetensors" ]; then
    huggingface-cli download chenwang/physctrl \
        --include "checkpoints/physctrl_base.safetensors" "checkpoints/physctrl_large.safetensors" \
        --local-dir "$CKPT" --local-dir-use-symlinks False
fi
if [ "$DOWNLOAD_RENDER" = "1" ]; then
    # Oct 14, 2026 (user's rule): the video model is large, so it goes to the hbku-research share
    # ($BIG_ROOT), not the shared home. Files land in $RENDER_CKPT_DIR (cogshader5B/, lgm_fp16, SAM);
    # PhysCtrl takes these paths as arguments (--das_ckpt_path, --lgm_ckpt_path, --sam_ckpt_path).
    mkdir -p "$BIG_ROOT/checkpoints"
    huggingface-cli download chenwang/physctrl --include "checkpoints/cogshader5B/*" "checkpoints/lgm_fp16.safetensors" \
        --local-dir "$BIG_ROOT/checkpoints" --local-dir-use-symlinks False
    wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O "$RENDER_CKPT_DIR/sam_vit_h_4b8939.pth" || true
    echo "== render checkpoints in $RENDER_CKPT_DIR =="; ls -la "$RENDER_CKPT_DIR"
fi
for z in testset sample; do
    if [ ! -d "$DATA_ROOT/$z" ] || [ -z "$(ls -A "$DATA_ROOT/$z" 2>/dev/null)" ]; then
        huggingface-cli download chenwang/physctrl "$z.zip" --repo-type dataset \
            --local-dir "$DATA_ROOT/zips" --local-dir-use-symlinks False
        mkdir -p "$DATA_ROOT/$z"
        unzip -q -o "$DATA_ROOT/zips/$z.zip" -d "$DATA_ROOT/$z"
    fi
done
echo "== test set .h5 files: $(find "$DATA_ROOT/testset" -name '*.h5' | wc -l) =="
echo "== checkpoints:"; ls -la "$CKPT/checkpoints" || true

# ---- 5. checks ------------------------------------------------------------------------
python -c "import torch, warp, taichi, h5py, omegaconf, diffusers, torch_cluster, scipy; print('imports ok; cuda available here:', torch.cuda.is_available())"
python code/selftest.py
SOMW_RESULTS="$DATA_ROOT/dry_run_results" bash slurm/dry_run_all.sh

echo
echo "== setup done. Next, on the login node: bash slurm/preflight.sh, then bash slurm/submit_phase0_1.sh =="
