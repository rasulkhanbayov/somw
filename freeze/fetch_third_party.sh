#!/bin/bash
# Restore every third-party file the results depend on, at the exact revisions used (freeze/third_party/sources.tsv),
# then check each file against the sha256 recorded at freeze time (manifests/tp_*.tsv from the results repo).
# Run on a machine with internet (a dev server). Needs: git, unzip, wget, huggingface-cli (huggingface_hub < 1.0, as in
# the somw env); FLUX.1-schnell is gated on Hugging Face, so log in first (hf auth login) and accept its licence.
# Paths come from 4/slurm/cluster.env (edit it first if you restore to a new machine).
set -euo pipefail
cd "$(dirname "$0")/.."
source 4/slurm/cluster.env
MAN="${1:-freeze/manifests}"      # folder with tp_*.tsv (in this repo after make_manifest.py, or downloaded from the results repo)
PHYS_MODEL_REV=4b3cf613dd15f56dec2e7bee6012fb2bf18de654
PHYS_DATA_REV=bd8d316bed9edaafd8df3d4ecea493566d13519b

# PhysCtrl code
if [ ! -d "$DATA_ROOT/physctrl/.git" ]; then git clone https://github.com/cwchenwang/physctrl.git "$DATA_ROOT/physctrl"; fi
git -C "$DATA_ROOT/physctrl" checkout eaebd534719ce209436a610ea583ee7aee38943b

# PhysCtrl weights (same layout as 00_setup_env.sh)
huggingface-cli download chenwang/physctrl --revision $PHYS_MODEL_REV \
    --include "checkpoints/physctrl_base.safetensors" "checkpoints/physctrl_large.safetensors" \
    --local-dir "$DATA_ROOT/checkpoints" --local-dir-use-symlinks False
# PhysCtrl test and sample sets
for z in testset sample; do
    huggingface-cli download chenwang/physctrl "$z.zip" --repo-type dataset --revision $PHYS_DATA_REV \
        --local-dir "$DATA_ROOT/zips" --local-dir-use-symlinks False
    mkdir -p "$DATA_ROOT/$z" && unzip -q -o "$DATA_ROOT/zips/$z.zip" -d "$DATA_ROOT/$z"
done
# Renderer weights
mkdir -p "$BIG_ROOT/checkpoints"
huggingface-cli download chenwang/physctrl --revision $PHYS_MODEL_REV \
    --include "checkpoints/cogshader5B/*" "checkpoints/lgm_fp16.safetensors" \
    --local-dir "$BIG_ROOT/checkpoints" --local-dir-use-symlinks False
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O "$BIG_ROOT/checkpoints/checkpoints/sam_vit_h_4b8939.pth"
# Hugging Face models into the HF cache layout the code expects (HF_HOME=$BIG_ROOT/hf_cache)
export HF_HOME="$BIG_ROOT/hf_cache"
huggingface-cli download chenguolin/sv3d-diffusers --revision cf4c88acda116901e1e21aa9afa61976a21363f8
huggingface-cli download black-forest-labs/FLUX.1-schnell --revision 741f7c3ce8b383c54771c7003378a50191e9efe9
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --revision eed13092ef92e448dd6875b2a00151bd3f7db0ac

# Check every file against the freeze-time manifest (missing or different files are listed; exit 1 if any)
python3 freeze/check_files.py --manifests "$MAN" --areas tp_physctrl_code,tp_physctrl_ckpt,tp_physctrl_testset,tp_physctrl_sample,tp_physctrl_zips,tp_render_ckpt,tp_hf_models
