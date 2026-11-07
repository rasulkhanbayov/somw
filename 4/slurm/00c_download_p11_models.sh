#!/bin/bash
# Phase 11 (Oct 20, 2026): download the image model and the VLM judge into $BIG_ROOT/hf_cache.
# Run on a dev server with internet (e.g. crimv3mgpu015 = 10.4.64.27), NOT on the login node.
#     bash slurm/00c_download_p11_models.sh        (from the project root)
# FLUX.1-schnell: Apache-2.0, about 34 GB. Qwen2-VL-7B-Instruct: Apache-2.0, about 16 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/cluster.env
source ~/miniconda3/etc/profile.d/conda.sh
set +u; conda activate somw; set -u
export PYTHONNOUSERSITE=1 HF_HOME="$BIG_ROOT/hf_cache"
# FLUX.1-schnell is gated: use the token from wherever `hf auth login` saved it (newest first; nothing is copied).
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
    for t in "$DATA_ROOT/hf_cache/token" "$HOME/.cache/huggingface/token"; do
        if [ -f "$t" ]; then export HF_TOKEN_PATH="$t"; break; fi
    done
fi
python -c "from huggingface_hub import whoami; print('logged in to Hugging Face as', whoami()['name'])" || {
    echo "ERROR: no valid Hugging Face login found. Run: hf auth login (see the instructions in this file's header)" >&2; exit 1; }
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("black-forest-labs/FLUX.1-schnell", "Qwen/Qwen2-VL-7B-Instruct"):
    p = snapshot_download(repo)
    print("downloaded", repo, "->", p, flush=True)
PY
echo "== 00c done: models in $HF_HOME =="
