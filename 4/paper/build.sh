#!/bin/bash
# Build the paper with tectonic (borrowed read-only from another env on the dev server). Run from paper/.
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-$HOME/miniconda3/envs/matpcr_tex/bin/tectonic}"
"$TECTONIC" -X compile main.tex --keep-logs
echo "built: $(pwd)/main.pdf"
