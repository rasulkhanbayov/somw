#!/bin/bash
# Rebuild the Phase 12 PolyFEM binary exactly as on Oct 22, 2026 (commands recovered from the session record; the
# configure cache is in CMakeCache.build_conda.txt). Needs the conda env somw-fem (freeze/env/somw-fem.explicit.txt:
# conda create -n somw-fem --file freeze/env/somw-fem.explicit.txt), which provides gcc 15.3 with the glibc 2.17 sysroot,
# cmake and ninja. The binary needs glibc >= 2.14 and the env's libstdc++ at run time (LD_LIBRARY_PATH=$CONDA_PREFIX/lib).
# Check: sha256sum build_conda/PolyFEM_bin against PolyFEM_bin.sha256 (byte identity is not guaranteed across hosts;
# the Phase 12 smoke test, code/p12_s0.py --smoke, is the functional check).
set -euo pipefail
P=${CONDA_PREFIX:-$HOME/miniconda3/envs/somw-fem}
D=${1:-$BIG_ROOT/tools/polyfem}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -d "$D/.git" ] || git clone https://github.com/polyfem/polyfem.git "$D"
git -C "$D" checkout 008a1aeae97b277c8bcf644adf0673749b7889ef
mkdir -p "$D/shim" && cp "$HERE/roundeven.c" "$D/shim/"
(cd "$D/shim" && $P/bin/x86_64-conda-linux-gnu-gcc -O2 -c roundeven.c -o roundeven.o && $P/bin/x86_64-conda-linux-gnu-ar rcs libroundeven.a roundeven.o)
export PATH=$P/bin:$PATH
cd "$D"
cmake -S . -B build_conda -G Ninja -DCMAKE_BUILD_TYPE=Release -DPOLYFEM_WITH_TESTS=OFF \
  -DCMAKE_C_COMPILER=$P/bin/x86_64-conda-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$P/bin/x86_64-conda-linux-gnu-g++ \
  -DSIMD_CXX_FLAGS="-march=x86-64-v2" -DSIMD_C_FLAGS="-march=x86-64-v2" \
  -DCMAKE_INSTALL_RPATH="$P/lib" -DCMAKE_BUILD_RPATH="$P/lib" \
  -DCMAKE_EXE_LINKER_FLAGS="" -DCMAKE_CXX_STANDARD_LIBRARIES="$D/shim/libroundeven.a -lm"
ninja -C build_conda -j 16 PolyFEM_bin
ls -la build_conda/PolyFEM_bin
