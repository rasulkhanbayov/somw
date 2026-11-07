#!/bin/bash
# Run from the project root before each submit_phase*.sh, twice: on a development server
# (environment checks) and on the login node (Slurm checks only; Python is not allowed there).
#     bash slurm/preflight.sh
# Catches in seconds what would otherwise fail only after a job waits in the queue:
# leftover placeholders, missing env/checkpoints/test set, and account/QOS/partition
# combinations Slurm rejects. `sbatch --test-only` validates a job without submitting it.
# Exits non-zero if anything blocks Phase 0-2.
set -uo pipefail
cd "$(dirname "$0")/.."

fails=0
ok()   { echo "  [ OK ] $*"; }
bad()  { echo "  [FAIL] $*"; fails=$((fails + 1)); }
warn() { echo "  [WARN] $*"; }

echo "== config =="
source slurm/cluster.env
for v in PROJECT_ROOT DATA_ROOT; do
    [[ "${!v}" == *"<"* ]] && bad "$v is a placeholder" || ok "$v=${!v}"
done
[ "$(pwd -P)" = "$(cd "$PROJECT_ROOT" 2>/dev/null && pwd -P)" ] \
    && ok "running from PROJECT_ROOT" || bad "run from $PROJECT_ROOT (now in $(pwd))"
left=$(grep -l -E "<(A100|H200)_(ACCOUNT|QOS)>" slurm/[0-9][0-9]_*.sh 2>/dev/null)
[ -z "$left" ] && ok "Slurm headers filled" || bad "placeholders in: $left (run slurm/configure.sh)"
[ -d logs ] && ok "logs/ exists" || bad "logs/ missing (Slurm will not create it)"

echo "== environment (from 00_setup_env.sh) =="
# Panther guidelines (Oct 15, 2026): no conda activation or Python on the login node (a monitor
# kills such processes). Environment checks therefore run only on a development server
# (10.4.64.25-28, e.g. crimv3mgpu015 = 10.4.64.27); on the login node only the Slurm checks run.
if command -v sbatch >/dev/null; then
    warn "login node: conda/python checks skipped (not allowed here); run 'bash slurm/preflight.sh' on a dev server for them"
else
    if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
        set +u; source "$CONDA_DIR/etc/profile.d/conda.sh"
        if conda activate "$ENV_NAME" 2>/dev/null; then
            set -u; ok "conda env $ENV_NAME"
            python -c "import torch, warp, h5py, omegaconf, diffusers, torch_cluster, scipy, yaml" 2>/dev/null \
                && ok "python imports" || bad "python imports fail in $ENV_NAME (re-run 00_setup_env.sh)"
            python -c "import torch, numpy, sys; sys.exit('/.local/' in torch.__file__ + numpy.__file__)" 2>/dev/null \
                && ok "torch/numpy load from the env, not ~/.local" || bad "torch or numpy loads from ~/.local (PYTHONNOUSERSITE)"
            python code/selftest.py >/dev/null 2>&1 && ok "selftest" || bad "selftest fails: python code/selftest.py"
        else
            set -u; bad "conda env $ENV_NAME missing (run 00_setup_env.sh)"
        fi
    else
        bad "no conda at $CONDA_DIR"
    fi
fi
[ -d "$PHYSCTRL_ROOT/.git" ] && ok "PhysCtrl clone @ $(git -C "$PHYSCTRL_ROOT" rev-parse --short HEAD)" \
    || bad "PhysCtrl not cloned at $PHYSCTRL_ROOT"
for m in base large; do
    f="$DATA_ROOT/checkpoints/checkpoints/physctrl_${m}.safetensors"
    [ -s "$f" ] && ok "checkpoint $m ($(du -h "$f" | cut -f1))" || bad "missing $f"
done
n=$(find "$DATA_ROOT/testset" -name '*.h5' 2>/dev/null | wc -l)
[ "$n" -gt 0 ] && ok "test set: $n .h5 files" || bad "test set empty at $DATA_ROOT/testset"

echo "== Slurm (validated with --test-only; nothing is submitted) =="
if command -v sbatch >/dev/null; then
    for s in slurm/01_p0_check_A100.sh slurm/02_p0_check_H200.sh; do
        out=$(sbatch --test-only "$s" 2>&1)
        if echo "$out" | grep -qi "to start"; then ok "$(basename "$s"): $(echo "$out" | tail -1)"
        else bad "$(basename "$s"): $out"; fi
    done
else
    warn "sbatch not found on $(hostname): this is not the login node. Run preflight and submit_*.sh on panther-login."
fi

echo "== existing results (jobs append/overwrite here) =="
for d in results/p0_check_A100 results/p1_cost_A100 results/p1_fidelity_A100 results/p2b_probe; do
    [ -e "$d" ] && warn "$d already exists"
done

echo
if [ "$fails" -eq 0 ]; then echo "PREFLIGHT OK"; else echo "PREFLIGHT: $fails problem(s)"; exit 1; fi
