#!/bin/bash
# Sourced by every job script right after its #SBATCH header:
#   source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" <A100|H200|CPU>
#
# Why SLURM_SUBMIT_DIR and not $(dirname "$0"): inside an sbatch job, $0 is Slurm's
# spooled copy of the script (e.g. /var/spool/slurmd/job123/slurm_script), so paths
# relative to the script break. All jobs must therefore be submitted from the project
# root (the "4/" folder): cd "$PROJECT_ROOT" && sbatch slurm/NN_....sh
set -euo pipefail

_SUBMIT="${SLURM_SUBMIT_DIR:-$PWD}"
# Submitted from slurm/ itself (Oct 12, 2026: jobs 420738/420739 died silently that way):
# accept it and use the project root.
if [ ! -f "$_SUBMIT/slurm/cluster.env" ] && [ -f "$_SUBMIT/cluster.env" ] && [ "$(basename "$_SUBMIT")" = "slurm" ]; then
    _SUBMIT="$(cd "$_SUBMIT/.." && pwd)"
fi
if [ ! -f "$_SUBMIT/slurm/cluster.env" ]; then
    echo "ERROR: $_SUBMIT/slurm/cluster.env not found. Submit from the project root (the 4/ folder)." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$_SUBMIT/slurm/cluster.env"

for v in PROJECT_ROOT DATA_ROOT; do
    if [[ "${!v}" == *"<"* ]]; then
        echo "ERROR: $v is still a placeholder in slurm/cluster.env (${!v}). Edit it first." >&2
        exit 1
    fi
done
if [ "$(cd "$_SUBMIT" && pwd -P)" != "$(cd "$PROJECT_ROOT" && pwd -P)" ]; then
    echo "ERROR: submitted from $_SUBMIT but PROJECT_ROOT is $PROJECT_ROOT." >&2
    exit 1
fi

HW="${1:-CPU}"
# conda's activate hooks may read unset variables; suspend -u around them.
set +u
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
if [ "$HW" = "A100" ]; then
    module load "$A100_CUDA_MODULE" 2>/dev/null || true
elif [ "$HW" = "H200" ]; then
    module load "$H200_CUDA_MODULE" 2>/dev/null || true
fi

cd "$PROJECT_ROOT"
mkdir -p logs "$SOMW_RESULTS" "$WARP_CACHE_DIR"

echo "=============================================================="
echo "job        : ${SLURM_JOB_NAME:-interactive} id=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "host       : $(hostname)"
echo "start      : $(date -Is)"
echo "hardware   : $HW"
nvidia-smi -L 2>/dev/null || echo "gpus       : none visible"
# Is the big-artefact share mounted on this node? (logged so every job shows it)
if [ -n "${BIG_ROOT:-}" ]; then
    if [ -d "$BIG_ROOT" ] && [ -w "$BIG_ROOT" ]; then echo "big_root   : $BIG_ROOT (ok)"
    else echo "big_root   : $BIG_ROOT NOT ACCESSIBLE on $(hostname)"; fi
fi
echo "python     : $(which python) ($(python --version 2>&1))"
echo "project    : $PROJECT_ROOT @ $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
echo "physctrl   : $PHYSCTRL_ROOT @ $(git -C "$PHYSCTRL_ROOT" rev-parse HEAD 2>/dev/null || echo 'missing')"
echo "=============================================================="
# GPU-utilization log (Oct 15, 2026; Panther guidelines: GPUs must be used actively, low
# utilization can get jobs killed). Samples every 30 s to logs/gpu_<job>_<task>.csv; the exit
# line prints the mean utilization and peak memory so every log shows it.
_GPU_LOG=""
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    _GPU_LOG="$PROJECT_ROOT/logs/gpu_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.csv"
    # only the job's own GPU(s): on a node without device cgroups nvidia-smi lists every GPU
    nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i "$CUDA_VISIBLE_DEVICES"} --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits -l 30 > "$_GPU_LOG" 2>/dev/null &
    _GPU_MON=$!
fi
_gpu_summary() {
    [ -n "$_GPU_LOG" ] || return 0
    kill "$_GPU_MON" 2>/dev/null || true
    awk -F', *' 'NF>=4 {n++; u+=$2; if ($3>m) m=$3; t=$4} END {if (n) printf "gpu usage  : mean util %.0f%% over %d samples, peak memory %d / %d MiB\n", u/n, n, m, t}' "$_GPU_LOG"
}
# Capture $? first: inside the echo, $(date) would reset it to 0 (every Oct 10 log said "exit 0").
trap 'rc=$?; _gpu_summary; echo "end        : $(date -Is) (exit $rc)"' EXIT

# Print the exact config a job uses into its log.
log_config() {
    echo "----- config: $1 -----"
    cat "$1"
    echo "----------------------"
}

# Run a Python experiment, restarting it with --resume whenever it exits 75
# (EXIT_CUDA_LOST in code/common.py): a simulator blow-up corrupted the CUDA context,
# the script recorded the crashed unit, and a fresh process continues after it.
run_resumable() {
    local rc=0 n=0 max="${MAX_CUDA_RESTARTS:-50}"
    "$@" && rc=0 || rc=$?
    while [ "$rc" -eq 75 ] && [ "$n" -lt "$max" ]; do
        n=$((n + 1))
        echo "== CUDA context lost (exit 75): restart $n/$max with --resume =="
        "$@" --resume && rc=0 || rc=$?
    done
    [ "$n" -gt 0 ] && echo "== $n CUDA-loss restart(s); crashed units are recorded in the results =="
    return "$rc"
}

# Run one Python step as TWO shards side by side on this job's GPU (Oct 15, 2026: one MPM process
# used ~0.7 GB and 50-70% of an H200; two reached 95-97%). Array task T runs shards 2T and 2T+1 of 8.
# Usage: run_two_shards python code/x.py --config configs/y.yaml
run_two_shards() {
    local T="${SLURM_ARRAY_TASK_ID:-0}" p1 p2 rc=0
    run_resumable "$@" --shard $((2 * T)) --num-shards 8 &
    p1=$!
    run_resumable "$@" --shard $((2 * T + 1)) --num-shards 8 &
    p2=$!
    wait "$p1" || rc=$?
    wait "$p2" || rc=$?
    echo "== shards $((2 * T)), $((2 * T + 1)) of 8: exit $rc =="
    return "$rc"
}
