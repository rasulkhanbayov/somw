#!/bin/bash -l
#SBATCH -J somw_114_p12_test
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/114_p12_test_cpu_%A_%a.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/114_p12_test_cpu_%A_%a.err
#SBATCH -p cpu-all
#SBATCH -c 32
#SBATCH --mem=48000MB
#SBATCH -t 1-12:00:00
#SBATCH --array=0-3

# PHASE 12 test runs (pre-registered Oct 22, 2026; PLAN.md "Phase 12"): PolyFEM on the 500 primary Phase 9 single-object
# queries, 4 shards. Runs only if results/p12_s0_v2/s0.json says S0 PASS (it checks itself; submit after job 102).
# CPU only (cpu-all); workers = cores / 8. Verdict afterwards on a dev server: python code/p12_summary.py
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/export/home/rakh90100/cvpr_physics_plausibility/4}"
set +u; source slurm/cluster.env; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate somw-fem; set -u
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NCPU=$(nproc); WORKERS=$(( NCPU / 8 > 0 ? NCPU / 8 : 1 ))
echo "host       : $(hostname)  cpus: $NCPU (workers $WORKERS x 8 threads)  shard $SLURM_ARRAY_TASK_ID/4"
echo "big_root   : $BIG_ROOT ($( [ -d "$BIG_ROOT" ] && echo ok || echo MISSING ))"
echo "start      : $(date -Is)"
python code/p12_test.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4 --workers $WORKERS --threads 8
echo "end        : $(date -Is) (exit $?)"
