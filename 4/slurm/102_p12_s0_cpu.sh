#!/bin/bash -l
#SBATCH -J somw_102_p12_s0
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/102_p12_s0_cpu_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/102_p12_s0_cpu_%j.err
#SBATCH -p cpu-all
#SBATCH -c 32
#SBATCH --mem=32000MB
#SBATCH -t 1-00:00:00

# PHASE 12 gate S0 (pre-registered Oct 22, 2026; PLAN.md "Phase 12" + amendment): the independent simulator (PolyFEM,
# fixed-corotated) on DEVELOPMENT queries only (data/calib/id_new_0..2): time-step rule, free fall, small-strain
# agreement with the MPM, push check. CPU only (Panther rule: CPU work on cpu-all). 4 workers x 8 threads.
# No account/QOS line: add yours if cpu-all requires one. Verdict printed and written to results/p12_s0_v2/SUMMARY.md.
# v2 (Oct 24): run 1 (job 432388) failed S0c; the push is now a snapped-box force (PLAN "Phase 12" amendment). One rerun only.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/export/home/rakh90100/cvpr_physics_plausibility/4}"
set +u; source slurm/cluster.env; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate somw-fem; set -u
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
# job 431597 failed on an older node: the binary needs the env's libstdc++ (built with conda's toolchain, glibc 2.17 sysroot)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NCPU=$(nproc); WORKERS=$(( NCPU / 8 > 0 ? NCPU / 8 : 1 ))
echo "host       : $(hostname)  cpus: $NCPU (workers $WORKERS x 8 threads)  $(lscpu | grep 'Model name' | sed 's/  */ /g')"
echo "glibc      : $(ldd --version | head -1)"
echo "big_root   : $BIG_ROOT ($( [ -d "$BIG_ROOT" ] && echo ok || echo MISSING ))"
echo "polyfem    : $(ls -la $BIG_ROOT/tools/polyfem/build_conda/PolyFEM_bin | awk '{print $5, $6, $7, $8}')"
echo "start      : $(date -Is)"
python code/p12_s0.py --workers $WORKERS --threads 8
echo "end        : $(date -Is) (exit $?)"
