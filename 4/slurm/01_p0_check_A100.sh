#!/bin/bash -l
#SBATCH -J somw_p0_check_A100
#SBATCH -o /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p0_check_A100_%j.out
#SBATCH -e /export/home/rakh90100/cvpr_physics_plausibility/4/logs/p0_check_A100_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40000MB
#SBATCH -A a100_hbku
#SBATCH -q a100_hbku_qos
#SBATCH -t 02:00:00

# PHASE 0 on A100. Run first: every later phase reads the test-set manifest it writes.
#   1. CPU self-test (math and guards).
#   2. code/p0_check.py: imports, checkpoints, test-set manifest, replay determinism
#      (re-simulate 8 stored elastic trajectories in all three timing modes), generator
#      sanity (base and large load with every key matched; error vs ground truth).
#   3. PhysCtrl's own eval.py on up to 20 elastic test items, an independent cross-check
#      of our bridge: compare its printed MSE with mse_model_coords in p0_report.json.
# STOP if p0_check exits non-zero: either replay failed (probes would measure simulator
# mismatch, not model error) or a checkpoint loaded with missing keys.
# Output: results/p0_check_A100/{p0_report.json, testset_manifest.json, physctrl_eval/}
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" A100
log_config configs/p0_check.yaml

python code/selftest.py
python code/p0_check.py --config configs/p0_check.yaml --tag A100

for which in base large; do
    cfgf="$SOMW_RESULTS/p0_check_A100/physctrl_eval/eval_${which}_filled.yaml"
    echo "== PhysCtrl eval.py ($which) =="
    (cd "$PHYSCTRL_ROOT/src" && python eval.py --config "$cfgf") \
        2>&1 | tee "$SOMW_RESULTS/p0_check_A100/physctrl_eval/eval_${which}.log" | tail -5
done
echo "== P0 (A100) done. Read results/p0_check_A100/p0_report.json before submitting Phase 1. =="
