#!/bin/bash
# End-to-end dry run of Phases 0-2 on the synthetic CPU backend (~10 min on a laptop CPU).
# Exercises every script, config and summarizer without a GPU or PhysCtrl.
# Outputs go to $SOMW_RESULTS/SYNTHETIC_* (set SOMW_RESULTS to keep them out of results/).
#     bash slurm/dry_run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export SOMW_RESULTS="${SOMW_RESULTS:-$PWD/results}"
py=python; python -c "pass" 2>/dev/null || py=python3

$py code/selftest.py
$py code/p0_check.py      --config configs/p0_check.yaml   --dry-run --tag A100
$py code/p1_cost.py       --config configs/p1_cost.yaml    --dry-run --tag A100
$py code/p1_cost.py       --config configs/p1_cost.yaml    --dry-run --tag H200
for i in 0 1; do
  $py code/p1_fidelity.py --config configs/p1_fidelity.yaml --dry-run --shard $i --num-shards 2
done
$py code/p1_summarize.py  --config configs/p1_summary.yaml --allow-synthetic
$py code/p2_build_ood.py  --config configs/p2_ood.yaml     --dry-run
for i in 0 1; do
  $py code/p2a_lift_floor.py --config configs/p2a_lift_floor.yaml --dry-run --shard $i --num-shards 2
done
$py code/p2b_generate.py  --config configs/p2b.yaml        --dry-run
for i in 0 1; do
  $py code/p2b_probe.py   --config configs/p2b.yaml        --dry-run --shard $i --num-shards 2
done
$py code/p2_summarize.py  --config configs/p2_summary.yaml --allow-synthetic
echo "== dry run complete: see $SOMW_RESULTS/SYNTHETIC_p1_summary and SYNTHETIC_p2_summary =="
