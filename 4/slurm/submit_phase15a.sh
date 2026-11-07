#!/bin/bash
# Phase 15a (pre-registered Oct 22, 2026; PLAN.md "Phase 15"): joint rollouts -> training -> evaluation, chained.
# Needs results/p15a_scenes (Claude ran the scene design on a dev server). Verdict afterwards on a dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f results/p15a_scenes/scenes.jsonl ] || { echo "results/p15a_scenes missing"; exit 1; }
a=$(sbatch --parsable slurm/111_p15a_build_A100.sh)
b=$(sbatch --parsable --dependency=afterok:$a slurm/112_p15a_train_H200.sh)
c=$(sbatch --parsable --dependency=afterok:$b slurm/113_p15a_eval_H200.sh)
echo "15a submitted: build $a -> train $b -> eval $c"
