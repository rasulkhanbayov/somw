#!/bin/bash
# Everything that is ready on Oct 29, 2026 (each pre-registered in PLAN.md). Run on panther-login from the project root.
#   Phase 12  S0 gate rerun (cpu-all, ~6-10 h) -> test runs on 500 queries (cpu-all, 4 shards; start only if S0 PASS)
#   Phase 15a contact-aware generator pilot: joint rollouts (A100) -> training (H200) -> evaluation (H200), ~6-8 GPU-h
#   Phase 14a-v2 simulator-distillation retry: training (4 x H200) -> evaluation (H200), ~7 GPU-h
set -euo pipefail
cd "$(dirname "$0")/.."
a=$(sbatch --parsable slurm/102_p12_s0_cpu.sh)
b=$(sbatch --parsable --dependency=afterok:$a slurm/114_p12_test_cpu.sh)
echo "Phase 12: S0 $a -> test $b"
bash slurm/submit_phase15a.sh
bash slurm/submit_phase14a2.sh
