#!/usr/bin/env bash
# Reproduce everything. Total about 50-60 min on a 2-core laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. tests (~10 s)"
python -m pytest -q tests/

echo "== 2. main policy (~17 min). Skip this step to use the shipped results/policy.npz"
if [ "${SKIP_MAIN_TRAINING:-0}" != "1" ]; then
  python train.py
fi

echo "== 3. rejected reward variant: +50 per accepted request (~17 min)"
python train.py --accept-bonus 50 --out results/policy_bonus.npz

echo "== 4. evaluation: 200 test days, fixed instance, sensitivity, auditor (~8-10 min)"
python evaluate.py

echo "== 5. analysis scripts (~3 min)"
python experiments/jit_rejection_causes.py
python experiments/failure_analysis.py

echo "== 6. optional: rollout teacher check (~2-3 min)"
python experiments/rollout_teacher_check.py --days 10 --m 4

echo "Done. Main tables: results/tables.md. Plots: results/*.png"
