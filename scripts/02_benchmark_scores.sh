#!/usr/bin/env bash
# Stage 2: score the shared pools with ours and the baselines (budgeted, resumable; a budget overrun is recorded as a timeout row).
# Sources: docstrings of dbench/benchmark_v05/run_scores.py and transfer_v05.py; per-fit budgets are run_scores.BUDGET.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
for S in $STUDIES; do
  $PY -m dbench.benchmark_v05.run_scores "$S" --stage gpu --methods "$OURS,tabdistill"
  $PY -m dbench.benchmark_v05.run_scores "$S" --stage cpu --methods "$CPU_METHODS" --workers "$WORKERS"
  $PY -m dbench.benchmark_v05.run_scores "$S" --stage cpu --methods "$FAIR_METHODS" --workers "$WORKERS"
done
for Q in $QUERIES; do
  $PY -m dbench.benchmark_v05.transfer_v05 "$SUPPORT" "$Q" scores --stage gpu --methods "$OURS"
  $PY -m dbench.benchmark_v05.transfer_v05 "$SUPPORT" "$Q" scores --stage cpu --methods "ebm,gbm_shap,gbm_h,misty,rulefit,irf,knockoffs" --workers "$WORKERS"
done
