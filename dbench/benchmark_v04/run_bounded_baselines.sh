#!/usr/bin/env bash
# Bounded, restartable scheduler for the v04 common-pool benchmark.
#
# One worker owns one (method, target) job.  Existing .npz receipts are
# skipped by score_methods.py, so an interrupted run is safe to resume.
# A method which cannot score a target inside its declared budget becomes a
# recorded timeout rather than blocking every other method behind it.
set -euo pipefail

cd ~/morpheus_work_bench
OUT=runs/development/benchmark_v04
LOG="$OUT/logs/bounded_core_$(date -u +%Y%m%dT%H%M%SZ).log"
TIMEOUT_SEC=${TIMEOUT_SEC:-1200}
CPU_WORKERS=${CPU_WORKERS:-3}
TARGETS=${TARGETS:-40}

mkdir -p "$OUT/logs" "$OUT/timeouts"

run_one() {
  local method="$1" idx="$2" device="$3" jobs="$4"
  local tag="${method}_t${idx}"
  if timeout --signal=TERM --kill-after=30s "${TIMEOUT_SEC}s" \
       ~/morpheus_venv/bin/python -m dbench.benchmark_v04.score_methods \
       gse237183_glioma --methods "$method" --slow-subset \
       --target-index "$idx" --jobs "$jobs" --device "$device" \
       >> "$LOG" 2>&1; then
    printf '%s completed\n' "$tag" >> "$OUT/timeouts/status.tsv"
  else
    rc=$?
    printf '%s timeout_or_failure exit=%s budget_sec=%s\n' "$tag" "$rc" "$TIMEOUT_SEC" \
      >> "$OUT/timeouts/status.tsv"
  fi
}
export -f run_one
export TIMEOUT_SEC LOG OUT

# CPU methods.  These are independently scheduled rather than chained behind
# RuleFit.  We deliberately use their standard declared wrappers; expensive
# "fair" permutation variants remain supplemental diagnostics.
for method in ebm gbm_h rulefit linear_screen stability_selection irf; do
  (
    seq 0 $((TARGETS - 1)) | xargs -r -n1 -P "$CPU_WORKERS" -I{} \
      bash -lc 'run_one "'$method'" "$1" cpu 4' _ {}
  ) &
done

# GPU methods run sequentially so TabDistill and NID never contend for the A100.
(
  for method in nid tabdistill; do
    for idx in $(seq 0 $((TARGETS - 1))); do
      run_one "$method" "$idx" cuda 1
    done
  done
) &
wait

printf 'finished %s\n' "$(date -u +%FT%TZ)" >> "$OUT/timeouts/status.tsv"
