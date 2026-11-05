#!/usr/bin/env bash
# Blind-discovery sweep for the hest_multi protocols (renal_spatial_v07, ependymoma_spatial_v01 and the drafts
# written by dbench/downstream/scripts/make_blind_drafts.py). Agent R, 2026-10-21.
#
#   CAMPAIGN_PROTOCOL=dbench/protocols/renal_spatial_v07.json bash dbench/scripts/blind_sweep.sh RUN_DIR WORKERS [STAGE]
#   STAGE: sweep (default) | crossfold | aggregate | all          DRY=1 prints the commands and writes nothing.
#
# Refuses to run (DRY=0) unless the protocol status starts with REGISTERED, the checkpoint fields are filled and
# match the file, RUN_DIR/campaign_manifest.json exists (python -m dbench.prospective_campaign --protocol P
# --run-dir RUN_DIR) and its protocol_sha256 equals the protocol's canonical hash. Cells: every (family, target)
# of discovery.relationship_families x every fold; receipts RUN_DIR/receipts/<family>/f<fold>_<safe target>.json
# ('/' and ' ' -> '_'; the receipt carries the exact target); a cell with a receipt is skipped (resumable). The
# runner reads proposals_per_target, null_suite.draws, min_gain, relation and proposers_in_all from the protocol.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
PROTO=${CAMPAIGN_PROTOCOL:?set CAMPAIGN_PROTOCOL}
RUN=${1:?RUN_DIR}; WORKERS=${2:-6}; STAGE=${3:-sweep}; DRY=${DRY:-0}
PY=${PY:-$HOME/morpheus_venv/bin/python}
[ "$DRY" = 1 ] && WORKERS=1   # dry output must not interleave
T=${OMP_NUM_THREADS:-1}
export PYTHONPATH=$REPO:$HOME/q_pylib OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T HDF5_USE_FILE_LOCKING=FALSE
HELPER=$REPO/dbench/scripts/blind_sweep_cells.py

read -r STATUS CKPT CKSHA STUDY NFOLDS SEED0 PHASH < <($PY "$HELPER" header "$PROTO")
if [ "$DRY" != 1 ]; then
  [ "$STATUS" = REGISTERED ] || { [ "${DEV_PROTOCOL:-0}" = 1 ] && [ "$STATUS" = DEVELOPMENT ]; } || { echo "protocol status is $STATUS, not REGISTERED (DEV_PROTOCOL=1 admits DEVELOPMENT)" >&2; exit 1; }
  case "$CKPT" in to_be_filled*) echo "checkpoint not filled in $PROTO" >&2; exit 1;; esac
  [ "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$CKSHA" ] || { echo "checkpoint sha256 mismatch" >&2; exit 1; }
  [ -f "$RUN/campaign_manifest.json" ] || { echo "not initialised: python -m dbench.prospective_campaign --protocol $PROTO --run-dir $RUN" >&2; exit 1; }
  REG=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol_sha256"])' "$RUN/campaign_manifest.json")
  [ "$REG" = "$PHASH" ] || { echo "protocol differs from the registered campaign" >&2; exit 1; }
fi

cell() {   # one argument, fields separated by 0x1f (unit separator): family, fold, target, [score_frozen file], output
  local fam f t extra out
  IFS=$'\x1f' read -r fam f t extra out <<< "$1"
  [ -s "$out" ] && { echo "SKIP $fam f$f $t"; return; }
  local cmd=("$PY" -m morpheus_encoder.scripts.run_general_discovery_packet --episode hest_multi --study "$STUDY"
             --protocol "$PROTO" --family "$fam" --fold "$f" --n-folds "$NFOLDS" --target "$t" --arm source
             --proposer "${PROPOSER:-all}" --seed $((SEED0 + f)) --device cuda --learned-checkpoint "$CKPT" --output "$out")
  [ -n "$extra" ] && cmd+=(--score-frozen "$extra")
  if [ "$DRY" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; return; fi
  mkdir -p "$(dirname "$out")" "$RUN/logs"
  sleep $((RANDOM % 30))
  # MIN_FREE_GB (default 0): wait until that much RAM is available before starting a cell (shared box)
  # MIN_FREE_GPU_MB (default 0): wait until that much GPU memory is free before starting a cell (shared card)
  while [ "${MIN_FREE_GPU_MB:-0}" -gt 0 ] && [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "${MIN_FREE_GPU_MB:-0}" ]; do sleep $((30 + RANDOM % 30)); done
  while [ "${MIN_FREE_GB:-0}" -gt 0 ] && [ "$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)" -lt "${MIN_FREE_GB:-0}" ]; do sleep $((30 + RANDOM % 30)); done
  local log="$RUN/logs/$(basename "$out" .json)_${fam}${extra:+_crossfold}.log"
  echo "START $fam f$f $t $(date -u +%FT%TZ)"
  "${cmd[@]}" > "$log" 2>&1 && echo "DONE $fam f$f $t $(date -u +%FT%TZ)" || echo "FAIL $fam f$f $t exit=$? $(date -u +%FT%TZ)"
}
export -f cell; export PY PROTO RUN STUDY NFOLDS SEED0 CKPT DRY PROPOSER MIN_FREE_GB MIN_FREE_GPU_MB

if [ "$STAGE" = sweep ] || [ "$STAGE" = all ]; then
  $PY "$HELPER" cells "$PROTO" "$RUN" | xargs -d '\n' -P "$WORKERS" -n 1 bash -c 'cell "$0"'
fi
if [ "$STAGE" = crossfold ] || [ "$STAGE" = all ]; then
  if [ "$DRY" = 1 ]; then
    echo "$PY -m dbench.prospective_campaign --protocol $PROTO --run-dir $RUN --crossfold-requests > $RUN/crossfold_requests.json"
  else
    $PY -m dbench.prospective_campaign --protocol "$PROTO" --run-dir "$RUN" --crossfold-requests > "$RUN/crossfold_requests.json"
    # Materialize the post-filter receipt matrix before launching.  A
    # cross-fold request can be split into parts and can exclude variables
    # absent from a scoring fold's support-HVG design.  The exact resulting
    # count is the only valid completion criterion for aggregation/watchers.
    CF_CELLS="$RUN/crossfold_requests/cells.usv"
    $PY "$HELPER" crossfold "$RUN/crossfold_requests.json" "$RUN" > "$CF_CELLS"
    wc -l < "$CF_CELLS" | tr -d '[:space:]' > "$RUN/crossfold_requests/expected_receipts.txt"
    [ "$(cat "$RUN/crossfold_requests/expected_receipts.txt")" -gt 0 ] || {
      echo "cross-fold request matrix is empty" >&2; exit 2;
    }
    xargs -d '\n' -P "$WORKERS" -n 1 bash -c 'cell "$0"' < "$CF_CELLS"
  fi
fi
if [ "$STAGE" = aggregate ] || [ "$STAGE" = all ]; then
  if [ "$DRY" = 1 ]; then echo "$PY -m dbench.prospective_campaign --protocol $PROTO --run-dir $RUN --aggregate"
  else $PY -m dbench.prospective_campaign --protocol "$PROTO" --run-dir "$RUN" --aggregate > "$RUN/aggregate.out"; fi
fi
echo "BLIND_${STAGE}_DONE $(date -u +%FT%TZ)"
