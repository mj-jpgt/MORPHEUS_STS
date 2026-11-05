#!/usr/bin/env bash
# Stage 1: benchmark v05 labels (label engine v05 = v04 arithmetic with factorised solves), then transfer labels (B2 / MC1).
# Sources: docstrings of dbench/benchmark_v05/labels_fast.py and transfer_v05.py; dbench/benchmark_v04/run_labels.sh is the v04 engine
# launcher (pilot, then --workers 4), kept for `labels_fast --verify`.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
export MORPHEUS_GPU_CHUNK_ELEMS="${MORPHEUS_GPU_CHUNK_ELEMS:-4e7}"
for S in $STUDIES; do
  $PY -m dbench.benchmark_v05.labels_fast "$S"
done
for Q in $QUERIES; do
  $PY -m dbench.benchmark_v05.transfer_v05 "$SUPPORT" "$Q" labels
done
