#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/morpheus_work_v06
export PYTHONPATH=/home/ubuntu/morpheus_work_v06${PYTHONPATH:+:$PYTHONPATH}
OUT=runs/confirmation/v08_external_batched_residual_internal_v01
mkdir -p "$OUT"
exec /home/ubuntu/morpheus_venv/bin/python dbench/scripts/run_v08_external_batches_v01.py \
  --run-dir /home/ubuntu/morpheus_work_v08/runs/registered_v08/renal-spatial-v08-multiscale-20261026 \
  --out "$OUT" --families residual_programme_targets --batch-size 2 \
  --external-cellvit-root runs/confirmation/v08_external_cellvit_zenodo_v01 \
  > "$OUT/run.log" 2>&1
