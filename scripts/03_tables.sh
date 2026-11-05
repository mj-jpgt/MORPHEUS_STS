#!/usr/bin/env bash
# Stage 3: Table 1 / 2 (metrics_v05), calibration, FDR, E1/E2, budget sweep, few-shot, transfer, stability, seconds,
# B4-B7 and the operator leave-out ablation. Sources: docstrings of the dbench/benchmark_v05 modules.
# Outputs: runs/benchmark_v05/*.{md,json}
set -euo pipefail
source "$(dirname "$0")/_env.sh"
M=dbench.benchmark_v05
# Table 1 pooled and per study (RFF labels, 95%, baseline B_known)
$PY -m $M.metrics_v05 table --studies "$STUDIES_CSV" --methods "$ALL_METHODS" --level 95 --baseline known --out table1_pooled_3studies
for S in $STUDIES; do
  $PY -m $M.metrics_v05 table --studies "$S" --methods "$ALL_METHODS" --level 95 --baseline known --out "table1_$S"
done
$PY -m $M.metrics_v05 overlap --studies "$STUDIES_CSV" --ref sel_screen --methods nid,linear_screen,ebm
# Method characterisation
for S in $STUDIES; do
  $PY -m $M.calib_v05 score "$S"
  $PY -m $M.fdr_v05 score "$S"
  $PY -m $M.e12_v05 score "$S"
  $PY -m $M.budget_sweep_v05 run "$S"
  $PY -m $M.fewshot_v05 "$S"
done
$PY -m $M.calib_v05 table "$STUDIES_CSV"
$PY -m $M.fdr_v05 table "$STUDIES_CSV"
$PY -m $M.e12_v05 tables "$STUDIES_CSV"
$PY -m $M.e1_matched_v05 "$STUDIES_CSV"
$PY -m $M.budget_sweep_v05 table "$STUDIES_CSV"
$PY -m $M.fewshot_table "$STUDIES_CSV"
$PY -m $M.stability_v05
$PY -m $M.seconds_table "$STUDIES_CSV"
# B2 / MC1 transfer without refit: per external cohort and pooled
TM="$OURS,ebm,gbm_shap,gbm_h,misty,rulefit,irf,knockoffs"
TS=""
for Q in $QUERIES; do
  T="${SUPPORT}__to__${Q}"; TS="${TS:+$TS,}$T"
  $PY -m $M.metrics_v05 table --studies "$T" --methods "$TM" --level 95 --baseline known --out "b2_$Q"
done
$PY -m $M.metrics_v05 table --studies "$TS" --methods "$TM" --level 95 --baseline known --out b2_pooled_3external
# B4/B5 (receipts), B6 (Zenodo TLS annotation recovery), B7/MC6 (H&E student layer), operator leave-out ablation
$PY -m $M.b4b5
$PY -m $M.b6_zenodo
$PY -m $M.b7_summary
$PY -m $M.ablation_ops
