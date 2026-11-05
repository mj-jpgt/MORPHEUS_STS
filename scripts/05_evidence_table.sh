#!/usr/bin/env bash
# Stage 5: registered v08 funnel -> external family BH -> characterisation ladder -> cards -> relation grades -> evidence table.
# The registered discovery run itself is launched with (v08 version of blind_sweep.sh; the run directory is initialised first with
# `python -m dbench.prospective_campaign --protocol P --run-dir RUN_DIR`; it needs the frozen learned checkpoint, see README):
#   CAMPAIGN_PROTOCOL=dbench/protocols/renal_spatial_v08_multiscale.json bash dbench/scripts/blind_sweep.sh RUN_DIR WORKERS all
# Sources: dbench/scripts/run_v08_*.sh, docstrings of the downstream modules, PAPER_EVALUATION_CAMPAIGN_V01 (row D1).
set -euo pipefail
source "$(dirname "$0")/_env.sh"
D=dbench.downstream
# external confirmation in resumable batches, then the global aggregate and the family-wide BH (masked variant)
bash dbench/scripts/run_v08_compartment_external_batches_v01.sh
bash dbench/scripts/run_v08_residual_external_batches_v01.sh
bash dbench/scripts/run_v08_proposed_heter_external_batches_v01.sh
$PY dbench/scripts/aggregate_v08_external_family_bh_v01.py --out runs/confirmation/v08_external_family_bh_v01
# characterisation (Meylan 2022 ccRCC, relation_v03 engine) and evidence-card data
for ST in 1 2 3; do $PY -m $D.characterise_v10 --stage "$ST"; done
$PY -m $D.characterise_v10 --aggregate
$PY -m $D.cards_v10 1
$PY -m $D.decode_v10 1 2 3
# two blind graders combined (grader files are read from runs/development/novelty_v10/grades/)
for ST in 1 2 3; do $PY -m $D.novelty_v10_score "$ST"; done
$PY -m $D.scripts.relation_grades runs/development/relation_grades_v08b
# evidence table from receipts (hashes every input)
$PY dbench/downstream/evidence_table_v01.py --root "$PKG" --tx "$PKG" --v08 "$PKG" --out runs/confirmation/evidence_table_v01
