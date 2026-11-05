#!/usr/bin/env bash
# Stage 4: translation v10, P5 conserved rules / cross-cancer transfer of the eight frozen kidney relations (G1-G8, no refit).
# s10_p5: gain over the nuisance baseline with the Moran-matched null; s11_p5_known: gain over B_known (addendum D20).
# Sources: docstrings of dbench/translation_v10/s10_p5.py and s11_p5_known.py. Needs the registered v08 run (C.v08_ids()).
set -euo pipefail
source "$(dirname "$0")/_env.sh"
T=dbench.translation_v10
RELS="${RELS:-G1,G2,G3,G4,G5,G6,G7,G8}"
for COH in valdeolivas2024_crc ependymoma_pfa gse294385 ji2020_cscc; do
  $PY -m $T.s10_p5 "$COH"
  $PY -m $T.s11_p5_known "$COH" "$RELS"
done
$PY -m $T.s10b_p5_matrix
$PY -m $T.s11b_addendum
