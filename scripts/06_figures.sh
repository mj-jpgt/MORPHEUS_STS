#!/usr/bin/env bash
# Stage 6: figures. The scripts are verbatim and carry absolute Windows paths (SCRATCH, SKILL) to their inputs; see README, "Figures".
# usage: SCRATCH=<directory holding cards/, int21/, g3crop/, crc/> bash scripts/06_figures.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"
F=figures
$PY $F/fig7_benchmark/plot_benchmark_v05.py
$PY $F/fig6_evidence_cards/heldout_int21.py
$PY $F/fig6_evidence_cards/plot_evidence_cards_v02.py --out "$F/fig6_evidence_cards"
$PY $F/fig6_evidence_cards/plot_evidence_cards_v02_k.py
G3=$F/fig6_evidence_cards/card_G3_71ed4499fb99
$PY $G3/closeup/plot_g3_closeup.py "${SCRATCH:?set SCRATCH}"
$PY $G3/crosscancer/plot_g3_crosscancer.py "${SCRATCH:?set SCRATCH}"
