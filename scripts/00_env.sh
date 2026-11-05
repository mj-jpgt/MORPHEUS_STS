#!/usr/bin/env bash
# One-off: map the original box layout onto this package. The code addresses ~/morpheus_work_{bench,sw,v06,tx,v08}
# and /lambda/nfs/geeg/biorag3_persistent_20260711 (hest_multi.STORE) as literal paths; none of it was rewritten.
# Receipts (runs/...) are written below the package root through these links.
# usage: STORE_DIR=/path/to/downloaded/MORPHEUS_dataset bash scripts/00_env.sh
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
for c in bench sw v06 tx v08; do
  [ -e "$HOME/morpheus_work_$c" ] || ln -s "$PKG" "$HOME/morpheus_work_$c"
done
if [ -n "${STORE_DIR:-}" ] && [ ! -e /lambda/nfs/geeg/biorag3_persistent_20260711 ]; then
  sudo mkdir -p /lambda/nfs/geeg && sudo ln -s "$STORE_DIR" /lambda/nfs/geeg/biorag3_persistent_20260711
fi
ls -ld "$HOME"/morpheus_work_{bench,sw,v06,tx,v08}
