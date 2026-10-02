#!/usr/bin/env bash
# Build the PLOS ONE revision figures (Fig2-Fig5, S1 Fig) into paper/plos/figures/.
#
#   scripts/make_plos_figures.sh                 # all figures + FIGURE_NOTES.md
#   scripts/make_plos_figures.sh --only Fig4     # a subset (notes file not rewritten)
#
# Figures whose inputs are missing or incomplete are skipped with a message
# (e.g. Fig3 while merge_eval.py is still running, or a stopping config that is
# not fully judged); re-run once the experiments finish.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/Users/ihorkendiukhov/anaconda3/bin/python}"
SCRATCH="${MIW_SCRATCH:-/Volumes/Crucial X6/tmp_miw}"   # external drive, not /private/tmp

export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/mpl}"
export TMPDIR="${TMPDIR_OVERRIDE:-$SCRATCH/tmp}"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"

cd "$REPO"
exec "$PY" scripts/make_plos_figures.py "$@"
