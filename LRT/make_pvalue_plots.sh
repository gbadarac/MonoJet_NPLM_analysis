#!/bin/bash
# Regenerate the per-ensemble p-value-vs-Ntest scan plots (Sparker kernels, 2D).
# The TITLES and the OUTPUT FOLDER live here, not inside plot_pvalue_vs_ntest.py —
# this is the one place to edit the wording or where the figures land.
#
#   figures -> LRT/results/kernels/2d_ensemble/figs/   (next to the runs they summarize)
#
# Usage:  ./make_pvalue_plots.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=/work/gbadarac/miniforge3/envs/kernels_env/bin/python

ENS_BASE=results/kernels/2d_ensemble
SINGLE_BASE=results/kernels/2d_single_model
FIGS=$ENS_BASE/figs

for E in 16 32 64 128; do
  $PY plot_pvalue_vs_ntest.py \
    --base "$ENS_BASE" "$SINGLE_BASE" \
    --nens "$E" --min_ntest 25000 \
    --out  "$FIGS/pvalue_vs_ntest_Nens${E}.pdf" \
    --csv  "$FIGS/pvalue_vs_ntest_Nens${E}.csv" \
    --title "2D GMM · Sparker kernels · Nens=$E vs single"
done
echo "wrote plots to $FIGS/"
