#!/bin/bash
# run_plot_lrt_num_kernels.sh
#
# Generates per-toy LRT NUM kernel diagnostic plots for the 128-model ensemble.
# Run this AFTER the LRT test jobs have completed (CALIBRATION=0).
#
# Produces per seed in each result dir:
#   seed{N}_kernels_2d.png
#   seed{N}_kernel_marginal_feat1.png   (ensemble + 1s/2s bands + NUM density + ratio)
#   seed{N}_kernel_marginal_feat2.png
#
# Usage:
#   bash run_plot_lrt_num_kernels.sh

set -euo pipefail

. /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate kernels_env

# -------------------------
# Paths (edit if ensemble changes)
# -------------------------
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
SCRIPT="$REPO_ROOT/LRT/plot_lrt_num_kernels.py"

# Target data (histogram background in test plots) — MUST be the SAME file the LRT
# test runs against (submit_LRT_toys.sh TARGET_DATA) so the histogram matches the
# tested sample: seed-42 gmm_skew data_test.npy (100k = N_test, oversampling factor 1).
TARGET_DATA="$REPO_ROOT/data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_test.npy"

# Pre-computed WiFi marginal npz files for 128-model ensemble
MARGINAL_NPZ_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernels/N_100000_dim_2_kernels_SparKer_models128_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents128/wifi_ensemble_plots"

# LRT results base
RESULTS_BASE="$REPO_ROOT/LRT/results"

# Run tags for 128-model ensemble (default and frozen)
TAG_DEFAULT="SparKer128_Ntest100000_M100_W0.3_L10000_clip0.0005_wifi_tail_ensemblecomponents128"
TAG_FROZEN="${TAG_DEFAULT}_frozen_weights"

KERNEL_SIGMA=0.3
FEATURE_NAMES=("Feature 1" "Feature 2")

# -------------------------
# Helper
# -------------------------
run_plots() {
    local result_dir="$1"
    local data_file="$2"
    local label="$3"

    echo ""
    echo "=== $label ==="
    echo "  result_dir : $result_dir"
    echo "  data_file  : $data_file"

    if [[ ! -d "$result_dir" ]]; then
        echo "  WARNING: result_dir does not exist yet — skipping."
        return
    fi

    python -u "$SCRIPT" \
        --result_dir    "$result_dir" \
        --data_file     "$data_file" \
        --marginal_npz_dir "$MARGINAL_NPZ_DIR" \
        --kernel_sigma  "$KERNEL_SIGMA" \
        --feature_names "${FEATURE_NAMES[@]}"
}

# -------------------------
# Default mode
# -------------------------
run_plots "$RESULTS_BASE/$TAG_DEFAULT/test"        "$TARGET_DATA"  "default — test"
run_plots "$RESULTS_BASE/$TAG_DEFAULT/calibration" "$TARGET_DATA"  "default — calibration"

# -------------------------
# Frozen mode
# -------------------------
run_plots "$RESULTS_BASE/$TAG_FROZEN/test"         "$TARGET_DATA"  "frozen — test"
run_plots "$RESULTS_BASE/$TAG_FROZEN/calibration"  "$TARGET_DATA"  "frozen — calibration"

echo ""
echo "All done."
