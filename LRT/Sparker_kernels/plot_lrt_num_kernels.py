#!/usr/bin/env python3
"""
plot_lrt_num_kernels.py

For each seed directory in an LRT result folder, generates per-toy diagnostic plots
showing where the numerator's auxiliary Gaussian kernels are located relative to the
pre-trained ensemble marginal density.

Plots produced per seed:
  seed{N}_kernels_2d.png                   -- 2D scatter of kernel centers, colored by coefficient
  seed{N}_kernel_marginal_feat{1,2}.png    -- 1D marginal: ensemble + uncertainty bands (same
                                              format as plot_ensemble_marginals_2d_kernel) with
                                              NUM fitted density overlaid; bottom panel = NUM/Ensemble

Requirements:
  --marginal_npz_dir   Path to the directory containing the pre-computed npz files produced by
                       plot_ensemble_marginals_2d_kernel() (e.g. Fit_Weights results/<K>models/):
                         marginal_feature_1_data.npz   (keys: f_binned, f_err, bin_centers)
                         marginal_feature_2_data.npz

  If --marginal_npz_dir is not provided, the script falls back to plotting only the kernel
  correction curve (old behaviour) without the ensemble base density.

Usage:
  python plot_lrt_num_kernels.py \
      --result_dir   results/<run_tag>/test/ \
      --data_file    /path/to/target.npy \
      --marginal_npz_dir /path/to/wifi/Fit_Weights/results/<K>models/ \
      --feature_names "Feature 1" "Feature 2" \
      --kernel_sigma 0.3
"""

import argparse
import glob
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import norm as scipy_norm


# ---------------------------------------------------------------------------
# Core plot functions
# ---------------------------------------------------------------------------

def _gaussian_1d(x, mu, sigma):
    return scipy_norm.pdf(x, loc=mu, scale=sigma)


def plot_kernel_2d(kernel_centers, kernel_coeffs, x_data, feature_names,
                   outdir, seed_label):
    """
    2D scatter: data (gray) + kernel centers colored by mean-centred coefficient.
    Positive coefficient (red) = NUM adds density there (signal-like excess).
    Negative coefficient (blue) = NUM subtracts density there.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    if x_data is not None:
        ax.scatter(x_data[:, 0], x_data[:, 1],
                   s=1, alpha=0.08, c="gray", rasterized=True, label="Data")

    vmax = np.abs(kernel_coeffs).max()
    if vmax == 0.0:
        vmax = 1.0

    sc = ax.scatter(
        kernel_centers[:, 0], kernel_centers[:, 1],
        c=kernel_coeffs, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        s=60, zorder=5, edgecolors="k", linewidths=0.4,
        label="Kernel centres (colour = coeff)",
    )
    plt.colorbar(sc, ax=ax, label="Mean-centred coefficient")

    n_pos = int((kernel_coeffs > 0).sum())
    n_neg = int((kernel_coeffs < 0).sum())

    ax.set_xlabel(feature_names[0], fontsize=14)
    ax.set_ylabel(feature_names[1], fontsize=14)
    ax.set_title(
        f"LRT NUM kernels — seed {seed_label}\n"
        f"positive: {n_pos}  negative: {n_neg}",
        fontsize=12,
    )
    ax.legend(markerscale=3, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"seed{seed_label}_kernels_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_marginal_1d_overlay(
    kernel_centers, kernel_coeffs,
    x_data,
    marginal_npz_path,
    feature_idx, feature_name, kernel_sigma,
    outdir, seed_label,
):
    """
    1D marginal plot matching plot_ensemble_marginals_2d_kernel() EXACTLY, with one addition:
      Top panel:
        - Green histogram + errorbars (target data, if provided)
        - Red solid line: ensemble mean density f(x) = sum_i w_i f_i(x)
        - Blue fill: ±1σ uncertainty band (from WiFi Cov, delta method)
        - Purple fill: ±2σ uncertainty band
        - Orange dashed line: NUM density = ensemble + kernel corrections
      Bottom panel:
        - NUM / Ensemble ratio per bin (shows kernel correction direction; varies per seed)

    Requires a pre-computed .npz with keys: f_binned, f_err, bin_centers
    (produced by plot_ensemble_marginals_2d_kernel in utils_kernel_wifi.py).
    """
    data_npz = np.load(marginal_npz_path)
    f_binned    = data_npz["f_binned"]     # (B,) ensemble mean density
    f_err       = data_npz["f_err"]        # (B,) 1-sigma uncertainty
    bin_centers = data_npz["bin_centers"]  # (B,)

    dbin = bin_centers[1] - bin_centers[0]

    # Kernel correction: analytical 1D Gaussian marginal
    # p_NUM_marginal(c) = p_ensemble_marginal(c) + sum_k c_k * N(c; mu_k_i, sigma)
    f_corr = np.zeros_like(bin_centers)
    for k in range(len(kernel_coeffs)):
        f_corr += kernel_coeffs[k] * _gaussian_1d(
            bin_centers, kernel_centers[k, feature_idx], kernel_sigma
        )

    f_num = f_binned + f_corr
    # Re-normalise NUM (mean-centred coefficients make sum~0, but enforce exactly)
    area_num = np.sum(f_num * dbin)
    if area_num > 0:
        f_num = f_num / area_num

    # Uncertainty bands (same as original)
    band_1s_l = f_binned - f_err
    band_1s_h = f_binned + f_err
    band_2s_l = f_binned - 2.0 * f_err
    band_2s_h = f_binned + 2.0 * f_err

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 10),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- target data histogram ---
    if x_data is not None:
        xi = x_data[:, feature_idx]
        low  = bin_centers[0]  - dbin / 2.0
        high = bin_centers[-1] + dbin / 2.0
        bin_edges = np.linspace(low, high, len(bin_centers) + 1)
        counts, _ = np.histogram(xi, bins=bin_edges)
        N_tot = counts.sum()
        hist_density = counts / (N_tot * dbin)
        hist_err     = np.sqrt(counts) / (N_tot * dbin)
        ax_main.bar(
            bin_centers, hist_density, width=dbin,
            alpha=0.2, color="green", edgecolor="black", label="Target"
        )
        ax_main.errorbar(
            bin_centers, hist_density, yerr=hist_err,
            fmt="none", color="green", alpha=0.7
        )

    # --- ensemble mean + bands (identical to plot_ensemble_marginals_2d_kernel) ---
    valid = f_binned > 0
    ax_main.plot(
        bin_centers[valid], f_binned[valid], "-",
        color="red", lw=1.2, label=r"$f(x)=\sum_i w_i f_i(x)$"
    )
    ax_main.fill_between(
        bin_centers, band_1s_l, band_1s_h,
        alpha=0.15, color="blue", label=r"$\pm 1\sigma$"
    )
    ax_main.fill_between(
        bin_centers, band_2s_l, band_2s_h,
        alpha=0.08, color="purple", label=r"$\pm 2\sigma$"
    )

    # --- NUM density overlaid ---
    ax_main.plot(
        bin_centers, f_num, "--",
        color="darkorange", lw=2.0, label="NUM (ensemble + kernels)"
    )

    ax_main.set_xlabel(feature_name, fontsize=16)
    ax_main.set_ylabel("Density", fontsize=16)
    ax_main.set_title(f"LRT NUM marginal — seed {seed_label} — {feature_name}", fontsize=12)
    ax_main.legend(fontsize=13)

    # --- ratio panel: NUM / ensemble ---
    f_safe = np.where(f_binned > 0, f_binned, np.nan)
    ratio  = f_num / f_safe

    valid_r = ~np.isnan(ratio)
    ax_ratio.plot(
        bin_centers[valid_r], ratio[valid_r], "o-",
        color="darkorange", alpha=0.8, label="NUM / Ensemble"
    )
    ax_ratio.axhline(1.0, color="black", linestyle="--", lw=1)
    ax_ratio.set_ylim(0.9, 1.1)
    ax_ratio.set_ylabel("NUM / Ensemble", fontsize=14)
    ax_ratio.set_xlabel(feature_name, fontsize=14)
    ax_ratio.legend(fontsize=12)
    ax_ratio.grid(True, which="both", linestyle="--", lw=0.5, alpha=0.7)

    plt.tight_layout()
    fname = f"seed{seed_label}_kernel_marginal_feat{feature_idx + 1}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_marginal_1d_fallback(
    kernel_centers, kernel_coeffs, x_data,
    feature_idx, feature_name, kernel_sigma,
    outdir, seed_label, bins=50, n_grid=600,
):
    """
    Fallback when no pre-computed marginal npz is available.
    Shows data histogram + kernel correction curve only (no ensemble base).
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 9),
                              gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_ticks = axes

    if x_data is not None:
        xi = x_data[:, feature_idx]
    else:
        xi = kernel_centers[:, feature_idx]

    low  = xi.min() - 0.15 * (xi.max() - xi.min())
    high = xi.max() + 0.15 * (xi.max() - xi.min())
    c_grid = np.linspace(low, high, n_grid)

    if x_data is not None:
        counts, edges = np.histogram(xi, bins=bins, range=(low, high))
        N_data   = counts.sum()
        widths   = np.diff(edges)
        bcenters = 0.5 * (edges[:-1] + edges[1:])
        density  = counts / (N_data * widths)
        err      = np.sqrt(counts) / (N_data * widths)
        ax.bar(bcenters, density, width=widths,
               alpha=0.25, color="steelblue", label="Data (this toy)")
        ax.errorbar(bcenters, density, yerr=err,
                    fmt="none", color="steelblue", alpha=0.6)

    f_corr = np.zeros(n_grid)
    for k in range(len(kernel_coeffs)):
        f_corr += kernel_coeffs[k] * _gaussian_1d(
            c_grid, kernel_centers[k, feature_idx], kernel_sigma
        )

    ax.plot(c_grid, f_corr, color="red", lw=2.0,
            label=r"Kernel correction $\sum_k c_k G_k$")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Density / correction", fontsize=13)
    ax.set_title(
        f"LRT NUM kernel marginal — seed {seed_label} — {feature_name}\n"
        "(no pre-computed ensemble marginal provided)",
        fontsize=11,
    )
    ax.legend(fontsize=11)

    vmax = np.abs(kernel_coeffs).max() or 1.0
    cmap = cm.RdBu_r
    for k in range(len(kernel_coeffs)):
        c_norm = 0.5 + 0.5 * kernel_coeffs[k] / vmax
        ax_ticks.axvline(kernel_centers[k, feature_idx],
                         color=cmap(c_norm), alpha=0.7, lw=1.0)
    ax_ticks.set_yticks([])
    ax_ticks.set_xlim(ax.get_xlim())
    ax_ticks.set_xlabel(feature_name, fontsize=13)
    ax_ticks.set_title("Kernel positions (red=positive, blue=negative)", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_ticks, orientation="horizontal", pad=0.25, label="Coefficient")

    fig.tight_layout()
    fname = f"seed{seed_label}_kernel_marginal_feat{feature_idx + 1}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot LRT numerator kernel locations per toy seed."
    )
    parser.add_argument(
        "--result_dir", type=str, required=True,
        help="Path to calibration/ or test/ subdirectory, e.g. results/<tag>/test/",
    )
    parser.add_argument(
        "--data_file", type=str, default=None,
        help=(
            "Data .npy file (or directory of .npy) used as target histogram. "
            "Optional — if omitted, histogram is not shown."
        ),
    )
    parser.add_argument(
        "--marginal_npz_dir", type=str, default=None,
        help=(
            "Directory containing marginal_feature_1_data.npz and "
            "marginal_feature_2_data.npz produced by plot_ensemble_marginals_2d_kernel(). "
            "If provided, plots use the same format as the WiFi marginal plots "
            "(ensemble mean + 1sigma/2sigma bands) with NUM density overlaid. "
            "If omitted, falls back to correction-curve-only format."
        ),
    )
    parser.add_argument(
        "--kernel_sigma", type=float, default=0.3,
        help="Gaussian kernel width used in the LRT run (default 0.3).",
    )
    parser.add_argument(
        "--feature_names", type=str, nargs="+", default=["Feature 1", "Feature 2"],
        help="Feature axis labels.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load global data for target histograms
    # ------------------------------------------------------------------
    data_all = None
    if args.data_file is not None:
        if os.path.isdir(args.data_file):
            files = sorted(glob.glob(os.path.join(args.data_file, "*.npy")))
            data_all = np.concatenate([np.load(f) for f in files], axis=0)
        else:
            data_all = np.load(args.data_file)
        print(f"Loaded target data: {data_all.shape}")
    else:
        print("No --data_file provided. Target histogram will be omitted.")

    # ------------------------------------------------------------------
    # Check for pre-computed marginal npz files
    # ------------------------------------------------------------------
    marginal_npz = [None, None]  # per feature
    if args.marginal_npz_dir is not None:
        for i in range(2):
            p = os.path.join(args.marginal_npz_dir, f"marginal_feature_{i+1}_data.npz")
            if os.path.exists(p):
                marginal_npz[i] = p
                print(f"Found pre-computed marginal for feature {i+1}: {p}")
            else:
                print(f"WARNING: marginal_feature_{i+1}_data.npz not found in {args.marginal_npz_dir}")
    else:
        print("No --marginal_npz_dir provided. Using fallback (correction-curve-only) format.")

    # ------------------------------------------------------------------
    # Iterate over seed directories
    # ------------------------------------------------------------------
    seed_dirs = sorted(glob.glob(os.path.join(args.result_dir, "seed*")))
    print(f"Found {len(seed_dirs)} seed directories in {args.result_dir}")

    n_plotted = 0
    n_skipped = 0

    for seed_dir in seed_dirs:
        seed_label = os.path.basename(seed_dir).replace("seed", "")

        coeffs_path  = os.path.join(seed_dir, f"seed{seed_label}_coeffs.npy")
        centers_path = os.path.join(seed_dir, f"seed{seed_label}_kernel_centers.npy")

        if not os.path.exists(coeffs_path):
            print(f"  skip {seed_dir}: missing seed{seed_label}_coeffs.npy")
            n_skipped += 1
            continue
        if not os.path.exists(centers_path):
            print(
                f"  skip {seed_dir}: missing seed{seed_label}_kernel_centers.npy "
                f"(re-run LRT.py >= current version to save kernel centres)"
            )
            n_skipped += 1
            continue

        kernel_coeffs  = np.load(coeffs_path)   # (n_kernels,)
        kernel_centers = np.load(centers_path)   # (n_kernels, 2)

        # 2D scatter (always produced)
        plot_kernel_2d(
            kernel_centers=kernel_centers,
            kernel_coeffs=kernel_coeffs,
            x_data=data_all,
            feature_names=args.feature_names,
            outdir=seed_dir,
            seed_label=seed_label,
        )

        # 1D marginals
        for feat_idx in range(2):
            npz_path = marginal_npz[feat_idx]
            if npz_path is not None:
                plot_kernel_marginal_1d_overlay(
                    kernel_centers=kernel_centers,
                    kernel_coeffs=kernel_coeffs,
                    x_data=data_all,
                    marginal_npz_path=npz_path,
                    feature_idx=feat_idx,
                    feature_name=args.feature_names[feat_idx],
                    kernel_sigma=args.kernel_sigma,
                    outdir=seed_dir,
                    seed_label=seed_label,
                )
            else:
                plot_kernel_marginal_1d_fallback(
                    kernel_centers=kernel_centers,
                    kernel_coeffs=kernel_coeffs,
                    x_data=data_all,
                    feature_idx=feat_idx,
                    feature_name=args.feature_names[feat_idx],
                    kernel_sigma=args.kernel_sigma,
                    outdir=seed_dir,
                    seed_label=seed_label,
                )

        n_plotted += 1
        print(f"  plotted seed {seed_label}")

    print(f"\nDone. Plotted: {n_plotted}  Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
