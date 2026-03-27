#!/usr/bin/env python3
"""
plot_lrt_num_kernels.py

For each seed directory in an LRT result folder, generates per-toy diagnostic plots
showing where the numerator's auxiliary Gaussian kernels are located and how large
their coefficients are.

Plots produced per seed:
  seed{N}_kernels_2d.png          -- 2D scatter of kernel centers, colored by coefficient
  seed{N}_kernel_marginal_feat1.png  -- 1D marginal: kernel correction vs feature 1
  seed{N}_kernel_marginal_feat2.png  -- 1D marginal: kernel correction vs feature 2

Usage:
  python plot_lrt_num_kernels.py \
      --result_dir results/<run_tag>/test/ \
      --data_file  /path/to/target.npy \
      --feature_names MET HT \
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
    """Evaluate normalised 1D Gaussian pdf."""
    return scipy_norm.pdf(x, loc=mu, scale=sigma)


def plot_kernel_2d(kernel_centers, kernel_coeffs, x_data, feature_names,
                   outdir, seed_label):
    """
    2D scatter: data (gray) + kernel centers colored by mean-centred coefficient.
    Positive coefficient → red (signal-like excess over background).
    Negative coefficient → blue (deficit).
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
    n_sat = int((np.abs(kernel_coeffs) >= np.abs(kernel_coeffs).max() * 0.999).sum())

    ax.set_xlabel(feature_names[0], fontsize=14)
    ax.set_ylabel(feature_names[1], fontsize=14)
    ax.set_title(
        f"LRT NUM kernels — seed {seed_label}\n"
        f"pos: {n_pos}  neg: {n_neg}  (saturated≈{n_sat})",
        fontsize=12,
    )
    ax.legend(markerscale=3, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"seed{seed_label}_kernels_2d.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_marginal_1d(kernel_centers, kernel_coeffs, x_data, feature_idx,
                             feature_name, kernel_sigma, outdir, seed_label,
                             bins=50, n_grid=600):
    """
    1D marginal plot for one feature:
      - Data histogram (if x_data is provided)
      - Analytical kernel correction: f_corr(c) = sum_k coeff_k * N(c; mu_k[i], sigma)
      - Vertical tick marks at kernel center positions, coloured by coefficient
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

    # ------ data histogram ------
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

    # ------ kernel correction curve ------
    f_corr = np.zeros(n_grid)
    for k in range(len(kernel_coeffs)):
        f_corr += kernel_coeffs[k] * _gaussian_1d(c_grid, kernel_centers[k, feature_idx],
                                                    kernel_sigma)

    ax.plot(c_grid, f_corr, color="red", lw=2.0,
            label="Kernel correction $\\sum_k c_k\\, G_k$")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")

    ax.set_ylabel("Density / correction", fontsize=13)
    ax.set_title(
        f"LRT NUM kernel marginal — seed {seed_label} — {feature_name}",
        fontsize=12,
    )
    ax.legend(fontsize=11)

    # ------ tick marks coloured by coefficient ------
    vmax = np.abs(kernel_coeffs).max()
    if vmax == 0.0:
        vmax = 1.0
    cmap = cm.RdBu_r
    for k in range(len(kernel_coeffs)):
        c_norm = 0.5 + 0.5 * kernel_coeffs[k] / vmax   # 0..1 for RdBu_r
        colour  = cmap(c_norm)
        ax_ticks.axvline(kernel_centers[k, feature_idx],
                         color=colour, alpha=0.7, lw=1.0)

    ax_ticks.set_yticks([])
    ax_ticks.set_xlim(ax.get_xlim())
    ax_ticks.set_xlabel(feature_name, fontsize=13)
    ax_ticks.set_title("Kernel positions (red=positive, blue=negative)", fontsize=11)

    # colourbar for tick marks
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_ticks, orientation="horizontal", pad=0.25,
                 label="Coefficient")

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
            "Data .npy file (or directory of .npy) used as background histogram. "
            "Optional — if omitted, only the kernel correction curve is plotted."
        ),
    )
    parser.add_argument(
        "--kernel_sigma", type=float, default=0.3,
        help="Gaussian kernel width used in the LRT run (default 0.3).",
    )
    parser.add_argument(
        "--feature_names", type=str, nargs="+", default=["Feature 1", "Feature 2"],
        help="Feature axis labels (default: 'Feature 1' 'Feature 2').",
    )
    parser.add_argument(
        "--ntest", type=int, default=None,
        help=(
            "Number of test events per toy (for reconstructing background; "
            "not needed when --data_file is provided)."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load global data for background histograms
    # ------------------------------------------------------------------
    data_all = None
    if args.data_file is not None:
        if os.path.isdir(args.data_file):
            files = sorted(glob.glob(os.path.join(args.data_file, "*.npy")))
            data_all = np.concatenate([np.load(f) for f in files], axis=0)
        else:
            data_all = np.load(args.data_file)
        print(f"Loaded background data: {data_all.shape} from {args.data_file}")
    else:
        print("No --data_file provided. Background histogram will be omitted.")

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
                f"(re-run LRT.py ≥ current version to save kernel centres)"
            )
            n_skipped += 1
            continue

        kernel_coeffs  = np.load(coeffs_path)   # (n_kernels,) mean-centred
        kernel_centers = np.load(centers_path)   # (n_kernels, 2)

        assert kernel_centers.shape[1] == 2, \
            f"Expected 2D kernel centres, got shape {kernel_centers.shape}"
        assert len(args.feature_names) == 2, \
            "Need exactly 2 feature names for 2D data."

        # 2D scatter
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
            plot_kernel_marginal_1d(
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

    print(f"\nDone. Plotted: {n_plotted}  Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
