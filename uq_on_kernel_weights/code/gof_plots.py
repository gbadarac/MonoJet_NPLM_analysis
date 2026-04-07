"""
Shared GoF plotting functions used by run_gof.py and plot_gof.py.

Produces three marginal comparison plots per dimension per variant:
  1. Ratio to Data (binned)
  2. Ratio to True (continuous) — if analytic true is available
  3. Ratio to HIKER (continuous for models, binned for data/true)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d


def _bin_curve(xg, fg, bins, n_bins):
    """Average a continuous curve into histogram bins."""
    f_interp = interp1d(xg, fg, kind='linear', bounds_error=False, fill_value=0)
    return np.array([f_interp(np.linspace(bins[b], bins[b+1], 20)).mean()
                     for b in range(n_bins)])


def _top_panel(ax, x_obs_dim, bins, x_grid, f_hiker, sigma_hiker,
               f_den, f_num, true_marg):
    """Draw the top panel (shared across all three plot types)."""
    ax.hist(x_obs_dim, bins=bins, density=True, color="orange", alpha=0.5,
            label="Test data")
    ax.fill_between(x_grid,
                    f_hiker - 2 * sigma_hiker,
                    f_hiker + 2 * sigma_hiker,
                    alpha=0.12, color="C0", label=r"HIKER $\pm 2\sigma$")
    ax.fill_between(x_grid,
                    f_hiker - 1 * sigma_hiker,
                    f_hiker + 1 * sigma_hiker,
                    alpha=0.25, color="C0", label=r"HIKER $\pm 1\sigma$")
    if true_marg is not None:
        ax.plot(x_grid, true_marg, "k--", lw=1.0, label="True")
    ax.plot(x_grid, f_hiker, "-", lw=0.8, color="C0", label="HIKER fit")
    ax.plot(x_grid, f_den, "-", lw=0.8, color="C1", label="Denominator")
    ax.plot(x_grid, f_num, "-", lw=0.8, color="C2", label="Numerator")
    ax.set_ylabel("Marginal density", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)


def plot_ratio_to_data(out_dir, dim, dim_label, label, suffix,
                       x_obs_dim, x_grid,
                       f_hiker, sigma_hiker, f_den, f_num, true_marg):
    """
    Plot 1: binned ratios to test data histogram.
    All lines are binned. True/Data shown as dots without errors.
    """
    n_bins = 60
    bins = np.linspace(x_grid[0], x_grid[-1], n_bins + 1)
    bin_w = bins[1] - bins[0]
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    h_data_counts, _ = np.histogram(x_obs_dim, bins=bins)
    N_data = len(x_obs_dim)
    h_data = h_data_counts / (N_data * bin_w)

    h_hiker = _bin_curve(x_grid, f_hiker, bins, n_bins)
    h_hiker_sigma = _bin_curve(x_grid, sigma_hiker, bins, n_bins)
    h_den = _bin_curve(x_grid, f_den, bins, n_bins)
    h_num = _bin_curve(x_grid, f_num, bins, n_bins)
    h_true = _bin_curve(x_grid, true_marg, bins, n_bins) if true_marg is not None else None

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    _top_panel(ax_top, x_obs_dim, bins, x_grid, f_hiker, sigma_hiker,
               f_den, f_num, true_marg)

    mask = h_data > 0
    data_sigma = np.zeros(n_bins)
    data_sigma[mask] = np.sqrt(h_data_counts[mask]) / (N_data * bin_w)

    def ratio_err(h_model, h_model_sigma=None):
        r = np.full(n_bins, np.nan)
        r_err = np.full(n_bins, np.nan)
        r[mask] = h_model[mask] / h_data[mask]
        if h_model_sigma is not None:
            r_err[mask] = np.sqrt(
                (h_model_sigma[mask] / h_data[mask]) ** 2 +
                (h_model[mask] * data_sigma[mask] / h_data[mask] ** 2) ** 2)
        else:
            r_err[mask] = np.abs(h_model[mask]) * data_sigma[mask] / h_data[mask] ** 2
        return r, r_err

    r_hiker, r_hiker_err = ratio_err(h_hiker, h_hiker_sigma)
    r_den, r_den_err = ratio_err(h_den)
    r_num, r_num_err = ratio_err(h_num)

    ax_bot.errorbar(bin_centres, r_hiker, yerr=r_hiker_err,
                    fmt='o', ms=3, lw=0.8, capsize=2, color="C0", label="HIKER / Data")
    ax_bot.errorbar(bin_centres - bin_w * 0.1, r_den, yerr=r_den_err,
                    fmt='s', ms=2.5, lw=0.6, capsize=1.5, color="C1", label="Den / Data")
    ax_bot.errorbar(bin_centres + bin_w * 0.1, r_num, yerr=r_num_err,
                    fmt='^', ms=2.5, lw=0.6, capsize=1.5, color="C2", label="Num / Data")

    if h_true is not None:
        r_true = np.full(n_bins, np.nan)
        r_true[mask] = h_true[mask] / h_data[mask]
        ax_bot.scatter(bin_centres, r_true, s=10, color="black", zorder=5,
                       label="True / Data")

    ax_bot.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax_bot.set_xlabel(dim_label, fontsize=13)
    ax_bot.set_ylabel("Ratio to Data", fontsize=13)
    ax_bot.set_ylim(0.7, 1.3)
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.3)

    plt.suptitle(f"GoF vs Data [{label}]", fontsize=13)
    plt.tight_layout()
    fname = f"gof_marginals_x{dim}{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    return fname


def plot_ratio_to_true(out_dir, dim, dim_label, label, suffix,
                       x_obs_dim, x_grid,
                       f_hiker, sigma_hiker, f_den, f_num, true_marg):
    """
    Plot 2: continuous ratios to analytic true density.
    All lines continuous. HIKER has error bands.
    """
    if true_marg is None:
        return None

    n_bins = 60
    bins = np.linspace(x_grid[0], x_grid[-1], n_bins + 1)

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    _top_panel(ax_top, x_obs_dim, bins, x_grid, f_hiker, sigma_hiker,
               f_den, f_num, true_marg)

    mask = true_marg > 1e-15
    r_hiker = np.ones_like(f_hiker)
    r_sigma = np.zeros_like(sigma_hiker)
    r_den = np.ones_like(f_den)
    r_num = np.ones_like(f_num)
    r_hiker[mask] = f_hiker[mask] / true_marg[mask]
    r_sigma[mask] = sigma_hiker[mask] / true_marg[mask]
    r_den[mask] = f_den[mask] / true_marg[mask]
    r_num[mask] = f_num[mask] / true_marg[mask]

    ax_bot.fill_between(x_grid, r_hiker - 2 * r_sigma, r_hiker + 2 * r_sigma,
                        alpha=0.12, color="C0")
    ax_bot.fill_between(x_grid, r_hiker - 1 * r_sigma, r_hiker + 1 * r_sigma,
                        alpha=0.25, color="C0")
    ax_bot.plot(x_grid, r_hiker, "-", lw=0.8, color="C0", label="HIKER / True")
    ax_bot.plot(x_grid, r_den, "-", lw=0.8, color="C1", label="Den / True")
    ax_bot.plot(x_grid, r_num, "-", lw=0.8, color="C2", label="Num / True")
    ax_bot.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax_bot.set_xlabel(dim_label, fontsize=13)
    ax_bot.set_ylabel("Ratio to True", fontsize=13)
    ax_bot.set_ylim(0.8, 1.2)
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    plt.suptitle(f"GoF vs True [{label}]", fontsize=13)
    plt.tight_layout()
    fname = f"gof_vs_true_x{dim}{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    return fname


def plot_ratio_to_hiker(out_dir, dim, dim_label, label, suffix,
                        x_obs_dim, x_grid,
                        f_hiker, sigma_hiker, f_den, f_num, true_marg):
    """
    Plot 3: ratios to HIKER fit.
    Model lines (Den, Num, True) are continuous.
    Data / HIKER is binned with Poisson errors.
    HIKER / HIKER = 1 with error band.
    """
    n_bins = 60
    bins = np.linspace(x_grid[0], x_grid[-1], n_bins + 1)
    bin_w = bins[1] - bins[0]
    bin_centres = 0.5 * (bins[:-1] + bins[1:])

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    _top_panel(ax_top, x_obs_dim, bins, x_grid, f_hiker, sigma_hiker,
               f_den, f_num, true_marg)

    # Continuous ratios: model / HIKER
    mask_h = f_hiker > 1e-15
    r_sigma = np.zeros_like(sigma_hiker)
    r_sigma[mask_h] = sigma_hiker[mask_h] / f_hiker[mask_h]

    r_den = np.ones_like(f_den)
    r_num = np.ones_like(f_num)
    r_den[mask_h] = f_den[mask_h] / f_hiker[mask_h]
    r_num[mask_h] = f_num[mask_h] / f_hiker[mask_h]

    # HIKER / HIKER = 1 with error band
    ax_bot.fill_between(x_grid, 1 - 2 * r_sigma, 1 + 2 * r_sigma,
                        alpha=0.12, color="C0", label=r"HIKER $\pm 2\sigma$")
    ax_bot.fill_between(x_grid, 1 - 1 * r_sigma, 1 + 1 * r_sigma,
                        alpha=0.25, color="C0", label=r"HIKER $\pm 1\sigma$")
    ax_bot.plot(x_grid, r_den, "-", lw=0.8, color="C1", label="Den / HIKER")
    ax_bot.plot(x_grid, r_num, "-", lw=0.8, color="C2", label="Num / HIKER")

    # True / HIKER (continuous, if available)
    if true_marg is not None:
        r_true = np.ones_like(true_marg)
        r_true[mask_h] = true_marg[mask_h] / f_hiker[mask_h]
        ax_bot.plot(x_grid, r_true, "k--", lw=0.8, label="True / HIKER")

    # Data / HIKER (binned with Poisson errors)
    h_data_counts, _ = np.histogram(x_obs_dim, bins=bins)
    N_data = len(x_obs_dim)
    h_data = h_data_counts / (N_data * bin_w)
    h_hiker = _bin_curve(x_grid, f_hiker, bins, n_bins)

    mask_b = (h_hiker > 1e-15) & (h_data_counts > 0)
    r_data = np.full(n_bins, np.nan)
    r_data_err = np.full(n_bins, np.nan)
    r_data[mask_b] = h_data[mask_b] / h_hiker[mask_b]
    r_data_err[mask_b] = np.sqrt(h_data_counts[mask_b]) / (N_data * bin_w) / h_hiker[mask_b]

    ax_bot.errorbar(bin_centres, r_data, yerr=r_data_err,
                    fmt='o', ms=3, lw=0.8, capsize=2, color="orange", label="Data / HIKER")

    ax_bot.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax_bot.set_xlabel(dim_label, fontsize=13)
    ax_bot.set_ylabel("Ratio to HIKER", fontsize=13)
    ax_bot.set_ylim(0.7, 1.3)
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.3)

    plt.suptitle(f"GoF vs HIKER [{label}]", fontsize=13)
    plt.tight_layout()
    fname = f"gof_vs_hiker_x{dim}{suffix}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    return fname


def make_all_marginal_plots(out_dir, dim, dim_label, label, suffix,
                            x_obs_dim, x_grid,
                            f_hiker, sigma_hiker, f_den, f_num, true_marg):
    """Produce all three marginal comparison plots for one dimension."""
    fnames = []
    f = plot_ratio_to_data(out_dir, dim, dim_label, label, suffix,
                           x_obs_dim, x_grid,
                           f_hiker, sigma_hiker, f_den, f_num, true_marg)
    fnames.append(f)
    print(f"  Saved {f}")

    f = plot_ratio_to_true(out_dir, dim, dim_label, label, suffix,
                           x_obs_dim, x_grid,
                           f_hiker, sigma_hiker, f_den, f_num, true_marg)
    if f:
        fnames.append(f)
        print(f"  Saved {f}")

    f = plot_ratio_to_hiker(out_dir, dim, dim_label, label, suffix,
                            x_obs_dim, x_grid,
                            f_hiker, sigma_hiker, f_den, f_num, true_marg)
    fnames.append(f)
    print(f"  Saved {f}")

    return fnames
