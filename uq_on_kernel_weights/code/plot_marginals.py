"""
Script 2b: Marginal density plots and covariance matrix plot.

Can be re-run without retraining. Reads precomputed data from train_and_uq.py.
Works for any benchmark and any dimensionality (one plot per dimension).

Each marginal plot has:
  - Top panel: training data histogram, HIKER fit (thin line),
    true density (if available), +/-1sigma and +/-2sigma bands
  - Bottom panel: ratio HIKER/true with error bands (if true is available),
    otherwise just the HIKER fit with bands

Also produces a covariance matrix heatmap.

Reads: {OUT_DIR}/data_train.npy, marginals.npz, covariance.npy
Saves: {OUT_DIR}/marginal_x{dim}.png, covariance_matrix.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT_DIR = os.environ.get("HIKER_OUT_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))

# ── Load ──────────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(OUT_DIR, "data_config.json")) as f:
    _dc = json.load(f)
DATA_DIR = _dc.get("data_dir", OUT_DIR)
data_train = np.load(os.path.join(DATA_DIR, "data_train.npy"))

marg = np.load(os.path.join(OUT_DIR, "marginals.npz"), allow_pickle=True)
d = int(marg["d"])
benchmark = str(marg["benchmark"])
N_ENSEMBLE = int(marg.get("N_ENSEMBLE", 1))
print(f"  Benchmark = {benchmark}, d = {d}, N_ENSEMBLE = {N_ENSEMBLE}")

cov_w = np.load(os.path.join(OUT_DIR, "covariance.npy"))
print("  All data loaded.")


# ══════════════════════════════════════════════════════════════════════
# 1. Marginal plots (one per dimension)
# ══════════════════════════════════════════════════════════════════════
print("\nMaking marginal plots...")

for dim in range(d):
    x_grid = marg[f"x{dim}_grid"]
    f_fit = marg[f"f_marg{dim}"]
    sigma_fit = marg[f"sigma_marg{dim}"]
    has_true = f"true_marg{dim}" in marg
    f_true = marg[f"true_marg{dim}"] if has_true else None
    data_dim = data_train[:, dim]
    dim_label = rf"$x_{dim}$"

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    n_bins = 60
    bins = np.linspace(x_grid[0], x_grid[-1], n_bins + 1)

    # ── Top panel ─────────────────────────────────────────────────
    ax_top.hist(data_dim, bins=bins, density=True, color="orange", alpha=0.5,
                label="Training data")
    ax_top.fill_between(x_grid, f_fit - 2 * sigma_fit, f_fit + 2 * sigma_fit,
                        alpha=0.15, color="C0", label=r"$\pm 2\sigma$")
    ax_top.fill_between(x_grid, f_fit - 1 * sigma_fit, f_fit + 1 * sigma_fit,
                        alpha=0.30, color="C0", label=r"$\pm 1\sigma$")
    ax_top.plot(x_grid, f_fit, "C0-", lw=0.8,
                label="Ensemble avg" if N_ENSEMBLE > 1 else "HIKER fit")
    if has_true:
        ax_top.plot(x_grid, f_true, "k--", lw=1.0, label="True")

    # Per-seed marginals (ensemble only)
    if N_ENSEMBLE > 1:
        for s in range(N_ENSEMBLE):
            key = f"f_marg{dim}_seed{s}"
            if key in marg:
                ax_top.plot(x_grid, marg[key], "-", lw=0.3, alpha=0.4, color="gray",
                            label=f"Seeds" if s == 0 else None)

    ax_top.set_ylabel("Marginal density", fontsize=13)
    ax_top.legend(fontsize=9, loc="upper right")
    ax_top.grid(True, alpha=0.3)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Bottom panel ──────────────────────────────────────────────
    if has_true:
        mask = f_true > 1e-15
        ratio = np.ones_like(f_fit)
        ratio[mask] = f_fit[mask] / f_true[mask]
        ratio_sigma = np.zeros_like(sigma_fit)
        ratio_sigma[mask] = sigma_fit[mask] / f_true[mask]

        ax_bot.fill_between(x_grid, ratio - 2 * ratio_sigma, ratio + 2 * ratio_sigma,
                            alpha=0.15, color="C0")
        ax_bot.fill_between(x_grid, ratio - 1 * ratio_sigma, ratio + 1 * ratio_sigma,
                            alpha=0.30, color="C0")
        ax_bot.plot(x_grid, ratio, "C0-", lw=0.8)
        ax_bot.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax_bot.set_ylabel("HIKER / True", fontsize=13)
        ax_bot.set_ylim(0.8, 1.2)
    else:
        # No true available — just show the fit with bands
        ax_bot.fill_between(x_grid, f_fit - 2 * sigma_fit, f_fit + 2 * sigma_fit,
                            alpha=0.15, color="C0")
        ax_bot.fill_between(x_grid, f_fit - 1 * sigma_fit, f_fit + 1 * sigma_fit,
                            alpha=0.30, color="C0")
        ax_bot.plot(x_grid, f_fit, "C0-", lw=0.8)
        ax_bot.set_ylabel("HIKER fit", fontsize=13)

    ax_bot.set_xlabel(dim_label, fontsize=13)
    ax_bot.grid(True, alpha=0.3)

    plt.suptitle(f"Marginal {dim_label}  [{benchmark}]", fontsize=13)
    plt.tight_layout()
    fname = f"marginal_x{dim}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════
# 2. Covariance matrix plot
# ══════════════════════════════════════════════════════════════════════
print("\nMaking covariance matrix plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

im1 = ax1.imshow(cov_w, interpolation="none", cmap="RdBu_r",
                 vmin=-np.abs(cov_w).max(), vmax=np.abs(cov_w).max())
ax1.set_title("Covariance matrix Cov(w)", fontsize=13)
ax1.set_xlabel("Weight index", fontsize=12)
ax1.set_ylabel("Weight index", fontsize=12)
plt.colorbar(im1, ax=ax1, shrink=0.8)

diag = np.sqrt(np.diag(cov_w))
diag_safe = np.where(diag > 0, diag, 1.0)
corr = cov_w / np.outer(diag_safe, diag_safe)
np.fill_diagonal(corr, 1.0)

im2 = ax2.imshow(corr, interpolation="none", cmap="RdBu_r", vmin=-1, vmax=1)
ax2.set_title("Correlation matrix", fontsize=13)
ax2.set_xlabel("Weight index", fontsize=12)
ax2.set_ylabel("Weight index", fontsize=12)
plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "covariance_matrix.png"), dpi=150)
plt.close()
print("  Saved covariance_matrix.png")

print("\nDone.")
