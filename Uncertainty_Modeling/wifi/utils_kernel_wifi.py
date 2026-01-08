#!/usr/bin/env python

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    hep.style.use("CMS")
except Exception:
    pass

# ------------------
# Core helpers (same spirit as utils_wifi.py)
# ------------------
def probs(weights, model_probs):
    """
    weights: (M,) torch tensor
    model_probs: (N, M) torch tensor with f_i(x_n)
    returns: (N,) torch tensor p(x_n) = sum_i w_i f_i(x_n)
    """
    return (model_probs * weights.view(1, -1)).sum(dim=1)

def log_likelihood(weights, model_probs):
    p_x = probs(weights, model_probs) + 1e-12
    return torch.log(p_x).mean()

def profile_likelihood_scan(model_probs, w_best, out_dir, n_points=200):
    """
    Same scan as in utils_wifi.py:
    scan w_i in [0,1] and rescale others to sum to (1-w_i).
    model_probs: (N,M) torch tensor (CPU ok)
    w_best: array-like length M
    """
    os.makedirs(os.path.join(out_dir, "likelihood_profiles"), exist_ok=True)

    model_probs = model_probs.detach().cpu()
    w_best = np.array(w_best, dtype=float)
    n_models = len(w_best)

    for i in range(n_models):
        w_scan = np.linspace(0.0, 1.0, n_points)
        nll_vals = []

        w_rest = np.delete(w_best, i)
        s = np.sum(w_rest)
        if s <= 0:
            print(f"Skipping w_{i}: sum of rest weights <= 0.")
            continue
        w_rest /= s

        for w_i_val in w_scan:
            w_other = (1.0 - w_i_val) * w_rest
            w_full = np.insert(w_other, i, w_i_val)
            w_tensor = torch.tensor(w_full, dtype=torch.float64)
            nll = -log_likelihood(w_tensor, model_probs.double()).item()
            nll_vals.append(nll)

        nll_vals = np.array(nll_vals)
        if not np.all(np.isfinite(nll_vals)):
            print(f"Skipping w_{i} due to NaN/Inf in NLL values.")
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(w_scan, nll_vals, label="NLL", color="black")
        plt.axvline(w_best[i], color="red", linestyle=":", label="Best fit")
        plt.xlabel(fr"$w_{i}$", fontsize=12)
        plt.ylabel("NLL", fontsize=12)
        plt.title(fr"NLL profile scan for $w_{i}$", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid()
        plt.tight_layout()

        outname = os.path.join(out_dir, "likelihood_profiles", f"profile_scan_w{i}.png")
        plt.savefig(outname)
        plt.close()

def ensemble_pred(weights, model_probs):
    """
    weights: (M,) torch
    model_probs: (B,M) torch
    returns: (B,) numpy
    """
    w = weights.to(model_probs.device)
    return (model_probs * w.view(1, -1)).sum(dim=1).detach().cpu().numpy()

def ensemble_unc(cov_w, model_probs):
    """
    cov_w: (M,M) numpy array
    model_probs: (B,M) torch
    returns: (B,) numpy sigma, with safe sqrt
    """
    mp = model_probs.detach().cpu().numpy()
    sigma_sq = np.einsum("bi,ij,bj->b", mp, cov_w, mp)
    return np.sqrt(np.maximum(sigma_sq, 0.0))

# ------------------
# Kernel specific evaluation
# ------------------
def kernel_eval_probs(kernel_models, x):
    """
    kernel_models: list of kernel members, each supports:
        m.call(x)[-1, :, 0] / m.get_norm()[-1]
    x: (B, d) torch tensor on the correct device
    returns: (B, M) torch tensor
    """
    with torch.no_grad():
        vals = []
        for m in kernel_models:
            p = m.call(x)[-1, :, 0] / m.get_norm()[-1]
            vals.append(p)
        return torch.stack(vals, dim=1)


def plot_ensemble_marginals_2d_kernel(
    kernel_models,
    x_data,
    weights,
    cov_w,
    feature_names,
    outdir,
    bins=40,
    K=1024,
    eval_batch=200000,
    seed=1234,
):
    """
    Proper 2D marginalisation, NF 4D style:
    For feature i, compute v(c) = E_{x_other ~ data}[ f_j(x_i=c, x_other) ] for each model j,
    then ensemble mean f(c) = v(c)^T w and sigma(c)^2 = v(c)^T Cov_w v(c).
    """
    os.makedirs(outdir, exist_ok=True)

    x = x_data.detach().cpu().numpy()
    N, D = x.shape
    assert D == 2, "This function is for 2D inputs only."

    if torch.is_tensor(weights):
        w_t = weights.detach().cpu().double()
    else:
        w_t = torch.tensor(weights, dtype=torch.float64)

    cov_w_np = cov_w  # numpy or None

    # device for kernel evaluation
    try:
        device = next(kernel_models[0].parameters()).device
    except Exception:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)

    for i in range(D):
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1, figsize=(8, 10),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        feature_label = feature_names[i]
        xi = x[:, i]

        margin = 0.05 * (xi.max() - xi.min())
        low, high = xi.min() - margin, xi.max() + margin

        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)

        # Target histogram + errors
        hist_target_counts, _ = np.histogram(xi, bins=bin_edges)
        N_target = hist_target_counts.sum()
        hist_target = hist_target_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_target_counts) / (N_target * bin_widths)

        # Pre sample other feature values from data, NF 4D style
        idx = rng.integers(0, N, size=K)
        X_others = x[idx].copy()  # (K, 2)

        B = len(bin_centers)

        # Build (B, K, 2) batch, overwrite column i with scan value
        X_batch = np.repeat(X_others[None, :, :], B, axis=0)  # (B, K, 2)
        X_batch[:, :, i] = bin_centers[:, None]

        # Flatten to (B*K, 2) and evaluate kernel models in chunks
        X_flat = X_batch.reshape(B * K, D)
        X_flat_t = torch.from_numpy(X_flat).to(device=device, dtype=torch.float32)

        probs_chunks = []
        with torch.no_grad():
            for start in range(0, X_flat_t.shape[0], eval_batch):
                xb = X_flat_t[start:start + eval_batch]
                probs_b = kernel_eval_probs(kernel_models, xb).detach().cpu().double()  # (chunk, M)
                probs_chunks.append(probs_b)

        probs_per_model = torch.cat(probs_chunks, dim=0)  # (B*K, M)

        # Average over K to get v(c) per bin center, shape (B, M)
        probs_per_model = probs_per_model.view(B, K, -1)
        v_mat = probs_per_model.mean(dim=1)  # (B, M)

        # Ensemble mean
        f_binned = (v_mat @ w_t).numpy()  # (B,)

        # Ensemble uncertainty
        if cov_w_np is not None:
            cov_t = torch.from_numpy(cov_w_np).double()
            sigma2 = torch.einsum("bi,ij,bj->b", v_mat, cov_t, v_mat).numpy()
            f_err = np.sqrt(np.maximum(sigma2, 0.0))
        else:
            f_err = np.zeros_like(f_binned)

        # Normalize over x_i axis
        area = np.sum(f_binned * bin_widths)
        if area > 0:
            f_binned /= area
            f_err /= area

        # Save npz like NF code
        out_marginal = os.path.join(outdir, f"marginal_feature_{i+1}_data.npz")
        np.savez_compressed(out_marginal, f_binned=f_binned, f_err=f_err, bin_centers=bin_centers)

        # Bands
        band_1s_l = f_binned - f_err
        band_1s_h = f_binned + f_err
        band_2s_l = f_binned - 2.0 * f_err
        band_2s_h = f_binned + 2.0 * f_err

        valid_bins = hist_target > 0

        # Main plot, same style as your NF plots
        ax_main.bar(
            bin_centers, hist_target, width=bin_widths, alpha=0.2,
            label="Target", color="green", edgecolor="black"
        )
        ax_main.errorbar(
            bin_centers, hist_target, yerr=err_target,
            fmt="None", color="green", alpha=0.7
        )
        ax_main.plot(
            bin_centers[valid_bins], f_binned[valid_bins], "-",
            color="red", linewidth=1.2, label=r"$f(x)=\sum_i w_i f_i(x)$"
        )

        if cov_w_np is not None:
            ax_main.fill_between(
                bin_centers, band_1s_l, band_1s_h,
                alpha=0.15, label=r"$\pm 1\sigma$", color="blue"
            )
            ax_main.fill_between(
                bin_centers, band_2s_l, band_2s_h,
                alpha=0.08, label=r"$\pm 2\sigma$", color="purple"
            )

        ax_main.set_xlabel(feature_label, fontsize=16)
        ax_main.set_ylabel("Density", fontsize=16)
        ax_main.legend(fontsize=14)

        # Ratio plot, identical logic to NF
        f_safe = np.where(f_binned > 0, f_binned, np.nan)
        r1h = band_1s_h / f_safe
        r2h = band_2s_h / f_safe
        r1l = band_1s_l / f_safe
        r2l = band_2s_l / f_safe

        valid = ~np.isnan(r1h)
        if cov_w_np is not None:
            ax_ratio.plot(bin_centers[valid], r1h[valid], "o-", color="blue", alpha=0.3, label=r"$+1\sigma$ / mean")
            ax_ratio.plot(bin_centers[valid], r2h[valid], "o-", color="purple", alpha=0.3, label=r"$+2\sigma$ / mean")
            ax_ratio.plot(bin_centers[valid], r1l[valid], "o-", color="blue", alpha=0.3, label=r"$-1\sigma$ / mean")
            ax_ratio.plot(bin_centers[valid], r2l[valid], "o-", color="purple", alpha=0.3, label=r"$-2\sigma$ / mean")

        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.set_ylabel("Band / Mean", fontsize=14)
        ax_ratio.set_xlabel(feature_label, fontsize=14)
        ax_ratio.legend(fontsize=12)
        ax_ratio.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        outpath = os.path.join(outdir, f"ensemble_marginal_feature_{i+1}.png")
        plt.savefig(outpath)
        plt.close()


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_final_marginals_and_ratio(
    ensemble, x_data, grid, x0, x1, Y, outdir, tag="",
    max_members=8, lw_member=1.2, lw_ens=2.0,
    nbins=40, pad_frac=0.05
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    colors = [
        "#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4",
        "#1d91c0", "#225ea8", "#253494", "#081d58", "purple",
    ]

    Nx = len(x0)
    Ny = len(x1)
    dx0 = float((x0[1] - x0[0]).detach().cpu())
    dx1 = float((x1[1] - x1[0]).detach().cpu())

    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()
    x_data_np = x_data.detach().cpu().numpy()

    # ------------------------------------------------------------
    # RANGE, show the whole distribution
    # Use UNION of data and grid ranges, then clamp to grid support
    # ------------------------------------------------------------
    x0_data_lo, x0_data_hi = float(x_data_np[:, 0].min()), float(x_data_np[:, 0].max())
    x1_data_lo, x1_data_hi = float(x_data_np[:, 1].min()), float(x_data_np[:, 1].max())

    x0_grid_lo, x0_grid_hi = float(x0_np.min()), float(x0_np.max())
    x1_grid_lo, x1_grid_hi = float(x1_np.min()), float(x1_np.max())

    x0_lo = min(x0_data_lo, x0_grid_lo)
    x0_hi = max(x0_data_hi, x0_grid_hi)
    x1_lo = min(x1_data_lo, x1_grid_lo)
    x1_hi = max(x1_data_hi, x1_grid_hi)

    # padding, but do not exceed grid range (ensemble not defined outside)
    x0_pad = pad_frac * (x0_hi - x0_lo + 1e-12)
    x1_pad = pad_frac * (x1_hi - x1_lo + 1e-12)

    x0_lo = max(x0_lo - x0_pad, x0_grid_lo)
    x0_hi = min(x0_hi + x0_pad, x0_grid_hi)
    x1_lo = max(x1_lo - x1_pad, x1_grid_lo)
    x1_hi = min(x1_hi + x1_pad, x1_grid_hi)

    # ------------------------------------------------------------
    # BINNING, same number of bins for both features
    # ------------------------------------------------------------
    bins0 = np.linspace(x0_lo, x0_hi, nbins + 1, dtype=float)
    bins1 = np.linspace(x1_lo, x1_hi, nbins + 1, dtype=float)

    centers0 = 0.5 * (bins0[:-1] + bins0[1:])
    centers1 = 0.5 * (bins1[:-1] + bins1[1:])
    bw0 = np.diff(bins0)
    bw1 = np.diff(bins1)

    # ------------------------------------------------------------
    # Helper, bin a marginal sampled on the x-grid into histogram bins
    # ------------------------------------------------------------
    def bin_from_grid(xgrid, pgrid, bins, dx):
        out = np.zeros(len(bins) - 1, dtype=float)
        for k in range(len(out)):
            lo, hi = bins[k], bins[k + 1]
            if k < len(out) - 1:
                mask = (xgrid >= lo) & (xgrid < hi)
            else:
                mask = (xgrid >= lo) & (xgrid <= hi)
            if np.any(mask):
                integral = pgrid[mask].sum() * dx
                out[k] = integral / (hi - lo)
        return out

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)
    ax_top_left  = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1], sharey=ax_top_left)
    ax_bot_left  = fig.add_subplot(gs[1, 0], sharex=ax_top_left)
    ax_bot_right = fig.add_subplot(gs[1, 1], sharex=ax_top_right)

    # ------------------------------------------------------------
    # Data histograms
    # ------------------------------------------------------------
    counts0, _ = np.histogram(x_data_np[:, 0], bins=bins0)
    counts1, _ = np.histogram(x_data_np[:, 1], bins=bins1)
    Ndata = len(x_data_np)

    H_data0 = counts0 / (Ndata * bw0)
    H_data1 = counts1 / (Ndata * bw1)

    ax_top_left.bar(centers0, H_data0, width=bw0, color="orange", alpha=0.5, label="data")
    ax_top_right.bar(centers1, H_data1, width=bw1, color="orange", alpha=0.5, label="data")

    # ------------------------------------------------------------
    # Ensemble marginals from 2D grid density
    # ------------------------------------------------------------
    Y2 = Y.detach().cpu().view(Ny, Nx)          # (Ny, Nx)

    marg0 = (Y2.sum(dim=0) * dx1).numpy()       # p(x0) evaluated on x0 grid
    marg1 = (Y2.sum(dim=1) * dx0).numpy()       # p(x1) evaluated on x1 grid

    H_ens0 = bin_from_grid(x0_np, marg0, bins0, dx0)
    H_ens1 = bin_from_grid(x1_np, marg1, bins1, dx1)

    ax_top_left.step(centers0, H_ens0, where="mid", color="black", lw=lw_ens, label="Ensemble")
    ax_top_right.step(centers1, H_ens1, where="mid", color="black", lw=lw_ens, label="Ensemble")

    # ------------------------------------------------------------
    # Individual members
    # ------------------------------------------------------------
    n_to_plot = min(max_members, len(ensemble.ensemble), len(colors))
    for ni in range(n_to_plot):
        m = ensemble.ensemble[ni]
        with torch.no_grad():
            z = (m.call(grid)[-1, :, 0] / m.get_norm()[-1]).detach().cpu().view(Ny, Nx)
            m0 = (z.sum(dim=0) * dx1).numpy()
            m1 = (z.sum(dim=1) * dx0).numpy()

        H_m0 = bin_from_grid(x0_np, m0, bins0, dx0)
        H_m1 = bin_from_grid(x1_np, m1, bins1, dx1)

        ax_top_left.step(centers0, H_m0, where="mid", lw=lw_member, color=colors[ni], alpha=0.9)
        ax_top_right.step(centers1, H_m1, where="mid", lw=lw_member, color=colors[ni], alpha=0.9)

    # ------------------------------------------------------------
    # Axes limits, force full range
    # ------------------------------------------------------------
    ax_top_left.set_xlim(bins0[0], bins0[-1])
    ax_bot_left.set_xlim(bins0[0], bins0[-1])
    ax_top_right.set_xlim(bins1[0], bins1[-1])
    ax_bot_right.set_xlim(bins1[0], bins1[-1])

    # ------------------------------------------------------------
    # Ratio + Poisson errors on data only
    # ------------------------------------------------------------
    mask0 = H_ens0 > 0
    mask1 = H_ens1 > 0

    ratio0 = np.full_like(H_data0, np.nan, dtype=float)
    ratio1 = np.full_like(H_data1, np.nan, dtype=float)
    ratio0[mask0] = H_data0[mask0] / H_ens0[mask0]
    ratio1[mask1] = H_data1[mask1] / H_ens1[mask1]

    err_data0 = np.zeros_like(H_data0)
    err_data1 = np.zeros_like(H_data1)
    nonzero0 = counts0 > 0
    nonzero1 = counts1 > 0
    err_data0[nonzero0] = np.sqrt(counts0[nonzero0]) / (Ndata * bw0[nonzero0])
    err_data1[nonzero1] = np.sqrt(counts1[nonzero1]) / (Ndata * bw1[nonzero1])

    ratio_err0 = np.full_like(ratio0, np.nan, dtype=float)
    ratio_err1 = np.full_like(ratio1, np.nan, dtype=float)
    ratio_err0[mask0] = err_data0[mask0] / H_ens0[mask0]
    ratio_err1[mask1] = err_data1[mask1] / H_ens1[mask1]

    ax_bot_left.errorbar(centers0, ratio0, yerr=ratio_err0, fmt="o", color="black", capsize=2, ms=4)
    ax_bot_right.errorbar(centers1, ratio1, yerr=ratio_err1, fmt="o", color="black", capsize=2, ms=4)
    ax_bot_left.axhline(1.0, color="gray", linestyle="--")
    ax_bot_right.axhline(1.0, color="gray", linestyle="--")

    ax_top_left.set_ylabel("Density")
    ax_bot_left.set_ylabel("Data / Ensemble")
    ax_bot_left.set_xlabel("x₀")
    ax_bot_right.set_xlabel("x₁")

    ax_bot_left.set_ylim(0.5, 1.5)
    ax_bot_right.set_ylim(0.5, 1.5)

    ax_top_left.legend(loc="upper left", frameon=False)
    ax_top_right.legend(loc="upper left", frameon=False)

    plt.setp(ax_top_left.get_xticklabels(), visible=False)
    plt.setp(ax_top_right.get_xticklabels(), visible=False)

    fig.set_constrained_layout(True)
    fname = f"marginals_ratio{('_' + tag) if tag else ''}.png"
    fig.savefig(outdir / fname, dpi=200)
    plt.close(fig)
