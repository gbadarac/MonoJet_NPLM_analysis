import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import os

import mplhep as hep

# Use CMS style for plots
hep.style.use("CMS")

def probs(weights, model_probs):
    """
    Args:
        weights: torch tensor of shape (M,) with requires_grad=True
        model_probs: torch tensor of shape (N, M), where each column is f_i(x) for all x
    Returns:
        p(x) ≈ ∑ w_i * f_i(x) for each x (shape N,)
    """
    return (model_probs * weights).sum(dim=1)  # shape: (N,)

def log_likelihood(weights, model_probs):
    p_x = probs(weights, model_probs) + 1e-8  # prevent log(0)
    ll = torch.log(p_x).mean() #averaging the loss over all datapoints 
    return ll 

def ensemble_pred(weights, model_probs):
    weights = weights.to(model_probs.device)
    model_vals = (model_probs * weights).sum(dim=1)
    return model_vals.cpu().numpy()

def ensemble_unc(cov_w, model_probs):
    # cov_w is (M-1, M-1); propagate through ∂f/∂u_j = f_j - f_M
    if hasattr(model_probs, 'cpu'):
        model_probs_np = model_probs.cpu().clone().numpy()
    else:
        model_probs_np = np.asarray(model_probs)
    if hasattr(cov_w, 'numpy'):
        cov_w = cov_w.numpy()
    g = model_probs_np[:, :-1] - model_probs_np[:, -1:]   # (N, M-1)
    sigma_sq = np.einsum('ni,ij,nj->n', g, cov_w, g)
    return np.sqrt(np.maximum(sigma_sq, 0.0))

def plot_ensemble_marginals_2d(f_i_models, x_data, weights, cov_w, feature_names, outdir):
    #convet pytorch input tensor x_data to a NumPy array for easier processing 
    x = x_data.cpu().numpy() 

    # Loop over each marginal feature
    num_features = x.shape[1]
    for i in range(num_features):
        fig, (ax_main, ax_ratio) = plt.subplots(2,1,figsize=(8, 10), gridspec_kw={'height_ratios': [3,1]})
        feature_label = feature_names[i]
        input_feature = x[:,i] #select input data for feature 1 and 2 separately 

        bins = 40

        # Bin over full data range with margin
        margin = 0.05 * (np.max(input_feature) - np.min(input_feature))
        low, high = np.min(input_feature) - margin, np.max(input_feature) + margin

        #define bin edges, centers an assign each point to a bin 
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)

        # Histogram of the target data
        hist_target_counts, _ = np.histogram(input_feature, bins=bin_edges)
        N_target = np.sum(hist_target_counts)
        hist_target = hist_target_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_target_counts) / (N_target * bin_widths)

        # ------------------------------------------
        # TRUE 1D marginal of each ensemble member (2D case)
        # ------------------------------------------
        # A normalizing flow's marginal has no closed form, so we marginalise the
        # joint density NUMERICALLY: for every bin center c of feature i we integrate
        #     f_m^marg(c) = ∫ f_m(x_i = c, x_j) dx_j
        # over the OTHER feature x_j on a fine grid. This replaces the previous
        # central-slice shortcut (x_j pinned at its mean), which is NOT the marginal
        # and made the NF fit look worse than it is. In 2D the grid integral is exact
        # and cheap; the kernel plotter does the analogous thing analytically, and the
        # 4D plotter uses Monte-Carlo because a dense grid is infeasible there.
        j = 1 - i                                    # the other feature (2D only)
        xj = x[:, j]
        margin_j = 0.05 * (xj.max() - xj.min())
        n_int = 300
        xj_grid = np.linspace(xj.min() - margin_j, xj.max() + margin_j, n_int)
        dxj = xj_grid[1] - xj_grid[0]

        # (B, n_int, 2): every bin center of feature i paired with every x_j node
        B = len(bin_centers)
        CC, GG = np.meshgrid(bin_centers, xj_grid, indexing="ij")   # (B, n_int)
        grid = np.empty((B, n_int, 2), dtype=np.float32)
        grid[:, :, i] = CC
        grid[:, :, j] = GG
        grid_flat = torch.from_numpy(grid.reshape(B * n_int, 2)).float().to(
            next(f_i_models[0].parameters()).device
        )

        with torch.no_grad():
            probs_grid = torch.stack(
                [torch.exp(flow.log_prob(grid_flat)) for flow in f_i_models],
                dim=1  # (B*n_int, M)
            )

        # Integrate out x_j (Riemann sum) -> per-member marginal at each bin center
        probs_marg = probs_grid.view(B, n_int, -1).sum(dim=1) * dxj   # (B, M)

        # Use helper functions (last dim = M, exactly as before)
        f_binned = ensemble_pred(weights, probs_marg)          # shape (B,)
        f_err = ensemble_unc(cov_w, probs_marg)                # shape (B,)

        # Members are already ~unit-normalised; renormalise defensively for the
        # finite grid so the plotted marginal integrates to 1 over the bin range.
        N = np.sum(f_binned * bin_widths)
        f_binned /= N
        f_err /= N

        # ------------------
        # Save marginals as npz
        # ------------------
        out_marginal = os.path.join(outdir, f"marginal_feature_{i+1}_data.npz")
        np.savez_compressed(out_marginal, 
                            f_binned=f_binned,
                            f_err=f_err,
                            bin_centers=bin_centers)

        # ------------------
        # 1 and 2 sigma bands calculation 
        # ------------------

        # Compute 1σ and 2σ bands directly from f_binned and f_err
        band_1s_l = f_binned - f_err
        band_1s_h = f_binned + f_err
        band_2s_l = f_binned - 2 * f_err
        band_2s_h = f_binned + 2 * f_err

        #Mask to keep only bins with target data
        valid_bins = hist_target > 0

        #Plot main distribution
        ax_main.bar(bin_centers, hist_target, width=np.diff(bin_edges), alpha=0.2, label="Target", color='green', edgecolor='black')
        ax_main.errorbar(bin_centers, hist_target, yerr=err_target, fmt='None', color='green', alpha=0.7)
        ax_main.plot(bin_centers[valid_bins], f_binned[valid_bins],'-', color='red', linewidth=1.2, label=r"$f(x) = \sum w_i f_i(x)$")
        ax_main.fill_between(bin_centers, band_1s_l, band_1s_h, alpha=0.15, label=r"$\pm 1\sigma$", color='blue')
        ax_main.fill_between(bin_centers, band_2s_l, band_2s_h, alpha=0.08, label=r"$\pm 2\sigma$", color='purple')

        ax_main.set_xlabel(feature_label, fontsize=16)
        ax_main.set_ylabel("Density", fontsize=16)
        ax_main.legend(fontsize=14)

        #Plot ratios 
        #Avoid division by zero
        f_binned_safe = np.where(f_binned > 0, f_binned, np.nan)

        #Compute lower band ratios
        ratio_1s_h = np.array(band_1s_h) / f_binned_safe #relative size of the lower band 
        ratio_2s_h = np.array(band_2s_h) / f_binned_safe

        ratio_1s_l = np.array(band_1s_l) / f_binned_safe #relative size of the lower band 
        ratio_2s_l = np.array(band_2s_l) / f_binned_safe

        #Only plot valid (non-NaN) ratios
        valid_h = ~np.isnan(ratio_1s_h)

        ax_ratio.plot(bin_centers[valid_h], ratio_1s_h[valid_h], 'o-', color='blue', alpha=0.3, label=r"$+1\sigma$ / mean")
        ax_ratio.plot(bin_centers[valid_h], ratio_2s_h[valid_h], 'o-', color='purple', alpha=0.3, label=r"$+2\sigma$ / mean")
        ax_ratio.plot(bin_centers[valid_h], ratio_1s_l[valid_h], 'o-', color='blue', alpha=0.3, label=r"$-1\sigma$ / mean")
        ax_ratio.plot(bin_centers[valid_h], ratio_2s_l[valid_h], 'o-', color='purple', alpha=0.3, label=r"$-2\sigma$ / mean")
        ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1)  # <-- Add horizontal line at y=1
        ax_ratio.set_ylim(0.9, 1.1) 
        ax_ratio.set_ylabel("Upper band / Mean", fontsize=14)
        ax_ratio.set_xlabel(feature_label, fontsize=14)
        ax_ratio.legend(fontsize=12)
        ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # <-- Enable grid

        #Save plot
        plt.tight_layout()
        outpath = os.path.join(outdir, f"ensemble_marginal_feature_{i+1}.png")
        plt.savefig(outpath)
        plt.close()

def plot_ensemble_marginals_nd(f_i_models, x_data, weights, cov_w, feature_names, outdir,
                            bins=40, K=1024, device="cpu"):
    """1D marginals of the WiFi NF ensemble for ANY dimensionality D >= 2.

    Each feature is marginalised by Monte-Carlo over the other features: the "other"
    columns are drawn from the data (K rows) and the scanned feature is swept over the
    bin centers, then averaged. A dense grid (as used exactly in the 2D plotter) is
    infeasible for D > 2, so MC is used here. Works for D = 2 as well, but for 2D
    prefer plot_ensemble_marginals_2d (exact grid, unbiased).
    """
    x = x_data.cpu().numpy()
    N, D = x.shape
    weights = weights.detach().cpu().double()
    cov_w = torch.from_numpy(cov_w).double()

    # Pre-sample the "other features" once per feature to reduce variance jitter across bins
    rng = np.random.default_rng(1234)
    others_bank = {}
    for i in range(D):
        # K rows, D columns; we will overwrite column i with the scan value
        idx = rng.integers(0, N, size=K)
        X_others = x[idx].copy()  # shape (K, D)
        others_bank[i] = X_others

    for i in range(D):
        fig, (ax_main, ax_ratio) = plt.subplots(2,1,figsize=(8, 10), gridspec_kw={'height_ratios': [3,1]})
        feature_label = feature_names[i]
        xi = x[:, i]

        # binning on the data support
        margin = 0.05 * (xi.max() - xi.min())
        low, high = xi.min() - margin, xi.max() + margin
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)

        # target histogram + errors
        hist_counts, _ = np.histogram(xi, bins=bin_edges)
        N_target = hist_counts.sum()
        hist_target = hist_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_counts) / (N_target * bin_widths)

        # ---- Monte Carlo marginalization over other features ----
        X_others = others_bank[i]  # (K, D)
        # Build (B, K, D) tensor: each bin center paired with the same K draws for other dims
        B = len(bin_centers)
        X_batch = np.repeat(X_others[None, :, :], B, axis=0)  # (B, K, D)
        X_batch[:, :, i] = bin_centers[:, None]               # set the i-th column to the bin center

        # Flatten to (B*K, D) and evaluate all models
        X_flat = torch.from_numpy(X_batch.reshape(B * K, D)).float().to(device)

        with torch.no_grad():
            # probs_per_model: (B*K, M)
            probs_per_model = torch.stack(
                [torch.exp(flow.log_prob(X_flat)).cpu().double() for flow in f_i_models],
                dim=1
            )  # (B*K, M) double on CPU

        # Average over K to get v(c) for each bin center: v(c) is (M,)
        probs_per_model = probs_per_model.view(B, K, -1)      # (B, K, M)
        v_mat = probs_per_model.mean(dim=1)                   # (B, M)

        # Ensemble mean and uncertainty at each center
        w_col = weights.view(-1, 1)                           # (M,1)
        f_binned = (v_mat @ weights).numpy()                  # (B,)
        # sigma^2(c) = g(c)^T Cov_u g(c),  g_j = v_j - v_M  (M-1 free weights)
        g_mat = v_mat[:, :-1] - v_mat[:, -1:]                # (B, M-1)
        sigma2 = (g_mat @ cov_w @ g_mat.T).diagonal().numpy()
        f_err = np.sqrt(np.maximum(sigma2, 0.0))              # (B,)

        # Normalize to unit area over the scan dimension
        area = np.sum(f_binned * bin_widths)
        if area > 0:
            f_binned /= area
            f_err    /= area

        print(f"[feat {i}] f_binned min/max:", f_binned.min(), f_binned.max())
        print(f"[feat {i}] f_err min/max:", f_err.min(), f_err.max())

        # save npz
        out_marginal = os.path.join(outdir, f"marginal_feature_{i+1}_data.npz")
        np.savez_compressed(out_marginal, f_binned=f_binned, f_err=f_err, bin_centers=bin_centers)

        # bands
        band_1s_l = f_binned - f_err
        band_1s_h = f_binned + f_err
        band_2s_l = f_binned - 2*f_err
        band_2s_h = f_binned + 2*f_err

        valid_bins = hist_target > 0
        # Target: green
        ax_main.bar(bin_centers, hist_target, width=bin_widths, alpha=0.2,
                    label="Target", color='green', edgecolor='black')
        ax_main.errorbar(bin_centers, hist_target, yerr=err_target,
                         fmt='None', color='green', alpha=0.7)

        # Ensemble mean: red
        ax_main.plot(bin_centers[valid_bins], f_binned[valid_bins], '-',
                     color='red', linewidth=1.2,
                     label=r"$f(x)=\sum_i w_i f_i(x)$")

        # Bands: blue (±1σ) and purple (±2σ)
        ax_main.fill_between(bin_centers, band_1s_l, band_1s_h,
                             alpha=0.15, color='blue', label=r"$\pm 1\sigma$")
        ax_main.fill_between(bin_centers, band_2s_l, band_2s_h,
                             alpha=0.08, color='purple', label=r"$\pm 2\sigma$")

        ax_main.set_xlabel(feature_label, fontsize=16)
        ax_main.set_ylabel("Density", fontsize=16)
        ax_main.legend(fontsize=14)

        # ratios
        f_safe = np.where(f_binned > 0, f_binned, np.nan)
        r1h = band_1s_h / f_safe
        r2h = band_2s_h / f_safe
        r1l = band_1s_l / f_safe
        r2l = band_2s_l / f_safe
        valid = ~np.isnan(r1h)

        # Ratio lines use same colors: blue/purple
        ax_ratio.plot(bin_centers[valid], r1h[valid], 'o-', color='blue', alpha=0.3, label=r"+1σ / mean")
        ax_ratio.plot(bin_centers[valid], r2h[valid], 'o-', color='purple', alpha=0.3, label=r"+2σ / mean")
        ax_ratio.plot(bin_centers[valid], r1l[valid], 'o-', color='blue', alpha=0.3, label=r"-1σ / mean")
        ax_ratio.plot(bin_centers[valid], r2l[valid], 'o-', color='purple', alpha=0.3, label=r"-2σ / mean")

        ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax_ratio.set_ylim(0.9, 1.1)  # match 2D function
        ax_ratio.set_ylabel("Band / Mean", fontsize=14)
        ax_ratio.set_xlabel(feature_label, fontsize=14)
        ax_ratio.legend(fontsize=12)
        ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        outpath = os.path.join(outdir, f"ensemble_marginal_feature_{i+1}.png")
        plt.savefig(outpath)
        plt.close()
