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

def plot_ensemble_marginals_2d(f_i_models, x_data, weights, cov_w, feature_names, outdir,
                               bins=60, n_curve=300, n_components=None,
                               estimator_label="Normalizing Flow", dataset_label="2D toy model"):
    """All feature marginals in ONE figure (2 rows x n_features cols: density on top,
    band/mean ratio below), with a title stating the density-estimator type and the
    number of ensemble components.

    The green target HISTOGRAM (``bins``) and the red model curve + uncertainty band
    (evaluated on a dense ``n_curve`` grid) are DECOUPLED: the model marginal is drawn
    smoothly regardless of the histogram binning, so wiggles in the red line are real
    density structure, not an artifact of coarse bins. Per-feature marginal_feature_*.npz
    (keys f_binned/f_err/bin_centers, now on the dense grid) are still written for the
    downstream LRT / NPLM consumers.
    """
    x = x_data.cpu().numpy()
    num_features = x.shape[1]

    fig, axes = plt.subplots(
        2, num_features, figsize=(6 * num_features, 9),
        gridspec_kw={'height_ratios': [3, 1]}, squeeze=False,
    )

    for i in range(num_features):
        ax_main, ax_ratio = axes[0, i], axes[1, i]
        feature_label = feature_names[i]
        input_feature = x[:, i]

        # Bin over full data range with margin
        margin = 0.05 * (np.max(input_feature) - np.min(input_feature))
        low, high = np.min(input_feature) - margin, np.max(input_feature) + margin

        # --- target histogram (thin bins, decoupled from the model curve) ---
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers_hist = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        hist_target_counts, _ = np.histogram(input_feature, bins=bin_edges)
        N_target = np.sum(hist_target_counts)
        hist_target = hist_target_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_target_counts) / (N_target * bin_widths)

        # ------------------------------------------
        # TRUE 1D marginal of each ensemble member (2D case), on a DENSE grid
        # ------------------------------------------
        # A normalizing flow's marginal has no closed form, so we marginalise the
        # joint density NUMERICALLY: for every eval point c of feature i we integrate
        #     f_m^marg(c) = ∫ f_m(x_i = c, x_j) dx_j
        # over the OTHER feature x_j on a fine grid. The eval points x_eval are a dense
        # (n_curve) grid independent of the histogram bins -> smooth red curve + band.
        j = 1 - i                                    # the other feature (2D only)
        xj = x[:, j]
        margin_j = 0.05 * (xj.max() - xj.min())
        n_int = 300
        xj_grid = np.linspace(xj.min() - margin_j, xj.max() + margin_j, n_int)
        dxj = xj_grid[1] - xj_grid[0]

        x_eval = np.linspace(low, high, n_curve)
        dx_eval = x_eval[1] - x_eval[0]

        # (n_curve, n_int, 2): every eval point of feature i paired with every x_j node
        CC, GG = np.meshgrid(x_eval, xj_grid, indexing="ij")          # (n_curve, n_int)
        grid = np.empty((n_curve, n_int, 2), dtype=np.float32)
        grid[:, :, i] = CC
        grid[:, :, j] = GG
        grid_flat = torch.from_numpy(grid.reshape(n_curve * n_int, 2)).float().to(
            next(f_i_models[0].parameters()).device
        )

        with torch.no_grad():
            probs_grid = torch.stack(
                [torch.exp(flow.log_prob(grid_flat)) for flow in f_i_models],
                dim=1  # (n_curve*n_int, M)
            )

        # Integrate out x_j (Riemann sum) -> per-member marginal at each eval point
        probs_marg = probs_grid.view(n_curve, n_int, -1).sum(dim=1) * dxj   # (n_curve, M)

        f_binned = ensemble_pred(weights, probs_marg)          # (n_curve,)
        f_err = ensemble_unc(cov_w, probs_marg)                # (n_curve,)

        # Members are already ~unit-normalised; renormalise defensively for the
        # finite grid so the plotted marginal integrates to 1 over the eval range.
        N = np.sum(f_binned * dx_eval)
        f_binned /= N
        f_err /= N

        # ------------------
        # Save marginals as npz (dense grid; keys unchanged for downstream consumers)
        # ------------------
        out_marginal = os.path.join(outdir, f"marginal_feature_{i+1}_data.npz")
        np.savez_compressed(out_marginal,
                            f_binned=f_binned,
                            f_err=f_err,
                            bin_centers=x_eval)

        # 1σ and 2σ bands
        band_1s_l = f_binned - f_err
        band_1s_h = f_binned + f_err
        band_2s_l = f_binned - 2 * f_err
        band_2s_h = f_binned + 2 * f_err

        # Red curve only across the data support (where the target has events)
        valid_curve = (x_eval >= input_feature.min()) & (x_eval <= input_feature.max())

        # --- main density panel ---
        ax_main.bar(bin_centers_hist, hist_target, width=bin_widths, alpha=0.2,
                    label="Target", color='green', edgecolor='black')
        ax_main.errorbar(bin_centers_hist, hist_target, yerr=err_target, fmt='None',
                         color='green', alpha=0.7)
        ax_main.plot(x_eval[valid_curve], f_binned[valid_curve], '-', color='red',
                     linewidth=1.5, label=r"$f(x) = \sum w_i f_i(x)$")
        ax_main.fill_between(x_eval, band_1s_l, band_1s_h, alpha=0.15,
                             label=r"$\pm 1\sigma$", color='blue')
        ax_main.fill_between(x_eval, band_2s_l, band_2s_h, alpha=0.08,
                             label=r"$\pm 2\sigma$", color='purple')
        ax_main.set_ylabel("Density", fontsize=15)
        ax_main.legend(fontsize=12)

        # --- ratio panel (band / mean) ---
        f_binned_safe = np.where(f_binned > 0, f_binned, np.nan)
        ratio_1s_h = np.array(band_1s_h) / f_binned_safe
        ratio_2s_h = np.array(band_2s_h) / f_binned_safe
        ratio_1s_l = np.array(band_1s_l) / f_binned_safe
        ratio_2s_l = np.array(band_2s_l) / f_binned_safe
        valid_h = ~np.isnan(ratio_1s_h)

        ax_ratio.plot(x_eval[valid_h], ratio_1s_h[valid_h], '-', color='blue', alpha=0.4, label=r"$+1\sigma$ / mean")
        ax_ratio.plot(x_eval[valid_h], ratio_2s_h[valid_h], '-', color='purple', alpha=0.4, label=r"$+2\sigma$ / mean")
        ax_ratio.plot(x_eval[valid_h], ratio_1s_l[valid_h], '-', color='blue', alpha=0.4)
        ax_ratio.plot(x_eval[valid_h], ratio_2s_l[valid_h], '-', color='purple', alpha=0.4)
        ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.set_ylabel("Band / mean", fontsize=13)
        ax_ratio.set_xlabel(feature_label, fontsize=15)
        ax_ratio.legend(fontsize=10)
        ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Figure title, e.g. "2D toy model marginals for Normalizing Flow (8 ensemble components)"
    title = f"{dataset_label} marginals for {estimator_label}"
    if n_components is not None:
        comp_word = "component" if n_components == 1 else "components"
        title += f" ({n_components} ensemble {comp_word})"
    fig.suptitle(title, fontsize=18)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = os.path.join(outdir, "ensemble_marginals.png")
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)

def plot_ensemble_marginals_nd(f_i_models, x_data, weights, cov_w, feature_names, outdir,
                            bins=40, n_samples=30000, device=None):
    """1D marginals of the WiFi NF ensemble for ANY dimensionality D >= 2.

    The marginal of feature j is computed EXACTLY via the linearity of the mixture:
        p_marg(x_j) = int sum_i w_i f_i(x) dx_{-j} = sum_i w_i * m_i(x_j)
    where m_i(x_j) = int f_i dx_{-j} is member i's own marginal. Each member is a
    proper normalized flow, so m_i is obtained UNBIASED by sampling that member and
    histogramming feature j (exactly what the training-side sample plot does).

    This replaces the old "data-proposal" estimator (hold the other features at data
    values and average the joint density), which actually computes
    int f(x_j, x_-j) p_data(x_-j) dx_-j -- equal to the true marginal ONLY when x_j is
    independent of the other features. That bias is invisible for a product density
    (the 2D toy) but severe once the features are correlated (the 4D embedding), where
    it produces a spurious spike at the mode.

    Reduces to the exact single-member sample marginal for M = 1 (band collapses to 0).
    """
    x = x_data.cpu().numpy()
    N, D = x.shape
    weights = weights.detach().cpu().double()
    cov_w = torch.from_numpy(cov_w).double()

    # Sample each member ONCE (reused across all D features). The flows are passed on
    # CPU; move each to the sampling device (GPU if available) and back, since
    # autoregressive sampling is slow on CPU. Reproducible per member.
    dev = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_samples = []
    with torch.no_grad():
        for j, flow in enumerate(f_i_models):
            torch.manual_seed(1234 + j)
            flow.to(dev)
            member_samples.append(flow.sample(n_samples).cpu().numpy())  # (n_samples, D)
            flow.to("cpu")

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

        # ---- Exact per-member marginal via sampling (unbiased under correlation) ----
        # v_mat[b, j] = m_j(bin_center_b): member j's marginal density of feature i, from
        # that member's own samples. Normalized by the TOTAL sample count (samples outside
        # the plot range are dropped, so each column integrates to ~1 over the shown
        # support); the weighted sum f_binned is renormalized to unit area just below.
        B = len(bin_centers)
        M = len(f_i_models)
        v_mat = torch.empty((B, M), dtype=torch.float64)
        for j, s in enumerate(member_samples):
            counts, _ = np.histogram(s[:, i], bins=bin_edges)
            v_mat[:, j] = torch.from_numpy(counts / (len(s) * bin_widths)).double()

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
