import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/work/gbadarac/zuko")
import zuko

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

def profile_likelihood_scan(model_probs, w_best, out_dir):
    """
    For each weight w_i, scan NLL as a function of w_i while rescaling all others to sum to (1 - w_i).
    """
    os.makedirs(os.path.join(out_dir, "likelihood_profiles"), exist_ok=True)

    n_models = len(w_best)
    w_best = np.array(w_best)

    for i in range(n_models):
        n_points=200
        w_scan = np.linspace(0, 1, n_points)
        nll_vals = []

        # The other weights (not i) from best-fit, normalized to sum to 1 - w_i
        w_rest = np.delete(w_best, i)
        w_rest /= np.sum(w_rest)  # renormalize

        for w_i_val in w_scan:
            w_other = (1.0 - w_i_val) * w_rest
            w_full = np.insert(w_other, i, w_i_val)
            w_tensor = torch.tensor(w_full, dtype=torch.float32)
            nll = -log_likelihood(w_tensor, model_probs).item()
            nll_vals.append(nll)

        nll_vals = np.array(nll_vals)

        # Skip if invalid
        if not np.all(np.isfinite(nll_vals)):
            print(f"Skipping w_{i} due to NaN or Inf in NLL values.")
            continue

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(w_scan[:len(nll_vals)], nll_vals, label="NLL", color="black")
        plt.axvline(w_best[i], color='red', linestyle=':', label="Best fit")

        plt.xlabel(fr"$w_{i}$", fontsize=12)
        plt.ylabel("NLL", fontsize=12)
        plt.title(fr"NLL profile scan for $w_{i}$", fontsize=14)
        plt.ylim(np.min(nll_vals) - 1e-4, np.max(nll_vals) + 1e-4)
        plt.legend(fontsize=10)
        plt.grid()
        plt.tick_params(labelsize=10)
        plt.tight_layout()

        outname = os.path.join(out_dir, "likelihood_profiles", f"profile_scan_w{i}.png")
        plt.savefig(outname)
        plt.close()

def ensemble_pred(weights, model_probs):
    weights = weights.to(model_probs.device)
    model_vals = (model_probs * weights).sum(dim=1)
    return model_vals.cpu().numpy()

def ensemble_unc(cov_w, model_probs):
    model_probs_np = model_probs.cpu().clone().numpy()
    sigma_sq = np.einsum('ni,ij,nj->n', model_probs_np, cov_w, model_probs_np)
    sigma = np.sqrt(sigma_sq)
    return sigma

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
        # Evaluate ensemble model at bin_centers
        # ------------------------------------------
        # Build 2D input with current feature varying and the other fixed (e.g., at mean)
        # Compute mean of the other feature
        x_mean = np.mean(x, axis=0)  # shape (2,)
        x_pad = np.tile(x_mean, (len(bin_centers), 1))  # initialize with mean
        x_pad[:, i] = bin_centers  # vary only the i-th feature
        x_centers_tensor = torch.from_numpy(x_pad).float().to(next(f_i_models[0].parameters()).device)

        # Evaluate all f_i(x_bin_center)
        with torch.no_grad():
            probs_centers = torch.stack(
                [torch.exp(flow.log_prob(x_centers_tensor)) for flow in f_i_models],
                dim=1  # shape: (B, M)
            )

        # Use helper functions
        f_binned = ensemble_pred(weights, probs_centers)       # shape (B,)
        f_err = ensemble_unc(cov_w, probs_centers)              # shape (B,)

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

def plot_ensemble_marginals_4d(f_i_models, x_data, weights, cov_w, feature_names, outdir,
                            bins=40, K=1024, device="cpu"):
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
        # sigma^2(c) = v(c)^T Cov_w v(c)
        cov_w_t = cov_w
        sigma2 = (v_mat @ cov_w_t @ v_mat.T).diagonal().numpy()
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

def plot_gaussian_toy_marginals(model_probs, x_data, weights, cov_w, feature_names, outdir):

    os.makedirs(outdir, exist_ok=True)
    x = x_data.cpu().numpy()
    model_probs_np = model_probs.numpy()
    weights_np = weights.detach().cpu().numpy() if torch.is_tensor(weights) else weights

    N, M = model_probs_np.shape
    D = x.shape[1]

    print(f"[DEBUG] N = {N}, M = {M}, D = {D}")
    print(f"[DEBUG] cov_w diag min/max = {np.min(np.diag(cov_w)):.2e}, {np.max(np.diag(cov_w)):.2e}")

    for i in range(D):
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
        feature = x[:, i]
        feature_label = feature_names[i]
        bins = 40

        # Define binning
        margin = 0.05 * (np.max(feature) - np.min(feature))
        low, high = np.min(feature) - margin, np.max(feature) + margin
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)

        # Target histogram
        hist_target_counts, _ = np.histogram(feature, bins=bin_edges)
        N_target = np.sum(hist_target_counts)
        hist_target = hist_target_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_target_counts) / (N_target * bin_widths)

        print(f"[DEBUG][Feature {i+1}] Target histogram integral ≈ {np.sum(hist_target * bin_widths):.4f}")

        # ------------------------------------------
        # Bin model_probs for each model
        # ------------------------------------------
        f_binned_models = []
        '''
        for m in range(M):
            # Use the current model's weights for each event
            weighted_hist, _ = np.histogram(feature, bins=bin_edges, weights=model_probs_np[:, m])
            weighted_hist /= bin_widths  # Convert to density
            f_binned_models.append(weighted_hist)
        '''
        for m in range(M):
            digitized = np.digitize(feature, bin_edges) - 1  # bin index for each event
            valid = (digitized >= 0) & (digitized < bins)

            binned = np.zeros(bins)
            counts = np.zeros(bins)
            np.add.at(binned, digitized[valid], model_probs_np[valid, m])
            np.add.at(counts, digitized[valid], 1)

            counts[counts == 0] = 1  # to avoid division by zero
            binned /= counts         # average predicted density per bin

            f_binned_models.append(binned)

        f_binned_models = np.array(f_binned_models).T  # shape: (bins, M)


        print(f"[DEBUG][Feature {i+1}] f_binned_models shape = {f_binned_models.shape}")

        # Convert to torch for ensemble ops
        f_binned_models_torch = torch.tensor(f_binned_models.T, dtype=torch.float64)  # (M, bins)
        weights_torch = torch.tensor(weights_np, dtype=torch.float64)
        cov_w_torch = torch.tensor(cov_w, dtype=torch.float64)

        # Evaluate ensemble mean and uncertainty
        f_binned = ensemble_pred(weights_torch, f_binned_models_torch.T)
        f_err = ensemble_unc(cov_w_torch, f_binned_models_torch.T)

        area = np.sum(f_binned * bin_widths)
        f_binned /= area
        f_err /= area


        # Sanity check on area
        print(f"[Feature {i+1}] ∫f(x) dx =", np.sum(f_binned * bin_widths))

        # Plot bands
        band_1s_l = f_binned - f_err
        band_1s_h = f_binned + f_err
        band_2s_l = f_binned - 2 * f_err
        band_2s_h = f_binned + 2 * f_err
        valid_bins = hist_target > 0

        # Main plot
        ax_main.bar(bin_centers, hist_target, width=bin_widths, alpha=0.2, label="Target", color='green', edgecolor='black')
        ax_main.errorbar(bin_centers, hist_target, yerr=err_target, fmt='None', color='green', alpha=0.7)
        ax_main.plot(bin_centers[valid_bins], f_binned[valid_bins], '-', color='red', linewidth=1.2, label=r"$f(x) = \sum w_i f_i(x)$")
        ax_main.fill_between(bin_centers, band_1s_l, band_1s_h, alpha=0.15, label=r"$\pm 1\sigma$", color='blue')
        ax_main.fill_between(bin_centers, band_2s_l, band_2s_h, alpha=0.08, label=r"$\pm 2\sigma$", color='purple')
        ax_main.set_xlabel(feature_label, fontsize=16)
        ax_main.set_ylabel("Density", fontsize=16)
        ax_main.legend(fontsize=14)

        # Ratio plot
        f_binned_safe = np.where(f_binned > 0, f_binned, np.nan)
        ratio_1s_h = band_1s_h / f_binned_safe
        ratio_2s_h = band_2s_h / f_binned_safe
        ratio_1s_l = band_1s_l / f_binned_safe
        ratio_2s_l = band_2s_l / f_binned_safe
        valid_h = ~np.isnan(ratio_1s_h)

        ax_ratio.plot(bin_centers[valid_h], ratio_1s_h[valid_h], 'o-', color='blue', alpha=0.3, label=r"$+1\sigma$ / mean")
        ax_ratio.plot(bin_centers[valid_h], ratio_2s_h[valid_h], 'o-', color='purple', alpha=0.3, label=r"$+2\sigma$ / mean")
        ax_ratio.plot(bin_centers[valid_h], ratio_1s_l[valid_h], 'o-', color='blue', alpha=0.3, label=r"$-1\sigma$ / mean")
        ax_ratio.plot(bin_centers[valid_h], ratio_2s_l[valid_h], 'o-', color='purple', alpha=0.3, label=r"$-2\sigma$ / mean")
        ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax_ratio.set_ylim(0.9, 1.1)
        ax_ratio.set_ylabel("Upper band / Mean", fontsize=14)
        ax_ratio.set_xlabel(feature_label, fontsize=14)
        ax_ratio.legend(fontsize=12)
        ax_ratio.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Save
        plt.tight_layout()
        outpath = os.path.join(outdir, f"gaussian_toy_marginal_feature_{i+1}.png")
        plt.savefig(outpath)
        plt.close()
