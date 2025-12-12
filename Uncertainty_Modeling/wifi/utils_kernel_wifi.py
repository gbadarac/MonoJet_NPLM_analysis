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

