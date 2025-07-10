import numpy as np
import torch
import torch.nn.functional as F
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
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

def make_flow(num_layers, hidden_features, num_bins, num_blocks, num_features=2, num_context=None, perm=True):
    base_dist = StandardNormal(shape=(num_features,))
    transforms = []
    if num_context == 0:
        num_context = None
    for i in range(num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform( #randomly initialized by PyTorch number generator
            features=num_features,
            context_features=num_context,
            hidden_features=hidden_features,
            num_bins=num_bins,
            num_blocks=num_blocks,
            tail_bound=10.0,
            tails='linear',
            dropout_probability=0.2, 
            use_batch_norm=False 
        ))
        if i < num_layers - 1 and perm:
            transforms.append(ReversePermutation(features=num_features))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

def make_flow_zuko(num_layers, hidden_features, num_bins, num_blocks, 
                   num_features=2, num_context=0, bayesian=False):
    flow = zuko.flows.NSF(
        features=num_features,
        context=num_context,
        transforms=num_layers,
        hidden_features=[hidden_features] * num_blocks,
        bins=num_bins,
        bayesian=bayesian
    )
    return flow

def average_state_dicts(state_dicts):
    """Uniformly average a list of PyTorch state dicts"""
    n = len(state_dicts) #stores number of dictionaries, i.e. how many bootstrapped datasets you trained 
    avg_dict = {} #initialized an empty dictionary where we will store the averaged parameters 
    for key in state_dicts[0]: #loops through all parameter names in the first model (every model has the same parameter names so we can just take the keys from state_dict[0] to iterate over all params)
        avg_dict[key] = sum(sd[key] for sd in state_dicts) / n #for each key we sum across models and then average 
    return avg_dict

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
    model_probs_np = model_probs.cpu().clone().numpy()
    sigma_sq = np.einsum('ni,ij,nj->n', model_probs_np, cov_w, model_probs_np)
    sigma = np.sqrt(sigma_sq)
    return sigma

def plot_ensemble_marginals(f_i_models, x_data, weights, cov_w, feature_names, outdir):
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


def plot_individual_marginals(models, x_data, feature_names, outdir, num_samples, use_zuko=False):
    """
    For each model in `models`, plot its generated marginals vs the target data (x_data).

    Args:
        models: list of trained normalizing flow models
        x_data: torch.Tensor of shape (N, D), target data
        feature_names: list of strings like ["Feature 1", "Feature 2"]
        outdir: str, path to save plots
        num_samples: int, number of samples to draw from each model (default 1000)
    """
    os.makedirs(outdir, exist_ok=True)

    x_data_small = x_data[:num_samples].cpu().numpy()
    num_features = x_data_small.shape[1]

    for idx, model in enumerate(models):
        with torch.no_grad():
            if use_zuko:
                samples = model().sample((num_samples,)).cpu().numpy()
            else:
                samples = model.sample(num_samples).cpu().numpy()

        for i in range(num_features):
            target_feature = x_data_small[:, i]
            model_feature = samples[:, i]
            feature_label = feature_names[i]

            # Consistent binning
            bins = 40
            all_data = np.concatenate([model_feature, target_feature])
            low, high = np.min(all_data), np.max(all_data)
            bin_edges = np.linspace(low, high, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = np.diff(bin_edges)

            def hist_with_err(data):
                counts, _ = np.histogram(data, bins=bin_edges)
                N_total = np.sum(counts)
                hist = counts / (N_total * bin_widths)
                err = np.sqrt(counts) / (N_total * bin_widths)
                return hist, err

            hist_model, err_model = hist_with_err(model_feature)
            hist_target, err_target = hist_with_err(target_feature)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(bin_centers, hist_model, width=bin_widths, alpha=0.3,
                   label=f"Model {idx}", color='blue', edgecolor='black')
            ax.errorbar(bin_centers, hist_model, yerr=err_model, fmt='None', color='blue', alpha=0.7)

            ax.bar(bin_centers, hist_target, width=bin_widths, alpha=0.15,
                   label="Target", color='red', edgecolor='black')
            ax.errorbar(bin_centers, hist_target, yerr=err_target, fmt='None', color='red', alpha=0.7)

            ax.set_xlabel(feature_label, fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.legend(fontsize=12)

            plot_path = os.path.join(outdir, f"model_{idx}_feature_{i+1}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()