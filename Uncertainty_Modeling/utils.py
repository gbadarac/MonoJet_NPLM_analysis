import numpy as np
import torch
import torch.nn.functional as F
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import matplotlib
import matplotlib.pyplot as plt
import os

import mplhep as hep

# Use CMS style for plots
hep.style.use("CMS")


def make_flow(num_layers, hidden_features, num_bins, num_blocks, tail_bound, num_features=2, num_context=None, perm=True):
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
            tail_bound=tail_bound,
            tails='linear',
            dropout_probability=0.2,
            use_batch_norm=False
        ))
        if i < num_layers - 1 and perm:
            transforms.append(ReversePermutation(features=num_features))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
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

def plot_marginals(dist1, dist2, target, feature_names, outdir, labels):
    target = target.cpu().numpy()
    num_features = target.shape[1]
    
    for i in range(num_features):
        fig, ax_main = plt.subplots(1, 1, figsize=(8, 6))           
        dist1_feature = dist1[:, i]
        dist2_feature = dist2[:, i]
        target_feature=target[:,i]
        feature_label = feature_names[i]
        
        # Combine data for consistent binning
        bins = 40
        all_data = np.concatenate([dist1_feature, dist2_feature, target_feature])
        # Use full support for binning to avoid normalization loss
        low, high = np.min(all_data), np.max(all_data)
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
    
        # Histograms (shared binning)
        def hist_with_err(data):
            counts, _ = np.histogram(data, bins=bin_edges)
            N_total = np.sum(counts)
            hist = counts / (N_total * bin_widths)
            err = np.sqrt(counts) / (N_total * bin_widths)      
            return hist, err
        
        hist_dist1, err_dist1 = hist_with_err(dist1_feature)
        hist_dist2, err_dist2 = hist_with_err(dist2_feature)
        hist_target, err_target = hist_with_err(target_feature)
        
        # Main plot
        ax_main.bar(bin_centers, hist_dist1, width=np.diff(bin_edges), alpha=0.3, label=labels[0], color='blue', edgecolor='black')
        ax_main.bar(bin_centers, hist_dist2, width=np.diff(bin_edges), alpha=0.3, label=labels[1], color='green', edgecolor='black')
        ax_main.bar(bin_centers, hist_target, width=np.diff(bin_edges), alpha=0.15, label='Target', color='red', edgecolor='black')
        ax_main.errorbar(bin_centers, hist_dist1, yerr=err_dist1, fmt='None', color='blue', alpha=0.7)
        ax_main.errorbar(bin_centers, hist_dist2, yerr=err_dist2, fmt='None', color='green', alpha=0.7)
        ax_main.errorbar(bin_centers, hist_target, yerr=err_target, fmt='None', color='red', alpha=0.7)

        ax_main.set_xlabel(feature_label, fontsize=16)
        ax_main.set_ylabel("Density", fontsize=16)
        ax_main.legend(fontsize=14)

        # Save plot
        plot_path = os.path.join(outdir, f"marginal_feature_{i+1}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()
        plt.close()

def log_likelihood(weights, model_probs):
    """
    Args:
        weights: torch tensor with requires_grad=True
        model_probs: torch.tensor of shape (N, M) with f_i(x)
    Returns:
        Scalar negative log-likelihood
    """
    p_x = probs(weights, model_probs) + 1e-8  # prevent log(0)
    ll = torch.log(p_x).mean() #averaging the loss over all datapoints 
    return ll # scalar

def ensemble_pred(weights, model_probs):
    """
    Compute ensemble prediction f(x) = sum_i w_i f_i(x)
    """
    model_vals=(model_probs * torch.tensor(weights, device=model_probs.device)).sum(dim=1)
    return model_vals.cpu().numpy()

def ensemble_unc(cov_w, model_probs):
    model_probs_np = model_probs.numpy()  # shape: (N, M)
    sigma_sq = np.einsum('ni,ij,nj->n', model_probs_np, cov_w, model_probs_np)
    return np.sqrt(sigma_sq)

def plot_ensemble_marginals(model_probs, x_data, weights, cov_w, feature_names, outdir):
    #convet pytorch input tensor x_data to a NumPy array for easier processing 
    x = x_data.cpu().numpy() 
    model_probs_np = model_probs.numpy()

    for i in range(model_probs_np.shape[1]):
        std_feat1 = np.std(model_probs_np[:, i][x[:, 0].argsort()])
        std_feat2 = np.std(model_probs_np[:, i][x[:, 1].argsort()])
        print(f"Model {i} std along feature 1: {std_feat1:.4e}")
        print(f"Model {i} std along feature 2: {std_feat2:.4e}")

    # Compute ensemble prediction and uncertainty at each point
    f_vals = ensemble_pred(weights, model_probs)
    f_unc = ensemble_unc(cov_w, model_probs)
       
    print(f"[Feature {i+1}] Uncertainty stats:")
    print(f"  f_unc min: {f_unc.min():.4e}, max: {f_unc.max():.4e}, mean: {f_unc.mean():.4e}, std: {f_unc.std():.4e}")

    # Loop over each marginal feature
    num_features = x.shape[1]
    for i in range(num_features):
        fig, ax_main = plt.subplots(figsize=(8, 6))
        feature_label = feature_names[i]
        input_feature = x[:,i]

        bins = 40

        # Bin over full data range with margin
        margin = 0.05 * (np.max(input_feature) - np.min(input_feature))
        low = np.min(input_feature) - margin
        high = np.max(input_feature) + margin
        bin_edges = np.linspace(low, high, bins + 1)

        #define bin edges, centers an assign each point to a bin 
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        bin_indices = np.digitize(input_feature, bin_edges)

        # Histogram of the target data
        hist_target_counts, _ = np.histogram(input_feature, bins=bin_edges)
        N_target = np.sum(hist_target_counts)
        hist_target = hist_target_counts / (N_target * bin_widths)
        err_target = np.sqrt(hist_target_counts) / (N_target * bin_widths)

        # Average ensemble values per bin
        f_binned, f_err = [], []
        for b in range(1, len(bin_edges)):
            idx = (bin_indices == b)
            if np.sum(idx) > 0:
                f_binned.append(np.mean(f_vals[idx]))
                bin_unc = f_unc[idx]
                f_err.append(np.sqrt(np.sum(bin_unc ** 2))/ len(bin_unc))
            else:
                f_binned.append(0)
                f_err.append(0)

        N = np.sum(f_binned) * bin_widths
        f_binned = np.array(f_binned)/N
        f_err = np.array(f_err)/N

        print(f"f_binned Feature {i+1}: {f_binned}")
        print(f"f_err Feature {i+1}: {f_err}")

        # ------------------
        # 1 and 2 sigma bands calculation 
        # ------------------

        # Ensemble marginal: mean and uncertainty bands per bin
        band_mean, band_1s_l, band_1s_h = [], [], []
        band_2s_l, band_2s_h = [], []

        # for each bin: compute average model prediction and uncertainty band percentiles.
        for b in range(1, len(bin_edges)):
            idx = (bin_indices == b)
            if np.sum(idx) == 0:
                band_mean.append(0)
                band_1s_l.append(0)
                band_1s_h.append(0)
                band_2s_l.append(0)
                band_2s_h.append(0)
                continue
            f_bin = f_vals[idx]
            s_bin = f_unc[idx]
           
            mu = np.mean(f_bin)
            sigma = np.sqrt(np.mean(s_bin**2))  # RMS average uncertainty
            band_1s_l.append(mu - sigma)
            band_1s_h.append(mu + sigma)
            band_2s_l.append(mu - 2*sigma)
            band_2s_h.append(mu + 2*sigma)

        # Normalize the bands the same way as f_binned
        band_1s_l = np.array(band_1s_l) / N
        band_1s_h = np.array(band_1s_h) / N
        band_2s_l = np.array(band_2s_l) / N
        band_2s_h = np.array(band_2s_h) / N

        # Mask to keep only bins with target data
        valid_bins = hist_target > 0

        # Plot main distribution
        ax_main.bar(bin_centers, hist_target, width=np.diff(bin_edges), alpha=0.2, label="Target", color='green', edgecolor='black')
        ax_main.errorbar(bin_centers, hist_target, yerr=err_target, fmt='None', color='green', alpha=0.7)
        ax_main.errorbar(bin_centers[valid_bins], f_binned[valid_bins], yerr=f_err[valid_bins], fmt='-', color='red', linewidth=1.2, capsize=0, marker=None, label=r"$f(x) = \sum w_i f_i(x)$")
        ax_main.fill_between(bin_centers, band_1s_l, band_1s_h, alpha=0.12, label=r"$\pm 1\sigma$", color='purple')
        ax_main.fill_between(bin_centers, band_2s_l, band_2s_h, alpha=0.07, label=r"$\pm 2\sigma$", color='purple')

        ax_main.set_xlabel(feature_label, fontsize=16)
        ax_main.set_ylabel("Density", fontsize=16)
        ax_main.legend(fontsize=14)

        # Save plot
        plt.tight_layout()
        outpath = os.path.join(outdir, f"ensemble_marginal_feature_{i+1}.png")
        plt.savefig(outpath)
        plt.close()