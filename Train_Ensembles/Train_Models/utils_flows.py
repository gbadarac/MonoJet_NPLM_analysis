import numpy as np
import torch
import torch.nn.functional as F
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
import sys
sys.path.insert(0, "/work/gbadarac/zuko")
import zuko

import matplotlib
import matplotlib.pyplot as plt
import os

import mplhep as hep

# Use CMS style for plots
hep.style.use("CMS")

def make_flow(num_layers, hidden_features, num_bins, num_blocks, num_features, num_context=None, perm=True):
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

    idx = torch.randperm(x_data.shape[0], device=x_data.device)[:num_samples]
    x_data_small = x_data[idx].cpu().numpy()
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
            bins = 80
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