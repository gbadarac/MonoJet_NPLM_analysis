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


def plot_bayesian_marginals(flow, log_probs_list, x_data, feature_names, outdir):
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
        x_centers_tensor = torch.from_numpy(x_pad).float().to(x_data.device)

        # Run Bayesian forward passes again, now on bin centers
        with torch.no_grad():
            pred_logps = []
            for _ in range(len(log_probs_list)):  # match number of MC samples
                logps = flow().log_prob(x_centers_tensor)
                pred_logps.append(logps)

        pred_logps = torch.stack(pred_logps, dim=1)
        probs = pred_logps.exp()
        f_binned = probs.mean(dim=1).cpu().numpy()
        f_err = probs.std(dim=1).cpu().numpy()

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
        outpath = os.path.join(outdir, f"bayesian_marginal_feature_{i+1}.png")
        plt.savefig(outpath)
        plt.close()


