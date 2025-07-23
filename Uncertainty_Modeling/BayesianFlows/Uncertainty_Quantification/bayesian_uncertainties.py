#!/usr/bin/env python
# coding: utf-8

import os, json, argparse
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import sys
sys.path.insert(0, "/work/gbadarac/zuko")
import zuko
print(zuko.__file__)
from zuko.utils import total_KL_divergence
from utils_flows import make_flow_zuko
from utils_BayesianFlows import plot_bayesian_marginals
import mplhep as hep
import matplotlib.pyplot as plt


# Use CMS style for plots
hep.style.use("CMS")

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--trial_dir", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--plot", type=str, required=True)
args = parser.parse_args()

# ------------------
# Load models and initial weights 
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets device to GPU if available 
os.makedirs(args.trial_dir, exist_ok=True)

# ------------------
# Load architecture config and data 
# ------------------
# Save architecture 

with open(os.path.join(args.trial_dir, "architecture_config.json")) as f:
    config = json.load(f) 

flow = make_flow_zuko(num_layers=config["num_layers"],hidden_features=config["hidden_features"], num_bins=config["num_bins"], num_blocks=config["num_blocks"], num_features=2, num_context=0, bayesian=config["bayesian"]).to(device)

# Loads the learned weights of the NF model into the flow model instance 
flow.load_state_dict(torch.load(os.path.join(args.trial_dir, "model_000", "model.pth"), map_location=device))
# Puts the model into evaluation mode
# This ensures the model behaves deterministically and xconsistently during inference, matching its behavior during validation/testing, not training 
flow.eval()

x_data = torch.from_numpy(np.load(args.data_path)).float().to(device)

log_probs_list = []
with torch.no_grad():
    for i in range(100):
        sampled_flow = flow()
        log_probs_list.append(sampled_flow.log_prob(x_data))

D = torch.stack(log_probs_list, dim=1)
P = D.exp()
P_mean = P.mean(dim=1)
P_std = P.std(dim=1)

np.save(os.path.join(args.out_dir, "log_prob_mean.npy"), P_mean.cpu().numpy())
np.save(os.path.join(args.out_dir, "density_std.npy"), P_std.cpu().numpy())

# ------------------
# Debug prints
# ------------------
print("DEBUG: Shape of P_mean:", P_mean.shape)
print("DEBUG: Shape of P_std:", P_std.shape)
print("DEBUG: Sample uncertainties (P_std):", P_std[:10].cpu().numpy())
print("DEBUG: Mean uncertainty:", P_std.mean().item())
print("DEBUG: Max uncertainty:", P_std.max().item())

# ------------------
# Plotting 
# ------------------
if args.plot.lower() == "true":
    feature_names = ["Feature 1", "Feature 2"]
    plot_bayesian_marginals(flow, log_probs_list, x_data, feature_names, args.out_dir)

    plt.figure(figsize=(6,4))
    plt.hist(P_std.cpu().numpy(), bins=50, color='purple', alpha=0.7)
    plt.xlabel("Predicted std (uncertainty)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Uncertainties (P_std)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "uncertainty_distribution.png"))
    plt.close()