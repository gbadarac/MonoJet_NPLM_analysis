import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc 

import sys
sys.path.append("/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles")

from utils_flows import make_flow
# must be in your Python path or notebook directory
# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

seed=args.seed

np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set paths
trial_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_seeds_60_4_16_256_15_l_1e5"
data_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/100k_target_training_set.npy"
out_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data"
os.makedirs(out_dir, exist_ok=True)

# Reference architecture (consistent across models)
arch_config_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/N_100000_seeds_60_4_16_256_15/architecture_config.json"

# Load model weights and configs
with open(arch_config_path) as f:
    config = json.load(f)

f_i_statedicts = torch.load(os.path.join(trial_dir, "f_i.pth"), map_location=device)
#w_i_fitted = torch.tensor(np.load(os.path.join(trial_dir, "w_i_fitted.npy")), dtype=torch.float64)
w_i_fitted = torch.tensor([ 2.6661e-02,  4.9029e-03,  5.4670e-05,  1.9149e-02,  6.4070e-02,
         9.3286e-03,  3.6494e-02,  8.7958e-02,  7.6974e-02, -1.5255e-02,
        -3.9479e-02, -1.5420e-02, -1.5285e-02,  8.7045e-02,  5.7291e-02,
         1.5961e-02, -3.6252e-02, -4.3677e-02, -1.4824e-04, -3.4309e-02,
        -2.3378e-02,  3.6204e-02,  3.1594e-02,  8.5678e-03,  1.2468e-02,
         2.5487e-02,  1.9723e-02,  2.5634e-02,  6.8171e-02, -9.8194e-03,
         8.0850e-03,  2.1400e-02,  3.2348e-02,  6.7841e-02,  4.6322e-02,
         4.8491e-02, -3.6525e-02, -1.7060e-02,  4.7686e-02,  1.4838e-02,
         3.4592e-02, -2.2595e-02, -6.4887e-02,  1.0992e-01, -3.1935e-03,
        -7.6125e-02, -2.0111e-02,  4.2485e-02,  3.6925e-02, -1.5192e-02,
         4.6197e-02,  4.1148e-02, -9.9237e-03,  4.3357e-02,  3.0554e-02,
        -1.3172e-02, -2.7091e-02,  6.0366e-02,  2.3911e-02,  6.8697e-02],
       dtype=torch.float64)

# Load data
x_data = torch.from_numpy(np.load(data_path)).float()
print("Data shape:", x_data.shape)
print("Loaded", len(f_i_statedicts), "models")

f_i_models = []
for state_dict in f_i_statedicts:
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_bins=config["num_bins"],
        num_blocks=config["num_blocks"],
    ).to(device)
    flow.load_state_dict(state_dict)
    flow.eval()
    f_i_models.append(flow)
    del flow
    gc.collect()

'''
# Define ensemble callable f(x1, x2) using the fitted weights and flow models
def f(x1, x2, f_i_models, w_i_fitted):
    """
    Compute ensemble density f(x1, x2) = sum_i w_i * f_i(x1, x2)
    Input: x1, x2 of shape (N,) — output: (N,)
    """
    x = torch.stack([x1, x2], dim=1)  # shape (N, 2)
    with torch.no_grad():
        probs = torch.stack([torch.exp(model.log_prob(x)) for model in f_i_models], dim=1)  # (N, M)
        w = w_i_fitted.to(probs.device)
        return (probs * w).sum(dim=1)  # shape (N,)
'''

def f(x1, x2, f_i_models, w_i_fitted):
    """
    Compute ensemble density f(x1, x2) = sum_i w_i * f_i(x1, x2)
    This version avoids keeping all models on GPU at once.
    """
    x = torch.stack([x1, x2], dim=1)
    probs_list = []
    with torch.no_grad():
        for model in f_i_models:
            batch_size = 5000
            model_probs = []
            model = model.to(x.device)  # move model once per loop
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i+batch_size].to(x.device)  # ← FIXED: move data to same device
                logp_batch = model.log_prob(x_batch)
                model_probs.append(torch.exp(logp_batch))
            model_probs_tensor = torch.cat(model_probs, dim=0).detach()
            probs_list.append(model_probs_tensor)
            model = model.to("cpu")  # Free GPU memory
            torch.cuda.empty_cache()
    probs = torch.stack(probs_list, dim=1)
    w = w_i_fitted.to(probs.device)
    return (probs * w).sum(dim=1)

def hit_or_miss_2d(x1_min, x1_max, x2_min, x2_max, f_i_models, w_i_fitted, N_events, max_attempts=1000000000):
    """
    Monte Carlo sampling via hit-or-miss in 2D using PyTorch.

    Args:
        x1_min, x1_max: float, bounds for x-axis
        x2_min, x2_max: float, bounds for y-axis
        f: callable, probability density function f(x1, x2) -> torch.tensor (same shape as x1/x2)
        N_events: int, number of samples to generate
        max_attempts: int, maximum number of sampling attempts (to avoid infinite loop)

    Returns:
        samples: torch.Tensor of shape (N_events, 2), the accepted (x1, x2) points
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Estimate f_max by grid sampling
    '''
    x1_grid = torch.linspace(x1_min, x1_max, 400, device=device)
    x2_grid = torch.linspace(x2_min, x2_max, 400, device=device)
    X1, X2 = torch.meshgrid(x1_grid, x2_grid, indexing='ij')
    '''
    with torch.no_grad():
        #f_vals = f(X1.flatten(), X2.flatten(), f_i_models, w_i_fitted)
        #f_max = f_vals.max().item() * 1.1  # Safety factor
        f_max=2
        print('f_max', f_max)

    accepted = []
    total_attempts = 0
    batch_size = int(0.5 * N_events)  # Oversample in batches for efficiency
    total_hits = 0
    while total_hits < N_events and total_attempts < max_attempts:
        # Sample uniformly in 2D space
        x1_samples = torch.empty(batch_size, device=device).uniform_(x1_min, x1_max)
        x2_samples = torch.empty(batch_size, device=device).uniform_(x2_min, x2_max)
        y_samples = torch.empty(batch_size, device=device).uniform_(0, f_max)

        with torch.no_grad():
            f_values = f(x1_samples, x2_samples, f_i_models, w_i_fitted)
            # Clamp negative densities to zero for sampling
            valid_mask = f_values > 0
            f_values_clamped = torch.where(valid_mask, f_values, torch.tensor(0.0, dtype=f_values.dtype, device=device))

        mask = y_samples < f_values_clamped
        hits = torch.stack([x1_samples[mask], x2_samples[mask]], dim=1)
        total_hits +=hits.shape[0]
        print('total hits', total_hits)
        accepted.append(hits)
        total_attempts += batch_size
        print('len(accepted)', len(accepted))

    if accepted:
        samples = torch.cat(accepted, dim=0)[:N_events]

    else:
        raise RuntimeError("No events accepted; check if your function f is correctly normalized and non-zero.")
    print('sampl size', samples.shape)
    return samples

# Sample new events from the ensemble distribution
#tb = config["tail_bound"]
tb = 2.8

samples = hit_or_miss_2d(-tb, tb, -tb, tb, f_i_models, w_i_fitted, N_events=5000) 
np.save(os.path.join(out_dir, "ensemble_generated_samples_4_16_256_15_seed_%i.npy"%(seed)), samples.cpu().numpy())
print("Saved generated samples.")