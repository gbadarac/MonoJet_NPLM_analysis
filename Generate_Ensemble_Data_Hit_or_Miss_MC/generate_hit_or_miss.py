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
trial_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15_trial"
data_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_target_training_set.npy"
subdir = f"N_100000_dim_2_seeds_60_4_16_128_15"
out_dir = os.path.join(
    "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data",
    subdir,
)
os.makedirs(out_dir, exist_ok=True)

# Reference architecture (consistent across models)
arch_config_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/2_dim/N_100000_dim_2_seeds_60_4_16_128_15/architecture_config.json"

# Load model weights and configs
with open(arch_config_path) as f:
    config = json.load(f)

f_i_statedicts = torch.load(os.path.join(trial_dir, "f_i.pth"), map_location=device)
#w_i_fitted = torch.tensor(np.load(os.path.join(trial_dir, "w_i_fitted.npy")), dtype=torch.float64)
w_i_fitted = torch.tensor([ 2.7465e-02,  3.9775e-02,  2.9415e-02,  6.0815e-02,  1.6158e-02,
         8.9290e-02,  5.7797e-03, -1.1812e-02,  2.4000e-02,  3.1447e-02,
         8.1284e-02, -2.3717e-02,  2.6356e-02, -1.7371e-02, -1.0963e-02,
         8.3692e-03, -8.1379e-03, -2.3745e-02,  2.1039e-02, -2.9520e-04,
        -4.8599e-02,  6.1569e-02, -4.8001e-02, -3.0407e-02, -1.4790e-02,
        -2.3764e-02,  5.1739e-02,  3.5346e-02,  8.6801e-03,  4.0055e-02,
        -8.1822e-03, -7.8225e-02,  5.8031e-02,  5.0157e-02, -5.7196e-03,
         1.7794e-02,  6.6487e-02,  1.7647e-02, -1.0253e-02,  1.3936e-02,
        -4.6986e-02,  3.5879e-03,  3.0478e-02,  9.8500e-02,  1.6057e-02,
        -4.4614e-03,  5.2503e-03, -4.2855e-03,  2.5735e-02,  1.7025e-02,
         7.2025e-02,  1.9940e-02, -1.0673e-02,  2.1200e-02,  9.2340e-02,
        -3.0226e-02,  4.8697e-02,  7.0400e-05,  7.8944e-02,  4.8129e-02],
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
        num_features=config["num_features"],
    )
    flow.load_state_dict(state_dict)
    flow.eval()
    f_i_models.append(flow)
    del flow
    gc.collect()


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
np.save(os.path.join(out_dir, "ensemble_generated_samples_4_16_128_15_seed_%i.npy"%(seed)), samples.cpu().numpy())
print("Saved generated samples.")