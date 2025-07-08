import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append("/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling")

from utils import make_flow, ensemble_pred  # must be in your Python path or notebook directory
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
trial_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Fit_Weights/results_no_norm_fit_weights_NF/N_100000_seeds_60_bootstraps_1_4_16_128_15"
data_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Generate_Data/saved_generated_target_data/100k_target_training_set.npy"
out_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data"
os.makedirs(out_dir, exist_ok=True)

# Reference architecture (consistent across models)
arch_config_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/EstimationNF_gaussians_outputs/N_100000_seeds_60_bootstraps_1_4_16_128_15/architecture_config.json"

# Load model weights and configs
with open(arch_config_path) as f:
    config = json.load(f)

f_i_statedicts = torch.load(os.path.join(trial_dir, "f_i_averaged.pth"), map_location=device)
#w_i_fitted = torch.tensor(np.load(os.path.join(trial_dir, "w_i_fitted.npy")), dtype=torch.float64)
w_i_fitted = torch.tensor([ 0.0489, -0.0549,  0.0217,  0.0400,  0.0104,  0.0634,  0.0098,  0.0756,
         0.0213,  0.0335,  0.0505, -0.0053,  0.0017,  0.0108,  0.0127,  0.0066,
         0.0315,  0.0264,  0.0262,  0.1018,  0.0717,  0.0223,  0.0015, -0.0130,
         0.0351,  0.0155,  0.0271,  0.0563,  0.0065,  0.0513,  0.0528, -0.0659,
         0.0601, -0.0258, -0.0198, -0.0387,  0.0280, -0.0160,  0.0431, -0.0626,
         0.0246,  0.0034, -0.0270,  0.0072, -0.0303,  0.0098, -0.0195, -0.0674,
         0.0384,  0.0571,  0.0251, -0.0116,  0.0485,  0.0325,  0.0231,  0.0251,
         0.0107, -0.0070,  0.0596,  0.0358], dtype=torch.float64)

# Load data
x_data = torch.from_numpy(np.load(data_path)).float().to(device)
print("Data shape:", x_data.shape)
print("Loaded", len(f_i_statedicts), "models")

f_i_models = []
for state_dict in f_i_statedicts:
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_bins=config["num_bins"],
        num_blocks=config["num_blocks"],
        tail_bound=config["tail_bound"]
    ).to(device)
    flow.load_state_dict(state_dict)
    flow.eval()
    f_i_models.append(flow)


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
    total_hits =0
    while total_hits < N_events and total_attempts < max_attempts:
        # Sample uniformly in 2D space
        x1_samples = torch.empty(batch_size, device=device).uniform_(x1_min, x1_max)
        x2_samples = torch.empty(batch_size, device=device).uniform_(x2_min, x2_max)
        y_samples = torch.empty(batch_size, device=device).uniform_(0, f_max)

        with torch.no_grad():
            f_values = f(x1_samples, x2_samples, f_i_models, w_i_fitted)

        mask = y_samples < f_values
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