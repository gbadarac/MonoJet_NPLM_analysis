#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils import make_flow, log_likelihood
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # ensure it works on clusters without displaying plots
from torch.autograd.functional import hessian
import mplhep as hep
from torchmin import minimize
import torch.nn.functional as F

# Use CMS style for plots
hep.style.use("CMS")

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mu_target_path", type=str, required=True)
parser.add_argument("--toy_seed", type=int, required=True)
parser.add_argument("--trial_dir", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
args = parser.parse_args()

# ------------------
# Load models and initial weights 
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets device to GPU if available 
os.makedirs(args.trial_dir, exist_ok=True)

#f_i_file = os.path.join(args.trial_dir, "f_i_averaged.pth")
f_i_file = os.path.join(args.trial_dir, "f_i_averaged.pth")
w_i_file = os.path.join(args.trial_dir, "w_i_initial.npy")

f_i_statedicts = torch.load(f_i_file, map_location=device) #list of state_dicts 
w_i_initial = np.load(w_i_file)  # Initial weights
print(f"Initial weights: {w_i_initial}")

# ------------------
# Load architecture config and data 
# ------------------
# Save architecture 
#'''
with open(os.path.join(args.trial_dir, "architecture_config.json")) as f:
    config = json.load(f) 
'''
configs = []
for model_name in ["good_model", "bad_model"]:
    config_path = os.path.join(args.trial_dir, model_name, "architecture_config.json")
    with open(config_path) as f:
        configs.append(json.load(f))
'''

all_data = np.load(args.data_path)
np.random.seed(args.toy_seed)
indices = np.random.choice(len(all_data), size=len(all_data), replace=True)
x_data = torch.from_numpy(all_data[indices]).float().to(device)

# ------------------
# Reconstruct flows f_i
# ------------------
f_i_models = []
for state_dict in f_i_statedicts:
#for state_dict, config in zip(f_i_statedicts, configs):
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_bins=config["num_bins"],
        num_blocks=config["num_blocks"],
        tail_bound=config["tail_bound"]
    ).to(device) #recreate the flow architecture for each model f_i 
    flow.load_state_dict(state_dict) #load the corresponding params into each model 
    flow.eval() # Set to eval mode to avoid training-time behavior (e.g., dropout)
    #You want all models to produce consistent, deterministic outputs for likelihood evaluation 
    f_i_models.append(flow)

# ------------------
# Evaluate f_i(x) -> model_probs
# ------------------
with torch.no_grad(): #disables gradient tracking since we are just evaluating the model (to speed things up)
    model_probs = torch.stack( #torch.stack() to stack all M models along a new dimension dim=1 cretaing a NxM matrix where N are the number of data points 
        [torch.exp(flow.log_prob(x_data)) for flow in f_i_models], 
        #each model f_i comes with a log probability, so for getting the actual pdf of the model itself one needs to take the exponential 
        #this needs to be done because of how NFs are built, i.e. they return log(f_i(x)) but we want to get f_i(x) to get the probability, so we need to take the exponential 
        #this is standsard in NFs because calculating log(f_i(x)) is numerically stable and avoids underflow 
        dim=1)  # Shape: (N, M)

# Move model_probs to CPU for memory safety
# model robsa is a (N,M) matrix of type (f_0, ... f_(M-1)) where N is the number of data points 
model_probs = model_probs.to("cpu")

# ------------------
# Optimize weights using MLE
# ------------------
# Instead of optimizing directly over the weights w_i (which must satisfy w_i ≥ 0 and ∑w_i = 1),
# we optimize over unconstrained logits z_i ∈ ℝ, and then map them to the simplex using softmax.
# This allows unconstrained optimization in logit space while ensuring the weights always stay valid.

# Initialize logits such that softmax(logits) ≈ initial weights w_i_initial
w_i_logits = torch.nn.Parameter(torch.log(torch.tensor(w_i_initial + 1e-8, dtype=torch.float32)))

# Optimizer acts on the logits z_i. After each update, we’ll map z_i to valid weights via softmax.
optimizer = optim.Adam([w_i_logits], lr=1e-2)

# Map logits to normalized weights using the softmax function:
#     w_i = exp(z_i) / ∑_j exp(z_j)
# This ensures the weights remain in [0,1] and sum to 1 at every step
def norm_weights(logits):
    return torch.nn.functional.softmax(logits, dim=0)

max_steps = 3000
patience = 100
min_delta_window = 5e-7
min_delta_global = 1e-5  # Minimum global improvement over best_loss

best_loss = float("inf")
nll_window = []
no_improve_counter = 0

nll_vals = []
weight_vals = []

for step in range(max_steps):
    optimizer.zero_grad()
    w_i_norm = norm_weights(w_i_logits)
    l=1e-3
    entropy = -torch.sum(w_i_norm * torch.log(w_i_norm + 1e-8))
    loss = -log_likelihood(w_i_norm, model_probs) + l * entropy
    #loss = -log_likelihood(w_i_norm, model_probs)
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    if step % 25 == 0 or step == max_steps - 1:
        w = w_i_norm.detach().cpu().numpy()
        nll_vals.append(loss_val)
        weight_vals.append(w.copy())

    # Update global best
    if loss_val + min_delta_global < best_loss:
        best_loss = loss_val
        no_improve_counter = 0
    else:
        no_improve_counter += 1

    # Update window
    nll_window.append(loss_val)
    if len(nll_window) > patience:
        nll_window.pop(0)

    # Check window-based plateau
    if len(nll_window) == patience:
        if max(nll_window) - min(nll_window) < min_delta_window:
            print(f"\nEarly stopping at step {step} (NLL plateaued: Δ_window < {min_delta_window})")
            break

    # Optional safety: stop if no global improvement for a while
    if no_improve_counter > patience * 2:
        print(f"\nEarly stopping at step {step} (No global NLL improvement in {no_improve_counter} steps)")
        break
w_i_final = norm_weights(w_i_logits).detach().cpu().numpy()
print("\nFinal fitted weights (PyTorch):", w_i_final)

effective_models = 1.0 / np.sum(w_i_final**2)
print("Effective number of models:", effective_models)

# ------------------
# Compute uncertainties via Hessian
# ------------------
#define negative log likelihood as a function of the logits (trainable param)
def loss_function(logits):
    weights = torch.nn.functional.softmax(logits, dim=0)
    return log_likelihood(weights, model_probs) #return log-likelihood

#compute hessian wrt logits (trainable param)
H_z = hessian(loss_function, w_i_logits.detach().requires_grad_()).detach().cpu().numpy()
# compute normalized Jacobian: J_ij = dw_i/dz_j = w_i * (δ_ij - w_j)
J = np.diag(w_i_final) - np.outer(w_i_final, w_i_final)
# Covariance propagation: Cov_w = J H^{-1} J^T
H_z = -H_z

# Regularize Hessian
epsilon = 1e-2
H_z_reg = H_z + epsilon * np.eye(H_z.shape[0])

# Invert and propagate
H_z_inv = np.linalg.pinv(H_z_reg)
cov_w = J @ H_z_inv @ J.T
print("Weight covariance matrix:\n", cov_w)

# Upload target's first moment:
# Load target first moment
mu_target = np.load(args.mu_target_path)  # shape: (D,)

# ------------------
# Compute model's first moment μ_model
# ------------------
#Goal: compute mu_i = ∫dx*x*f_i(x) ~ sum(x_k * f_i(x_k) * ∆x)
#Define bin grid over the 2D domanin of data 
x_np = x_data.cpu().numpy()                     # shape: (N, D
x_min= x_np.min(axis=0)
x_max= x_np.max(axis=0)
n_bins=40

#Create edges and centers for both fetaures 
edges_1 = np.linspace(x_min[0], x_max[0], n_bins+1) #bin edges for feature 1 
edges_2 = np.linspace(x_min[1], x_max[1], n_bins+1) #bin edges for feature 2 
dx1 = np.diff(edges_1)[0] #width of bins along feature 1 axis 
dx2 = np.diff(edges_2)[0] #width bin along fetaure 2 axis 
bin_area = dx1 * dx2 #area of each rectangular bin 

centers_1 = 0.5 * (edges_1[:-1] + edges_1[1:]) #bin centers for feature 1 
centers_2 = 0.5 * (edges_2[:-1] + edges_2[1:]) #bin centers for fetaure 2 

# Build 2D meshgrid of bin centers
X1, X2 = np.meshgrid(centers_1, centers_2) 
#flatten the grid to get a list of all 2D bin centers 
X_centers_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)  # shape: (B², 2)=(40x40, 2)

# Convert the X_grid (B², 2)=(40x40, 2) vector to PyTorch tensor and move to device
x_bin_centers_tensor = torch.from_numpy(X_centers_grid).float().to(device)

# Evaluate all f_i(x_bin_center)
with torch.no_grad():
    model_probs_grid = torch.stack([
        torch.exp(flow.log_prob(x_bin_centers_tensor)) for flow in f_i_models
    ], dim=1).cpu().numpy()  # shape: (B², M)

# Compute moment for each model:
#     μ_i = ∑_k x_k ⋅ f_i(x_k) ⋅ Δx
mu_i = (X_centers_grid[:, None, :] * model_probs_grid[:, :, None]).sum(axis=0) * bin_area  # shape: (M, D)'
print("mu_i[:, 0]=", mu_i[:,0])

# Compute μ_model = ∑_i w_i ⋅ μ_i
# This is the ensemble-averaged moment: weighted sum of the μ_i using the optimized weights w_i
mu_model = np.sum(w_i_final[:, None] * mu_i, axis=0)  # shape: (D,)

# ------------------
# Compute uncertainty σ_I (propagation of weight covariance)
# ------------------
# We propagate the uncertainty on the weights through the functional:
#     σ²_I = ∇I(w)^T ⋅ Cov(w) ⋅ ∇I(w)
# where ∇I(w) ≡ [∂I/∂w_i] = μ_i (since I(w) = ∑ w_i μ_i)

# Perform the matrix contraction:
#   σ²_I[d] = ∑_i ∑_j μ_i[d] ⋅ Cov_w[i, j] ⋅ μ_j[d]
# for each feature dimension d
sigma_sq = np.einsum("id,ij,jd->d", mu_i, cov_w, mu_i)  # shape: (D,)

# Take square root to get standard deviation (1σ uncertainty band)
sigma_I = np.sqrt(sigma_sq)

# Check if |μ_model - μ_target| < σ_I (per feature) and save boolean 
diff = np.abs(mu_model - mu_target)
within_band = diff < sigma_I  # boolean vector

# ------------------
# Save results for post-processing
# ------------------
# Convert NumPy arrays to lists for JSON serialization
results = {
    "mu_target": mu_target.tolist(),
    "mu_model": mu_model.tolist(),
    "diff": diff.tolist(),
    "sigma_integral": sigma_I.tolist(),
    "within_band": within_band.tolist(),
    "toy_seed": int(args.toy_seed),
    "weights": w_i_final.tolist()
}

out_file = os.path.join(args.out_dir, f"toy_{args.toy_seed:03d}.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
