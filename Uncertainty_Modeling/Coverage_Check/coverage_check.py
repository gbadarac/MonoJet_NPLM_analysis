#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils import make_flow, probs 
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
# Optimize weights via logits using MLE
# ------------------
# Convert initial weights to torch tensor
w_i_init_torch = torch.tensor(w_i_initial, dtype=torch.float64, requires_grad=True)

def normalize_weights(weights):
    weights_sq = weights ** 2
    return weights_sq 

def ensemble_model(weights, model_probs):
    norm_weights = normalize_weights(weights)
    return probs(norm_weights, model_probs)

def constraint_term(weights):
    l=1.0
    return l*(torch.sum(normalize_weights(weights))-1.0)

def nll(weights):
    return -torch.log(ensemble_model(weights, model_probs) + 1e-8).mean() + constraint_term(weights)

res = minimize(
    nll,
    w_i_init_torch,
    method='newton-exact',
    options={'disp': True, 'max_iter': 300}
)

w_i_opt = res.x.detach()               # ← what the optimizer produced
w_i_final = normalize_weights(w_i_opt) # ← the true weights you use

# ------------------
# Compute uncertainties via Hessian
# ------------------
def compute_manual_hessian(w, model_probs, lam=1.0):
    N, M = model_probs.shape
    w = w.detach().clone().double()
    #print('w shape:', w.shape)
    model_probs = model_probs.double()

    f_i = model_probs  # (N, M)
    w_sq = w ** 2
    f = (f_i * w_sq).sum(dim=1)  # (N,)

    # Compute per-sample Hessian correctly
    # First term: 2 * δ_jk * f_j(x) / f
    term1 = torch.diag_embed(2 * f_i / (f[:, None] + 1e-12))  # (N, M, M)

    # Second term: 4 * w_j * f_j(x) * w_k * f_k(x) / f^2
    outer = (w[None, :, None] * f_i[:, :, None]) * (w[None, None, :] * f_i[:, None, :])  # (N, M, M)
    term2 = 4 * outer / (f[:, None, None] ** 2 + 1e-12)  # (N, M, M)

    # Combine both terms: H_data = average over data points
    H_data = (term1 - term2).mean(dim=0)  # (M, M)

    # Add constraint term: 2λ * I
    H_constraint = 2 * lam * torch.eye(M, dtype=torch.float64)

    return H_data + H_constraint 

H_manual = compute_manual_hessian(w_i_opt, model_probs)

# Convert H_manual to torch tensor
H_torch = H_manual.clone().detach()

# Autograd Hessian for cross-check
H_autograd = hessian(nll, w_i_opt.double()).detach()

eigvals = np.linalg.eigvalsh(H_autograd)

def compute_sandwich_covariance(H, w, model_probs, lam=1.0):
    M = len(w)
    N = model_probs.shape[0]

    w = w.detach().clone().double().requires_grad_()
    model_probs = model_probs.double()

    V = torch.linalg.solve(H, torch.eye(M, dtype=torch.float64))

    # Compute f(x) = sum_j w_j^2 f_j(x)
    w_sq = w ** 2
    f = (model_probs * w_sq).sum(dim=1)  # shape: (N,)

    grads = torch.zeros((N, M), dtype=torch.float64)  # ∂L/∂w_i(x)

    for j in range(M):
        f_j = model_probs[:, j]
        grads[:, j] = - (2 * w[j] * f_j) / (f + 1e-12) + 2 * lam * w[j]

    # Compute U matrix
    mean_grad = grads.mean(dim=0, keepdim=True)
    U = ((grads - mean_grad).T @ (grads - mean_grad)) / N

    cov_w = V @ U @ V.T
    cov_w = cov_w / N

    return cov_w

cov_w = compute_sandwich_covariance(H_autograd, w_i_opt, model_probs)

def jacobian_square_transform(w):
    """
    Jacobian of w_final = w^2 with respect to raw w
    """
    w = w.detach().clone().double()
    return torch.diag(2 * w)
# Jacobian transform of cov_w
J = jacobian_square_transform(w_i_opt)
#print('J=', J)

cov_w_final = J @ cov_w @ J.T
print("Weight covariance matrix:\n", cov_w_final)

cov_w_final = cov_w_final.detach().cpu()
w_i_final = w_i_final.detach().cpu()

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
n_bins=100

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
X1, X2 = torch.from_numpy(X1), torch.from_numpy(X2)
#flatten the grid to get a list of all 2D bin centers 
X_centers_grid = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)  # shape: (B², 2)=(40x40, 2)
print('X_centers_grid:', X_centers_grid.shape)

# Convert the X_grid (B², 2)=(40x40, 2) vector to PyTorch tensor and move to device
x_bin_centers_tensor = X_centers_grid.float().to(device)

# Evaluate all f_i(x_bin_center)
with torch.no_grad():
    model_probs_grid = torch.stack([
        torch.exp(flow.log_prob(x_bin_centers_tensor)) for flow in f_i_models
    ], dim=1).cpu() # shape: (B², M)

    print('model_probs_grid:', model_probs_grid.shape)

# Compute moment for each model:
#     μ_i = ∑_k x_k ⋅ f_i(x_k) ⋅ Δx

X_centers_grid = X_centers_grid.double()
model_probs_grid = model_probs_grid.double()
mu_i = torch.matmul(X_centers_grid.T, model_probs_grid) * bin_area  # shape: (D, M)'
print('mu_i:', mu_i.shape)
# Compute μ_model = ∑_i w_i ⋅ μ_i
# This is the ensemble-averaged moment: weighted sum of the μ_i using the optimized weights w_i
mu_model = torch.matmul(w_i_final[None, :], mu_i.T)  # shape: (D,)

# ------------------
# Compute uncertainty σ_I (propagation of weight covariance)
# ------------------
# We propagate the uncertainty on the weights through the functional:
#     σ²_I = ∇I(w)^T ⋅ Cov(w) ⋅ ∇I(w)
# where ∇I(w) ≡ [∂I/∂w_i] = μ_i (since I(w) = ∑ w_i μ_i)

# Perform the matrix contraction:
#   σ²_I[d] = ∑_i ∑_j μ_i[d] ⋅ Cov_w[i, j] ⋅ μ_j[d]
# for each feature dimension d
sigma_sq = torch.einsum("di,ij,dj->d", mu_i, cov_w_final, mu_i)  # shape: (D,)
print('sigma_sq', sigma_sq)

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
    "weights": w_i_final.tolist(),
    "sum weights": w_i_final.sum().tolist()
}

out_file = os.path.join(args.out_dir, f"toy_{args.toy_seed:03d}.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
