#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils_wifi import probs
from utils_flows import make_flow
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # ensure it works on clusters without displaying plots
from torch.autograd.functional import hessian
import mplhep as hep
from torchmin import minimize
import torch.nn.functional as F
import traceback

# Use CMS style for plots
hep.style.use("CMS")

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mu_target_path", type=str, required=True)
parser.add_argument("--toy_seed", type=int, required=True)
parser.add_argument("--trial_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--mu_i_file", type=str, required=True)
parser.add_argument("--n_points", type=int, default=20000, help="Number of target samples to generate")
args = parser.parse_args()

# ------------------
# Load models and initial weights 
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets device to GPU if available 
os.makedirs(args.trial_dir, exist_ok=True)

f_i_file = os.path.join(args.trial_dir, "f_i.pth")
f_i_statedicts = torch.load(f_i_file, map_location=device) #list of state_dicts 

# ------------------
# Load architecture config and data 
# ------------------
# Save architecture 

with open(os.path.join(args.trial_dir, "architecture_config.json")) as f:
    config = json.load(f) 

# ------------------
# Generate Coverage Data for Target Distribution 
# ------------------
def generate_target_data(n_points, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean_feat1, std_feat1 = -0.5, 0.25
    mean_feat2, std_feat2 = 0.6, 0.4

    feat1 = np.random.normal(mean_feat1, std_feat1, n_points)
    feat2 = np.random.normal(mean_feat2, std_feat2, n_points)
    data = np.column_stack((feat1, feat2)).astype(np.float32)
    return data

data_np = generate_target_data(args.n_points, seed=args.toy_seed)
x_data = torch.from_numpy(data_np).float().to(device)

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

def ensemble_model(weights, model_probs):
    return probs(weights, model_probs)

def constraint_term(weights):
    l=1.0
    return l*(torch.sum(weights)-1.0)
'''
def nll(weights):
    return -torch.log(ensemble_model(weights, model_probs) + 1e-8).mean() + constraint_term(weights)
'''
def nll(weights):
    p = ensemble_model(weights, model_probs)
    # If any ensemble density value is ≤ 0, return +inf to signal an invalid region.
    # This prevents log(0) or log(negative) from causing NaNs and guides the optimizer away from this point.
    if not torch.all(p > 0):
        return torch.tensor(float("inf"), dtype=torch.float64)
    return -torch.log(p + 1e-8).mean() + constraint_term(weights)

max_attempts = 50  # to avoid infinite loops in pathological cases
attempt = 0

while attempt < max_attempts:
    w_i_init_torch = torch.tensor([ 0.0215, -0.0046,  0.0137,  0.0529, -0.0136,  0.0256,  0.0321,  0.0028,
         0.0204, -0.0011,  0.0243,  0.0077, -0.0153, -0.0093, -0.0202,  0.0439,
         0.0574,  0.0570,  0.0030,  0.0066,  0.0562, -0.0074, -0.0659, -0.0010,
        -0.0193,  0.0243,  0.0121,  0.0701,  0.0456, -0.0124,  0.0415,  0.0345,
         0.0566,  0.0450,  0.0043,  0.0105,  0.0392, -0.0049, -0.0073,  0.0012,
         0.0275, -0.0401,  0.0021,  0.0239, -0.0342, -0.0336,  0.0379,  0.0896,
        -0.0099,  0.0422,  0.0198,  0.0489, -0.0175,  0.0601,  0.0183,  0.0108,
         0.0147,  0.0307,  0.0095,  0.0718], dtype=torch.float64, requires_grad=True)  
     
    try:
        print('w_i_init_torch', w_i_init_torch)

        loss_val = nll(w_i_init_torch)
        if not torch.isfinite(loss_val):
            print(f"Attempt {attempt+1} skipped due to non-finite loss: {loss_val.item()}")
            attempt += 1
            continue
        
        res = minimize(
            nll,                         # your function
            w_i_init_torch,              # initial guess
            method='newton-exact',       # method
            options={'disp': False, 'max_iter': 300},     # any options
        )
        
        if res.success:
            break

    except Exception as e:
        print(f"Attempt {attempt+1} failed with exception: {e}")
        traceback.print_exc()
    attempt += 1

if res is None or not res.success:
    raise RuntimeError("Optimization failed after multiple attempts.")

w_i_final = res.x.detach()               # ← what the optimizer produced # ← the true weights you us
final_loss = nll(w_i_final).item()
print("Final loss (NLL):", final_loss)

print("Final weights:", w_i_final)
print("Sum of weights:", w_i_final.detach().cpu().numpy().sum())

# ------------------
# Compute uncertainties via Hessian
# ------------------
# Autograd Hessian for cross-check
H_autograd = hessian(nll, w_i_final.double()).detach()
#print("H_autograd:", torch.norm(H_autograd).item())

def compute_sandwich_covariance(H, w, model_probs, lam=1.0):
    M = len(w)
    N = model_probs.shape[0]

    w = w.detach().clone().double().requires_grad_()
    model_probs = model_probs.double()

    V = torch.linalg.solve(H, torch.eye(M, dtype=torch.float64))
    #print('eigvalues of V', torch.linalg.eigvalsh(V))
    #print('V', V)

    # Compute f(x) = sum_j w_j^2 f_j(x)
    f = (model_probs * w).sum(dim=1)  # shape: (N,)

    grads = torch.zeros((N, M), dtype=torch.float64)  # ∂L/∂w_i(x)

    for j in range(M):
        f_j = model_probs[:, j]
        grads[:, j] = - (f_j) / (f + 1e-12) + lam 

    # Compute U matrix
    mean_grad = grads.mean(dim=0, keepdim=True)
    U = ((grads - mean_grad).T @ (grads - mean_grad)) / N
    #print('eigvalues of U', torch.linalg.eigvalsh(U))
    #print('U', U)

    cov_w = V @ U @ V.T
    cov_w = cov_w/N

    #print("‖V @ U - I‖ =", (V @ U - torch.eye(M, dtype=torch.float64)))
    #print("‖V @ U @ V - V‖ =", (V @ U @ V - V))
    return cov_w

cov_w_autograd = compute_sandwich_covariance(H_autograd, w_i_final, model_probs)
#print("diag cov_autograd:", torch.diag(cov_w_autograd).mean().item())

cov_w_final = cov_w_autograd 
#print("Weight covariance matrix:\n", cov_w_final)

cov_w_final = cov_w_final.detach().cpu()
#print("Weight covariance matrix:\n", cov_w_final)
cov_w_final = cov_w_final.detach().cpu().numpy()

# ------------------
# Coverage test
# ------------------
w_i_final = w_i_final.detach().cpu()

# Upload target's first moment:
# Load target first moment
mu_target = np.load(args.mu_target_path)  # shape: (D,)

# ------------------
# Compute model's first moment μ_model
# ------------------
#Goal: compute mu_i = ∫dx*x*f_i(x) ~ sum(x_k * f_i(x_k) * ∆x)

mu_i = np.load(args.mu_i_file)  # Initial weights
mu_i = torch.tensor(mu_i, dtype=torch.float64)

# Compute μ_model = ∑_i w_i ⋅ μ_i
# This is the ensemble-averaged moment: weighted sum of the μ_i using the optimized weights w_i
mu_model = torch.matmul(w_i_final[None, :], mu_i)  # shape: (D,)

# ------------------
# Compute uncertainty σ_I (propagation of weight covariance)
# ------------------
# We propagate the uncertainty on the weights through the functional:
#     σ²_I = ∇I(w)^T ⋅ Cov(w) ⋅ ∇I(w)
# where ∇I(w) ≡ [∂I/∂w_i] = μ_i (since I(w) = ∑ w_i μ_i)

# Perform the matrix contraction:
#   σ²_I[d] = ∑_i ∑_j μ_i[d] ⋅ Cov_w[i, j] ⋅ μ_j[d]
# for each feature dimension d
cov_w_final = torch.from_numpy(cov_w_final).to(mu_i.dtype)
sigma_sq = torch.einsum("id,ij,jd->d", mu_i, cov_w_final, mu_i)  # shape: (D,)

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