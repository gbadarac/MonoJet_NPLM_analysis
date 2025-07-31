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
import gc 

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
x_data = torch.from_numpy(data_np).float()

# ------------------
# Evaluate f_i(x) -> model_probs
# ------------------
model_probs_list = []

for state_dict in f_i_statedicts:
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_bins=config["num_bins"],
        num_blocks=config["num_blocks"],
    )

    flow.load_state_dict(state_dict)
    flow = flow.to("cpu")  # keep model on CPU

    flow.eval()
    batch_size = 5000
    flow_probs = []

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i:i+batch_size].to("cpu")  # keep data on CPU too
            logp_batch = flow.log_prob(x_batch)
            flow_probs.append(torch.exp(logp_batch))

    flow_probs_tensor = torch.cat(flow_probs, dim=0).detach()

    model_probs_list.append(flow_probs_tensor)

    # Cleanup
    del flow_probs
    del flow
    torch.cuda.empty_cache()
    gc.collect()


model_probs = torch.stack(model_probs_list, dim=1).to("cpu").requires_grad_()

# model_probs is a (N,M) matrix of type (f_0, ... f_(M-1)) where N is the number of data points


# ------------------
# Optimize weights via logits using MLE
# ------------------
# Convert initial weights to torch tensor

def ensemble_model(weights, model_probs):
    return probs(weights, model_probs)

def constraint_term(weights):
    l=100.0
    return l*(torch.sum(weights)-1.0)**2

def nll(weights):
    weights = weights.to("cpu")
    p = probs(weights, model_probs)  # (N,)

    if not torch.all(p > 0):
        # Use same device and dtype
        return torch.tensor(float("inf"), dtype=weights.dtype, device=weights.device)

    loss = -torch.log(p + 1e-8).mean() + constraint_term(weights)
    return loss  # Do NOT detach, do NOT call `.item()`, do NOT convert to float


max_attempts = 50  # to avoid infinite loops in pathological cases
attempt = 0

#put it specifically from the fitting part 
w_i_init_torch = torch.tensor([ 2.6660e-02,  4.9029e-03,  5.4670e-05,  1.9149e-02,  6.4070e-02,
    9.3286e-03,  3.6494e-02,  8.7958e-02,  7.6974e-02, -1.5255e-02,
    -3.9479e-02, -1.5420e-02, -1.5285e-02,  8.7044e-02,  5.7290e-02,
    1.5961e-02, -3.6252e-02, -4.3677e-02, -1.4823e-04, -3.4308e-02,
    -2.3378e-02,  3.6204e-02,  3.1594e-02,  8.5678e-03,  1.2468e-02,
    2.5487e-02,  1.9723e-02,  2.5634e-02,  6.8171e-02, -9.8194e-03,
    8.0849e-03,  2.1400e-02,  3.2348e-02,  6.7841e-02,  4.6322e-02,
    4.8491e-02, -3.6525e-02, -1.7060e-02,  4.7686e-02,  1.4837e-02,
    3.4592e-02, -2.2595e-02, -6.4887e-02,  1.0992e-01, -3.1935e-03,
    -7.6124e-02, -2.0110e-02,  4.2485e-02,  3.6925e-02, -1.5192e-02,
    4.6197e-02,  4.1148e-02, -9.9237e-03,  4.3357e-02,  3.0554e-02,
    -1.3172e-02, -2.7091e-02,  6.0365e-02,  2.3911e-02,  6.8697e-02],
    dtype=torch.float64, requires_grad=True)  

while attempt < max_attempts:
    try:
        noise = 1e-3 * torch.abs(torch.randn_like(w_i_init_torch))
        w_i_init_torch = (w_i_init_torch * (1.0 + noise)).detach().clone().requires_grad_()

        # Pre-check: make sure resulting ensemble density is valid
        with torch.no_grad():
            p = probs(w_i_init_torch, model_probs)
            if not torch.all(p > 0):
                print(f"Attempt {attempt+1} skipped: f(x) contains non-positive values after perturbation")
                attempt += 1 
                continue

        print('w_i_init_torch', w_i_init_torch)

        loss_val = nll(w_i_init_torch)
        if not torch.isfinite(loss_val):
            print(f"Attempt {attempt+1} skipped due to non-finite loss: {loss_val.item()}")
            continue

        print(f"Attempt {attempt+1}: Starting optimization (loss = {loss_val.item():.4f})")

        torch.cuda.empty_cache()
        gc.collect()
        
        res = minimize(
            nll,                         # your function
            w_i_init_torch.to("cpu"),    # initial guess
            method='newton-exact',       # method
            options={'disp': False, 'max_iter': 300},     # any options
        )
        
        if res.success:
            print(f"Optimization succeeded on attempt {attempt}")
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