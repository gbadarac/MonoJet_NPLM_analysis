#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from Uncertainty_Modeling.wifi.utils_NF_wifi import probs
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
    """
    Generate a 4D Gaussian with per-feature means/stds.
    Returns an array of shape (n_points, 4), dtype float32.
    """
    if seed is not None:
        np.random.seed(seed)

    means = np.array([-0.5, 0.6, 0.2, -0.9], dtype=np.float32)
    stds  = np.array([ 0.25, 0.4, 0.3, 0.5], dtype=np.float32)

    data = np.random.normal(loc=means, scale=stds, size=(n_points, 4)).astype(np.float32)
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
        num_features=config["num_features"]
    )

    flow.load_state_dict(state_dict)
    flow = flow.to("cpu")  # keep model on CPU

    flow.eval()
    batch_size = 5000 #3200
    flow_probs = []

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i:i+batch_size].to("cpu")  # keep data on CPU too
            logp_batch = flow.log_prob(x_batch)
            flow_probs.append(torch.exp(logp_batch))
            print(f"Allocated CUDA: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved CUDA: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

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

'''
def constraint_term(weights):
    l=1e5
    return l*(torch.sum(weights)-1.0)**2

def nll(weights):
    p = probs(weights, model_probs)          # (N,)
    eps = 1e-9                               # numerical floor
    # barrier only on negative f(x)
    pen_neg = 1e6 * torch.relu(-p).pow(2).mean()
    # use clamped p in the log to avoid NaNs during early iters
    loss_ll = -torch.log(torch.clamp(p, min=eps)).mean()
    sum1_pen = 1e5 * (weights.sum() - 1.0)**2
    ridge = 1e-12 * (weights**2).sum()
    return loss_ll + sum1_pen + ridge + pen_neg
'''

def constraint_term(weights):
    l=1e0
    return l*(torch.sum(weights)-1.0)

def nll(weights):
    # assume: weights is float64 CPU and requires_grad=True
    p = torch.clamp_min(probs(weights, model_probs), 0.0)  # (N,)
    loss = -torch.log(p + 1e-12).mean() + constraint_term(weights)
    return loss

max_attempts = 50  # to avoid infinite loops in pathological cases
attempt = 0

w_i_initial = np.ones(len(f_i_statedicts)) / len(f_i_statedicts)

res = None
while attempt < max_attempts:
    w_i_init_torch = torch.tensor(w_i_initial, dtype=torch.float64, requires_grad=True)
    noise = 1e-2 * torch.randn_like(w_i_init_torch)
    w_i_init_torch = (w_i_init_torch + noise).detach().clone().requires_grad_()
    print('w_i_init_torch:', w_i_init_torch)

    try:
        loss_val = nll(w_i_init_torch)
        if not torch.isfinite(loss_val):
            print(f"Attempt {attempt+1} skipped due to non-finite loss: {loss_val.item()}")
            continue

        print(f"Attempt {attempt+1}: Starting optimization (loss = {loss_val.item():.4f})")

        torch.cuda.empty_cache()
        gc.collect()

        res = minimize(
            nll,
            w_i_init_torch.to("cpu"),
            method='newton-exact',
            options={'disp': False, 'max_iter': 300},
        )

        if res.success:
            print(f"Optimization succeeded on attempt {attempt}")
            break

        attempt += 1
        
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
w_for_hess = w_i_final.detach().double().requires_grad_()
H_autograd = hessian(nll, w_for_hess).detach()#print("H_autograd:", torch.norm(H_autograd).item())

def compute_sandwich_covariance_4d(H, w, model_probs):
    H = 0.5 * (H + H.T)
    M, N = w.numel(), model_probs.shape[0]

    # basis for the sum-to-one tangent space
    E = torch.eye(M, dtype=torch.float64)
    S_raw = E[:, :-1] - E[:, [-1]]
    S, _ = torch.linalg.qr(S_raw)             # (M, M-1)

    H_red = S.T @ H @ S

    f = (model_probs * w).sum(dim=1)          # (N,)
    # adaptive floor to avoid exploding scores on tiny f
    q01 = torch.quantile(f.detach(), 0.001)
    eps = max(1e-6, float(q01.item()) * 0.1)
    f_safe = torch.clamp(f, min=eps)

    # data-score Jacobian (NO +lam here)
    G  = -(model_probs / f_safe[:, None])     # (N, M)
    Gc = G - G.mean(dim=0, keepdim=True)
    U  = (Gc.T @ Gc) / N                      # (M, M)
    U_red = S.T @ U @ S

    # safe inverse of H_red
    he = torch.linalg.eigvalsh(H_red)
    ridge = (1e-10 + 1e-8 * float(he.abs().max()))
    H_red = H_red + ridge * torch.eye(M-1, dtype=torch.float64)
    V_red = torch.linalg.solve(H_red, torch.eye(M-1, dtype=torch.float64))

    cov_red = (V_red @ U_red @ V_red.T) / N
    cov_w   = S @ cov_red @ S.T
    cov_w   = 0.5 * (cov_w + cov_w.T)

    # tiny PSD clean-up
    ce, Ue = torch.linalg.eigh(cov_w)
    ce = torch.clamp(ce, min=1e-18)
    cov_w = Ue @ torch.diag(ce) @ Ue.T
    return cov_w

cov_w_final = compute_sandwich_covariance_4d(H_autograd, w_i_final, model_probs)
#print("diag cov_autograd:", torch.diag(cov_w_autograd).mean().item())
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