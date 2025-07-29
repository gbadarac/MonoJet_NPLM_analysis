#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils_wifi import probs, plot_ensemble_marginals, profile_likelihood_scan
from utils_flows import make_flow
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # ensure it works on clusters without displaying plots
from torch.autograd.functional import hessian
import mplhep as hep
from torchmin import minimize 
import traceback
import gc  # Add this at the top with other imports


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

f_i_file = os.path.join(args.trial_dir, "f_i.pth")
f_i_statedicts = torch.load(f_i_file, map_location=device) #list of state_dicts 

# ------------------
# Load architecture config and data 
# ------------------
# Save architecture 

with open(os.path.join(args.trial_dir, "architecture_config.json")) as f:
    config = json.load(f) 

x_data = torch.from_numpy(np.load(args.data_path)).float()

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
def ensemble_model(weights, model_probs):
    return probs(weights, model_probs)

def constraint_term(weights):
    return (torch.sum(weights) - 1.0)

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

w_i_initial = np.ones(len(f_i_statedicts)) / len(f_i_statedicts)

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

        print("Calling minimize with:")
        print(f"model_probs shape: {model_probs.shape}, device: {model_probs.device}")
        print(f"weights shape: {w_i_init_torch.shape}, device: {w_i_init_torch.device}")
        print(f"Allocated CUDA: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved CUDA: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

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
cov_w_final = cov_w_final.detach().cpu().numpy()

# ------------------
# Save final outputs
# ------------------
# Save final weights and covariance matrix to correct location
np.save(os.path.join(args.out_dir, "w_i_fitted.npy"), w_i_final.detach().cpu().numpy())
np.save(os.path.join(args.out_dir, "cov_w.npy"), cov_w_final)

torch.cuda.empty_cache()
gc.collect()

# ------------------
# Plotting 
# ------------------
if args.plot.lower() == "true":
    model_probs = model_probs.cpu()  # Before passing into plotting
    feature_names = ["Feature 1", "Feature 2"]

    # ------------------
    # Likelihood profile scan
    # ------------------
    #likelihood_scan(model_probs, w_i_final, args.out_dir)
    #profile_likelihood_scan(model_probs, w_i_final, args.out_dir)
    
    
    # ------------------
    # Uncertainties on the model
    # ------------------
    # Clean up memory before reloading models
    del model_probs_list, model_probs, x_data
    torch.cuda.empty_cache()

    # Reload f_i_models after weight fitting
    f_i_models = []
    for state_dict in f_i_statedicts:
        flow = make_flow(
            num_layers=config["num_layers"],
            hidden_features=config["hidden_features"],
            num_bins=config["num_bins"],
            num_blocks=config["num_blocks"],
        )
        flow.load_state_dict(state_dict)
        flow = flow.to("cpu")
        flow.eval()
        f_i_models.append(flow)

    # Reload x_data again after deletion and move to correct device
    x_data = torch.from_numpy(np.load(args.data_path)).float().to(device)
    plot_ensemble_marginals(f_i_models, x_data, w_i_final, cov_w_final, feature_names, args.out_dir)