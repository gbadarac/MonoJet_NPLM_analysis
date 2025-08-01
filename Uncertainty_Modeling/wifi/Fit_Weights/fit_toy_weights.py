#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils_wifi import probs, plot_gaussian_toy_marginals
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # ensure it works on clusters without displaying plots
from torch.autograd.functional import hessian
import mplhep as hep
from torchmin import minimize 
import traceback


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

f_i_file = os.path.join(args.trial_dir, "f_i.npy")

model_probs_np = np.load(f_i_file)  # shape (N, M)
model_probs = torch.tensor(model_probs_np, dtype=torch.float64, device=device)
# Load data (x_eval) – only needed for plotting
x_data = torch.from_numpy(np.load(args.data_path)).float().to("cpu")

# ------------------
# Optimize weights via logits using MLE
# ------------------
def constraint_term(weights):
    l =1e5
    return l * (torch.sum(weights) - 1.0)**2 

def probs(weights, model_probs):
    # weights: (M,), model_probs: (N, M)
    return (model_probs * weights).sum(dim=1)  # returns (N,)

def nll(weights):
    p = probs(weights, model_probs)
    if not torch.all(p > 0):
        # Use same device and dtype
        return weights.sum() * float("inf")
    loss = -torch.log(p + 1e-8).mean() + constraint_term(weights)
    return loss  # Do NOT detach, do NOT call `.item()`, do NOT convert to float

max_attempts = 50  # to avoid infinite loops in pathological cases
attempt = 0

w_i_initial = np.ones(model_probs.shape[1]) / model_probs.shape[1]

while attempt < max_attempts:
    w_i_init_torch = torch.tensor(w_i_initial, dtype=torch.float64, requires_grad=True)
    noise = 1e-2 * torch.randn_like(w_i_init_torch)
    w_i_init_torch = (w_i_init_torch + noise).detach().clone().requires_grad_()
    print('w_i_init_torch:', w_i_init_torch)

    try:
        loss_val = nll(w_i_init_torch)
        print("loss grad_fn:", loss_val.grad_fn)

        if not torch.isfinite(loss_val):
            print(f"Attempt {attempt+1} skipped due to non-finite loss: {loss_val.item()}")
            continue

        print(f"Attempt {attempt+1}: Starting optimization (loss = {loss_val.item():.4f})")
 
        res = minimize(
            nll,
            w_i_init_torch,
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

# ------------------
# Plotting 
# ------------------
if args.plot.lower() == "true":
    feature_names = ["Feature 1", "Feature 2"]

    # ------------------
    # Likelihood profile scan
    # ------------------
    #likelihood_scan(model_probs, w_i_final, args.out_dir)
    #profile_likelihood_scan(model_probs, w_i_final, args.out_dir)
    
    
    # ------------------
    # Uncertainties on the model
    # ------------------
    plot_gaussian_toy_marginals(model_probs, x_data, w_i_final, cov_w_final, feature_names, args.out_dir)