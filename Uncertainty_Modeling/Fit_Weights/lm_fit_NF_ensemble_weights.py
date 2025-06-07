#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils import make_flow, probs, log_likelihood, plot_marginals, plot_ensemble_marginals, profile_likelihood_scan
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # ensure it works on clusters without displaying plots
from torch.autograd.functional import hessian
import mplhep as hep
from torchmin import minimize 

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

x_data = torch.from_numpy(np.load(args.data_path)).float().to(device)

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
        #this is standard in NFs because calculating log(f_i(x)) is numerically stable and avoids underflow 
        dim=1)  # Shape: (N, M)

# Move model_probs and f_I_models to CPU for memory safety
# model_probs is a (N,M) matrix of type (f_0, ... f_(M-1)) where N is the number of data points 
model_probs = model_probs.to("cpu") 
# ------------------
# Optimize weights via logits using MLE
# ------------------
# Convert initial weights to torch tensor
w_i_init_torch = torch.tensor(w_i_initial, dtype=torch.float32, requires_grad=True)
M = model_probs.shape[1]
#w_i_init_torch = torch.tensor(np.ones(M), dtype=torch.float32, requires_grad=True)

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

print("Final weights:", w_i_final)
print("Sum of weights:", w_i_final.detach().cpu().numpy().sum())

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

#debug 
eigvals = np.linalg.eigvalsh(H_torch)
print("H:manual eigenvalues:", eigvals)
print("H_manual condition number:", eigvals.max() / eigvals.min())

# Autograd Hessian for cross-check
H_autograd = hessian(nll, w_i_opt.double()).detach()
#print("H_autograd:", torch.norm(H_autograd).item())

eigvals = np.linalg.eigvalsh(H_autograd)
print("H_autograd eigenvalues:", eigvals)
print("H_autograd condition number:", eigvals.max() / eigvals.min())

def compute_sandwich_covariance(H, w, model_probs, lam=1.0):
    M = len(w)
    N = model_probs.shape[0]

    w = w.detach().clone().double().requires_grad_()
    model_probs = model_probs.double()

    V = torch.linalg.solve(H, torch.eye(M, dtype=torch.float64))
    print('eigvalues of V', torch.linalg.eigvalsh(V))
    #print('V', V)

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
    print('eigvalues of U', torch.linalg.eigvalsh(U))
    #print('U', U)

    cov_w = V @ U @ V.T
    cov_w = cov_w / N

    print("‖V @ U - I‖ =", (V @ U - torch.eye(M, dtype=torch.float64)))
    print("‖V @ U @ V - V‖ =", (V @ U @ V - V))
    return cov_w

cov_w_manual = compute_sandwich_covariance(H_manual, w_i_opt, model_probs)
print("diag cov_manual:", torch.diag(cov_w_manual).mean().item())
cov_w_autograd = compute_sandwich_covariance(H_autograd, w_i_opt, model_probs)
print("diag cov_autograd:", torch.diag(cov_w_autograd).mean().item())

def jacobian_square_transform(w):
    """
    Jacobian of w_final = w^2 with respect to raw w
    """
    w = w.detach().clone().double()
    return torch.diag(2 * w)
# Jacobian transform of cov_w
J = jacobian_square_transform(w_i_opt)
#print('J=', J)

cov_w_final = J @ cov_w_manual @ J.T
print("Weight covariance matrix:\n", cov_w_final)
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
    
    # -----------------
    # Marginal plots
    # ------------------
    with torch.no_grad():
        samples_0 = f_i_models[0].sample(10000).cpu().numpy()
        samples_1 = f_i_models[1].sample(10000).cpu().numpy()

    labels = ('Model 1', 'Model 2')
    plot_marginals(samples_0, samples_1, x_data, feature_names, args.out_dir, labels)
    
    # ------------------
    # Likelihood profile scan
    # ------------------
    #likelihood_scan(model_probs, w_i_final, args.out_dir)
    profile_likelihood_scan(model_probs, w_i_final, args.out_dir)
    
    # ------------------
    # Uncertainties on the model
    # ------------------
    plot_ensemble_marginals(f_i_models, x_data, w_i_final, cov_w_final, feature_names, args.out_dir)
