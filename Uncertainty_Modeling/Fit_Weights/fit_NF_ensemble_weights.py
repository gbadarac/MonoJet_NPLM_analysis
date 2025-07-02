#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils import make_flow, probs, plot_individual_marginals, plot_ensemble_marginals, profile_likelihood_scan
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
def ensemble_model(weights, model_probs):
    return probs(weights, model_probs)

def constraint_term(weights):
    l=1.0
    return l*(torch.sum(weights)-1.0)

def nll(weights):
    return -torch.log(ensemble_model(weights, model_probs) + 1e-8).mean() + constraint_term(weights)


max_attempts = 30  # to avoid infinite loops in pathological cases
attempt = 0

while attempt < max_attempts:
    #w_i_init_torch = torch.tensor(w_i_initial, dtype=torch.float64, requires_grad=True)
    
    #for N_100000_seeds_30_bootstraps_1_8_4_64_10
    '''
    w_i_init_torch = torch.tensor([ 0.0895,  0.0318, -0.0158,  0.0264,  0.0955,  0.0367,  0.0655,  0.0600,
         0.0616,  0.0551,  0.0477, -0.0466,  0.0589,  0.0135,  0.0530,  0.1307,
        -0.0215, -0.0262,  0.0281,  0.0192,  0.0924,  0.0142,  0.0323,  0.0583,
         0.0078,  0.0100,  0.0474,  0.0114, -0.0129, -0.0239],
       dtype=torch.float64)
    '''
    #for N_100000_seeds_30_bootstraps_1_4_12_64_15
    w_i_init_torch = torch.tensor([ 0.0511,  0.0298,  0.0025,  0.1095,  0.0004,  0.0454, -0.0164,  0.0613,
         0.0319, -0.0104,  0.0301,  0.0738,  0.0335,  0.0279,  0.0335, -0.0180,
         0.0043,  0.0473,  0.0812,  0.0355,  0.0409,  0.0485, -0.0345,  0.0015,
         0.0458,  0.0380,  0.1338,  0.0490, -0.0297,  0.0523],
       dtype=torch.float64)

    try:
        noise = 1e-2 * torch.randn_like(w_i_init_torch)
        # Generate random signs (+1 or -1)
        sign_flips = torch.randint(0, 2, w_i_init_torch.shape, dtype=torch.float64) * 2 - 1
        w_i_init_torch = ((w_i_init_torch + noise)*sign_flips).detach().clone().requires_grad_()
        print('w_i_init_torch', w_i_init_torch)
        res = minimize(
            nll,                         # your function
            w_i_init_torch,              # initial guess
            method='newton-exact',       # method
            options={'disp': False, 'max_iter': 300},     # any options
        )
        break  # success
    except IndexError as e:
        print(f"Attempt {attempt+1}: Optimization failed with error: {e}")
        attempt += 1

    else:
        print("Optimization failed after multiple attempts.") 

w_i_opt = res.x.detach()               # ← what the optimizer produced # ← the true weights you us

w_i_final = w_i_opt

final_loss = nll(w_i_final).item()
print("Final loss (NLL):", final_loss)

print("Final weights:", w_i_final)
print("Sum of weights:", w_i_final.detach().cpu().numpy().sum())

# ------------------
# Compute uncertainties via Hessian
# ------------------
# Autograd Hessian for cross-check
H_autograd = hessian(nll, w_i_opt.double()).detach()
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

cov_w_autograd = compute_sandwich_covariance(H_autograd, w_i_opt, model_probs)
#print("diag cov_autograd:", torch.diag(cov_w_autograd).mean().item())

cov_w_final = cov_w_autograd 
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
    plot_individual_marginals(f_i_models, x_data, feature_names, args.out_dir)
    
    # ------------------
    # Likelihood profile scan
    # ------------------
    #likelihood_scan(model_probs, w_i_final, args.out_dir)
    profile_likelihood_scan(model_probs, w_i_final, args.out_dir)
    
    
    # ------------------
    # Uncertainties on the model
    # ------------------
    plot_ensemble_marginals(f_i_models, x_data, w_i_final, cov_w_final, feature_names, args.out_dir)
