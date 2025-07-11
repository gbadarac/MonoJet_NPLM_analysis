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

f_i_file = os.path.join(args.trial_dir, "f_i.pth")
f_i_statedicts = torch.load(f_i_file, map_location=device) #list of state_dicts 

# ------------------
# Load architecture config and data 
# ------------------
# Save architecture 

with open(os.path.join(args.trial_dir, "architecture_config.json")) as f:
    config = json.load(f) 

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


max_attempts = 50  # to avoid infinite loops in pathological cases
attempt = 0

w_i_initial = np.ones(len(f_i_statedicts)) / len(f_i_statedicts)
print(f"Initial weights: {w_i_initial}")

while attempt < max_attempts:
    #w_i_init_torch = torch.tensor(w_i_initial, dtype=torch.float64, requires_grad=True)
    '''
    #N_100000_seeds_60_4_16_128_15
    w_i_init_torch = torch.tensor([ 2.7465e-02,  3.9775e-02,  2.9415e-02,  6.0815e-02,  1.6158e-02,
         8.9291e-02,  5.7799e-03, -1.1811e-02,  2.4000e-02,  3.1447e-02,
         8.1284e-02, -2.3717e-02,  2.6356e-02, -1.7371e-02, -1.0964e-02,
         8.3690e-03, -8.1378e-03, -2.3745e-02,  2.1039e-02, -2.9559e-04,
        -4.8600e-02,  6.1569e-02, -4.8001e-02, -3.0407e-02, -1.4790e-02,
        -2.3764e-02,  5.1739e-02,  3.5346e-02,  8.6797e-03,  4.0055e-02,
        -8.1824e-03, -7.8225e-02,  5.8031e-02,  5.0158e-02, -5.7191e-03,
         1.7794e-02,  6.6486e-02,  1.7646e-02, -1.0253e-02,  1.3935e-02,
        -4.6986e-02,  3.5882e-03,  3.0478e-02,  9.8500e-02,  1.6057e-02,
        -4.4616e-03,  5.2506e-03, -4.2853e-03,  2.5735e-02,  1.7025e-02,
         7.2026e-02,  1.9939e-02, -1.0673e-02,  2.1200e-02,  9.2340e-02,
        -3.0226e-02,  4.8697e-02,  7.0628e-05,  7.8944e-02,  4.8129e-02],
       dtype=torch.float64, requires_grad=True)
    '''
    #N_10000_seeds_60_4_16_128_15
    w_i_init_torch = torch.tensor([-0.0560, -0.0175,  0.0879,  0.0173, -0.0798,  0.1132, -0.0779,  0.1823,
         0.0411,  0.1199,  0.1308, -0.0283,  0.0303, -0.2220, -0.0483, -0.1097,
        -0.0115,  0.0294,  0.0640,  0.1994,  0.0181,  0.1054,  0.0027,  0.0639,
        -0.1045,  0.1262,  0.0713,  0.0673, -0.0320,  0.0080,  0.0598, -0.1492,
         0.0786,  0.0501,  0.0270, -0.0269,  0.0496, -0.0210,  0.2292,  0.0553,
        -0.0439, -0.0717, -0.0065, -0.0134, -0.0471, -0.0475,  0.1849, -0.0451,
         0.0177, -0.0921,  0.0528,  0.0493, -0.0757,  0.0068,  0.1353, -0.0066,
        -0.1119,  0.0534, -0.0302,  0.0478], dtype=torch.float64, requires_grad=True)

    try:
        #noise = 1e-2 * torch.randn_like(w_i_init_torch)
        #sign_flips = torch.randint(0, 2, w_i_init_torch.shape, dtype=torch.float64) * 2 - 1
        #w_i_init_torch = ((w_i_init_torch + noise) * sign_flips).detach().clone().requires_grad_()
        
        print('w_i_init_torch:', w_i_init_torch)
        
        res = minimize(
            nll,
            w_i_init_torch,
            method='newton-exact',
            options={'disp': False, 'max_iter': 300},
        )
        break  # Success
    except IndexError as e:
        print(f"Attempt {attempt + 1}: Optimization failed with error: {e}")
        attempt += 1
else:
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
    
    # -----------------
    # Marginal plots
    # ------------------
    num_samples=1000
    out_dir_maginals = os.path.join(args.out_dir, "marginals")
    os.makedirs(out_dir_maginals, exist_ok=True)
    plot_individual_marginals(f_i_models, x_data, feature_names, out_dir_maginals, num_samples)
    
    # ------------------
    # Likelihood profile scan
    # ------------------
    #likelihood_scan(model_probs, w_i_final, args.out_dir)
    profile_likelihood_scan(model_probs, w_i_final, args.out_dir)
    
    
    # ------------------
    # Uncertainties on the model
    # ------------------
    plot_ensemble_marginals(f_i_models, x_data, w_i_final, cov_w_final, feature_names, args.out_dir)