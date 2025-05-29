#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils import make_flow, log_likelihood, ensemble_pred, ensemble_unc, plot_marginals, plot_ensemble_marginals
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # ensure it works on clusters without displaying plots
from torch.autograd.functional import hessian
import mplhep as hep

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

for step in range(300):  # number of optimization steps
    optimizer.zero_grad()

    w_i_norm = norm_weights(w_i_logits) 
    loss= -log_likelihood(w_i_norm, model_probs)

    loss.backward()
    optimizer.step()

    if step % 25 == 0 or step == 299:
        w = w_i_norm.detach().cpu().numpy()
        print(f"Step {step:03d}: NLL = {loss.item():.6f}, weights = {w}")
        print(f"            → f₀(x): {w[0]:.4f}, f₁(x): {w[1]:.4f}")

w_i_final = norm_weights(w_i_logits).detach().cpu().numpy()
print("\nFinal fitted weights (PyTorch):", w_i_final)
print(f"  → Weight on f₀(x) : {w_i_final[0]:.4f}")
print(f"  → Weight on f₁(x) : {w_i_final[1]:.4f}")

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
H_z_inv = np.linalg.pinv(H_z)  # robust inversion
cov_w = J @ H_z_inv @ J.T # to get covariance in the weights space 

print("Weight covariance matrix:\n", cov_w)

# ------------------
# Save final outputs
# ------------------
np.save(os.path.join(args.out_dir, "w_i_fitted.npy"), w_i_final)
np.save(os.path.join(args.out_dir, "cov_w.npy"), cov_w)

# ------------------
# Plotting 
# ------------------
if args.plot.lower() == "true":
    # ------------------
    # Marginal plots
    # ------------------
    with torch.no_grad():
        samples_0 = f_i_models[0].sample(10000).cpu().numpy()
        samples_1 = f_i_models[1].sample(10000).cpu().numpy()

    feature_names = ["Feature 1", "Feature 2"]
    labels = ('Model 1', 'Model 2')
    plot_marginals(samples_0, samples_1, x_data, feature_names, args.out_dir, labels)

    # ------------------
    # Uncertainties on the model
    # ------------------
    plot_ensemble_marginals(model_probs, x_data, w_i_final, cov_w, feature_names, args.out_dir)