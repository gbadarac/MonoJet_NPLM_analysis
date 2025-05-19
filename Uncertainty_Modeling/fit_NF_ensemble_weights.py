#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import torch.optim as optim
import numpy as np
import argparse
from utils import make_flow, nll, nll_numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # ensure it works on clusters without displaying plots

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
f_i_file = os.path.join(args.trial_dir, "f_i.pth")
w_i_file = os.path.join(args.trial_dir, "w_i_initial.npy")

f_i_statedicts = torch.load(f_i_file, map_location=device) #list of state_dicts 
w_i_initial = np.load(w_i_file)  # Initial weights
print(f"Initial weights: {w_i_initial}")

# ------------------
# Load architecture config and data 
# ------------------
# Save architecture 
'''
with open(os.path.join(args.trial_dir, "architecture_config.json")) as f:
    config = json.load(f) 
'''

configs = []
for model_name in ["good_model", "bad_model"]:
    config_path = os.path.join(args.trial_dir, model_name, "architecture_config.json")
    with open(config_path) as f:
        configs.append(json.load(f))


x_data = torch.from_numpy(np.load(args.data_path)).float().to(device)

# ------------------
# Reconstruct flows f_i
# ------------------
f_i_models = []
#for state_dict in f_i_statedicts:
for state_dict, config in zip(f_i_statedicts, configs):
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_blocks=config["num_blocks"],
        num_bins=config["num_bins"]
    ).to(device) #recreate the flow architecture for each model f_i 
    flow.load_state_dict(state_dict) #load the corresponding params into each model 
    flow.eval() # Set to eval mode to avoid training-time behavior (e.g., dropout)
    #You want all models to produce consistent, deterministic outputs for likelihood evaluation 
    f_i_models.append(flow)

# === TEST 1: Log-prob mean comparison ===
with torch.no_grad():
    logp_0 = f_i_models[0].log_prob(x_data[:10000]).cpu().numpy()
    logp_1 = f_i_models[1].log_prob(x_data[:10000]).cpu().numpy()

print("\nLog-prob diagnostics:")
print(f"  f₀(x) [good model]: mean = {logp_0.mean():.4f}, std = {logp_0.std():.4f}")
print(f"  f₁(x) [bad  model]: mean = {logp_1.mean():.4f}, std = {logp_1.std():.4f}")
print(f"  Diff (mean): {abs(logp_0.mean() - logp_1.mean()):.4f}")

# === TEST 2: Visualize model probabilities ==
with torch.no_grad():
    probs0 = torch.exp(f_i_models[0].log_prob(x_data)).cpu().numpy()
    probs1 = torch.exp(f_i_models[1].log_prob(x_data)).cpu().numpy()

plt.figure(figsize=(6,4))
plt.hist(probs0, bins=100, alpha=0.5, label="f₀(x) [good model]")
plt.hist(probs1, bins=100, alpha=0.5, label="f₁(x) [bad model]")
plt.legend()
plt.title("Model probability distributions")
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "model_probs_hist.png"))

# ------------------
# Diagnostic: Compare f₀(x) vs f₁(x)
# ------------------
if len(f_i_models) >= 2:
    with torch.no_grad():
        probs0 = torch.exp(f_i_models[0].log_prob(x_data))
        probs1 = torch.exp(f_i_models[1].log_prob(x_data))
        diff = (probs0 - probs1).abs()

        print("\nModel probability difference statistics:")
        print("  prob₀ = f₀(x) [good model]")
        print("  prob₁ = f₁(x) [bad  model]")
        print(f"  Mean abs(prob₀ - prob₁): {diff.mean().item():.6e}")
        print(f"  Max abs diff: {diff.max().item():.6e}")
        print(f"  Std of diff: {diff.std().item():.6e}")

# ------------------
# Evaluate f_i(x) -> model_probs
# ------------------
with torch.no_grad(): #disables gradient tracking since we are just evaluating the model (to speed things up)
    model_probs = torch.stack( #torch.stack() to stack all M models along a new dimension dim=1 cretaing a NxM matrix where N are the number of data points 
        [torch.exp(flow.log_prob(x_data)) for flow in f_i_models], 
        #for each model f_i compute log(f_i(x)) for all data points, then applies torch.exp() to recover actual probability f_i(x)
        #this needs to be done because of how NFs are built, i.e. they return log(f_i(x)) but we want to get f_i(x) to get the probability, so we need to take the exponential 
        dim=1)  # Shape: (N, M)

# Move model_probs to CPU for memory safety
model_probs = model_probs.to("cpu")

# Initialize weights as a torch tensor with gradient tracking
w_logits = torch.nn.Parameter(torch.log(torch.tensor(w_i_initial + 1e-8, dtype=torch.float32)))

# Optimizer (Adam is simple and good enough here)
optimizer = optim.Adam([w_logits], lr=1e-2)

# Function to normalize weights, to ensure weights stay in [0,1] and sum to 1
def norm_weights(logits):
    return torch.nn.functional.softmax(logits, dim=0)

# === TEST 3: Sweep NLL across [w₀, w₁] ∈ [1,0] to [0,1] ===
print("\nNLL sweep over alpha ∈ [0, 1]:")
print("  NLL evaluated on: f(x) = (1 - α) * f₀(x) [good] + α * f₁(x) [bad]")
for alpha in np.linspace(0.0, 1.0, 11):
    w = np.array([1.0 - alpha, alpha])
    loss = nll_numpy(w, model_probs, device=device)
    print(f"  alpha = {alpha:.1f}  -> NLL = {loss:.6f}")

# ------------------
# Optimize weights using MLE
# ------------------

print("\nStarting PyTorch optimization:")
for step in range(300):  # number of optimization steps
    optimizer.zero_grad()

    w_i_norm = norm_weights(w_logits) #this might not work is just a function and not a model their not propagating 
    loss= nll(w_i_norm, model_probs)

    loss.backward()
    optimizer.step()

    if step % 25 == 0 or step == 299:
        w_np = w_i_norm.detach().cpu().numpy()
        print(f"Step {step:03d}: NLL = {loss.item():.6f}, weights = {w_np}")
        print(f"            → f₀(x) [good]: {w_np[0]:.4f}, f₁(x) [bad]: {w_np[1]:.4f}")
        #print(f"Step {step:03d}: NLL = {loss.item():.6f}, weights = {w_i_norm.detach().numpy()}")

'''
w_i_final = norm_weights(w_logits).detach().cpu().numpy()
print("Final fitted weights (PyTorch):", w_i_final)
'''
w_i_final = norm_weights(w_logits).detach().cpu().numpy()
print("\nFinal fitted weights (PyTorch):", w_i_final)
print(f"  → Weight on f₀(x) [good model]: {w_i_final[0]:.4f}")
print(f"  → Weight on f₁(x) [bad  model]: {w_i_final[1]:.4f}")

# ------------------
# Compute uncertainties via Hessian
# ------------------
# === TEST 4: Compute Hessian w.r.t. raw weights ===
print("\nComputing Hessian manually (no softmax)...")
w_raw = torch.tensor(w_i_final, dtype=torch.float32, requires_grad=True)

loss = nll(w_raw / w_raw.sum(), model_probs)
grad, = torch.autograd.grad(loss, w_raw, create_graph=True)
hess_diag = []
for g_i in grad:
    h_i, = torch.autograd.grad(g_i, w_raw, retain_graph=True)
    hess_diag.append(h_i)
hess_diag = torch.stack(hess_diag).detach().cpu().numpy()
print("Manual Hessian diag (raw weights):", hess_diag)

w_i_unc = 1.0 / np.sqrt(hess_diag + 1e-8)

# ------------------
# Save final outputs
# ------------------
np.save(os.path.join(args.out_dir, "w_i_fitted.npy"), w_i_final)
np.save(os.path.join(args.out_dir, "w_i_unc.npy"), w_i_unc)

'''
# ------------------
# Plotting 
# ------------------
if args.plot.lower() == "true":
    f_vals = ensemble_pred(w_i_np, model_probs)
    f_unc = ensemble_unc(w_i_unc, model_probs)

    # Histogram over f(x) values
    plt.figure(figsize=(8, 6))
    x_vals = np.linspace(np.min(f_vals - f_unc), np.max(f_vals + f_unc), 100)
    plt.hist(f_vals, bins=100, density=True, alpha=0.6, label="f(x) = ∑ w_i f_i(x)")
    plt.fill_between(x_vals,
                     np.interp(x_vals, np.sort(f_vals), np.sort(f_vals - f_unc)),
                     np.interp(x_vals, np.sort(f_vals), np.sort(f_vals + f_unc)),
                     alpha=0.3, label="Uncertainty band")

    plt.xlabel("Ensemble density f(x)")
    plt.ylabel("Density")
    plt.title("Ensemble NF prediction with uncertainty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, "ensemble_density_with_uncertainty.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

'''


