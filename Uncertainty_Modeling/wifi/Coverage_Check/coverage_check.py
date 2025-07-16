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

def nll(weights):
    return -torch.log(ensemble_model(weights, model_probs) + 1e-8).mean() + constraint_term(weights)

max_attempts = 50  # to avoid infinite loops in pathological cases
attempt = 0

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
       dtype=torch.float64, requires_grad=True)  # Initial weights
    
    #N_10000_seeds_60_4_16_128_15
    w_i_init_torch = torch.tensor([-0.0560, -0.0175,  0.0879,  0.0173, -0.0798,  0.1132, -0.0779,  0.1823,
            0.0411,  0.1199,  0.1308, -0.0283,  0.0303, -0.2220, -0.0483, -0.1097,
            -0.0115,  0.0294,  0.0640,  0.1994,  0.0181,  0.1054,  0.0027,  0.0639,
            -0.1045,  0.1262,  0.0713,  0.0673, -0.0320,  0.0080,  0.0598, -0.1492,
            0.0786,  0.0501,  0.0270, -0.0269,  0.0496, -0.0210,  0.2292,  0.0553,
            -0.0439, -0.0717, -0.0065, -0.0134, -0.0471, -0.0475,  0.1849, -0.0451,
            0.0177, -0.0921,  0.0528,  0.0493, -0.0757,  0.0068,  0.1353, -0.0066,
            -0.1119,  0.0534, -0.0302,  0.0478], dtype=torch.float64, requires_grad=True)  

    #N_10000_seeds_60_4_48_128_45
    w_i_init_torch = torch.tensor([-0.0244,  0.0198,  0.0226,  0.0310, -0.0179,  0.0921,  0.0374,  0.0370,
         0.0258,  0.0710,  0.0119, -0.0055, -0.0191, -0.0136, -0.0198,  0.0362,
         0.0285, -0.0056,  0.0097,  0.0093,  0.0283, -0.0203,  0.0071,  0.0431,
         0.0389, -0.0107,  0.0236, -0.0130,  0.0047,  0.0052, -0.0007,  0.0173,
         0.0194,  0.0308,  0.0552,  0.0220,  0.0020, -0.0174,  0.0146,  0.0193,
         0.0045,  0.0117,  0.0060,  0.0495,  0.0245,  0.0130,  0.0186,  0.0137,
        -0.0061,  0.0362,  0.0074,  0.0105, -0.0174,  0.0132, -0.0008,  0.0717,
         0.0588,  0.0322,  0.0245,  0.0324], dtype=torch.float64, requires_grad=True)  
    '''

    #N_10000_seeds_60_4_16_256_15
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
        -1.3172e-02, -2.7091e-02,  6.0365e-02,  2.3911e-02,  6.8697e-02], dtype=torch.float64, requires_grad=True)  
     
    try:
        #noise = 1e-2 * torch.randn_like(w_i_init_torch)
        #sign_flips = torch.randint(0, 2, w_i_init_torch.shape, dtype=torch.float64) * 2 - 1
        #w_i_init_torch = ((w_i_init_torch + noise)*sign_flips).detach().clone().requires_grad_()
        print('w_i_init_torch', w_i_init_torch)
        
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