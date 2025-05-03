#!/usr/bin/env python
# coding: utf-8

import os
import json
import torch
import numpy as np
import argparse
from nflows.flows import Flow
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from scipy.optimize import minimize

'''
f_final = make_flow(
        num_features=2,
        num_context=None,
        perm=True,
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_blocks=config["num_blocks"],
        num_bins=config["num_bins"]
    ).to(device)

state_dicts_fi = [f.state_dict() for f in f_i_models]
final_state_dict = average_state_dicts(state_dicts_fi, w_i_values)
f_final.load_state_dict(final_state_dict)
torch.save(f_final.state_dict(), os.path.join(ensemble_dir, "final_model.pth"))

print("Final ensemble model saved to 'final_model.pth'")
'''

# ------------------
# Helper functions
# ------------------

def load_model(model_dir):
    """Load a trained NF model from model.pth and config.json."""
    # Load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:  #loads model from a given model_dir 
        config = json.load(f)

    # Rebuild model from config file 
    num_features = 2
    base_dist = StandardNormal(shape=(num_features,))
    transforms = []
    for i in range(config["num_layers"]):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=num_features,
            hidden_features=config["hidden_features"],
            context_features=None,
            num_bins=config["num_bins"],
            num_blocks=config["num_blocks"],
            tail_bound=10.0,
            dropout_probability=0.2,
            use_batch_norm=False
        ))
        if i < config["num_layers"] - 1:
            transforms.append(ReversePermutation(features=num_features))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    
    # Load weights from model.pth file 
    flow.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
    flow.eval()
    return flow

# DEF PROBABILITY (probability deve essere definita in torch per fare back propagation) 
# variable function anf and requirees_grad=traine_coeffs (train_coeffs=True)

def negative_log_likelihood(weights, log_probs_matrix):
    """Compute negative log likelihood for the ensemble."""
    weights = np.clip(weights, 1e-8, 1.0)  # Avoid zeros
    # np.exp(log_probs_matrix) is the porbability predicted by the models for sample x 
    weighted_probs = np.exp(log_probs_matrix) @ weights # matrix-vector product between atrix of probabilities and vector of weights  
    nll = -np.sum(np.log(weighted_probs + 1e-8))
    return nll

def hessian_diag(func, x0, eps=1e-5):
    """Numerical diagonal Hessian approximation."""
    n = len(x0)
    hess_diag = np.zeros(n)
    fx = func(x0)
    for i in range(n):
        x_eps = np.array(x0)
        x_eps[i] += eps
        f_eps = func(x_eps)
        hess_diag[i] = (f_eps - 2*fx + func(x0 - eps*np.eye(1, n, i)[0])) / (eps**2)
    return hess_diag


# ------------------
# Main script
# ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_dir', type=str, required=True, help='Path to trial folder (e.g., trial_000)')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to save results')
    parser.add_argument('--n_seeds', type=int, required=True, help='Number of seeds')
    parser.add_argument('--n_runs', type=int, required=True, help='Number of runs per seed')
    parser.add_argument('--n_val_samples', type=int, default=10000, help='Number of validation samples')
    args = parser.parse_args()

    trial_dir = args.trial_dir
    output_dir = args.output_dir or os.path.join(trial_dir, "..", "Uncertainty_Modeling", f"results_{os.path.basename(trial_dir)}")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------
    # Load all models
    # ------------------
    models = []
    for seed in range(args.n_seeds):
        for run in range(args.n_runs):
            model_path = os.path.join(trial_dir, f"seed_{seed:04d}", f"run_{run:02d}")
            model = load_model(model_path)
            models.append(model)

    print(f"Loaded {len(models)} models.")

    # ------------------
    # Load validation data
    # ------------------
    # For now: randomly sample from standard normal for testing
    x_val = torch.randn(args.n_val_samples, 2)

    # ------------------
    # Compute log_probs for each model
    # ------------------
    log_probs_matrix = []
    with torch.no_grad():
        for model in models:
            log_probs = model.log_prob(x_val).numpy()
            log_probs_matrix.append(log_probs)
    log_probs_matrix = np.stack(log_probs_matrix, axis=1)  # shape (n_samples, n_models)

    print(f"Computed log_probs_matrix shape: {log_probs_matrix.shape}")

    # ------------------
    # Optimize weights
    # ------------------
    n_models = log_probs_matrix.shape[1]
    initial_weights = np.ones(n_models) / n_models

    #minimize nll function to get best weights 
    result = minimize(
        negative_log_likelihood,
        initial_weights,
        args=(log_probs_matrix,),
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * n_models
    )

    best_weights = result.x
    print(f"Optimization success: {result.success}")
    print(f"Best weights: {best_weights}")

    # ------------------
    # Estimate uncertainties
    # ------------------
    hess_diag = hessian_diag(lambda w: negative_log_likelihood(w, log_probs_matrix), best_weights)
    weights_uncertainties = np.sqrt(1.0 / (hess_diag + 1e-8))

    # ------------------
    # Save results
    # ------------------
    np.save(os.path.join(output_dir, "weights.npy"), best_weights)
    np.save(os.path.join(output_dir, "weights_uncertainties.npy"), weights_uncertainties)
    with open(os.path.join(output_dir, "fit_log.txt"), "w") as f:
        f.write(f"Optimization success: {result.success}\n")
        f.write(f"Best loss: {result.fun}\n")
        f.write(f"Best weights:\n{best_weights}\n")
        f.write(f"Weights uncertainties:\n{weights_uncertainties}\n")

    print(f"Saved weights and uncertainties to {output_dir}")
