"""
Script 3: Coverage test for the first-moment observable <x0>.

Checks that the Hessian-based uncertainty on <x0> achieves ~68% coverage
over bootstrap resamples of the training data. Handles both norm-kernel
and post-hoc normalization modes.

Reads: {OUT_DIR}/data_config.json, train_config.json, model.pt, covariance.npy
       {DATA_DIR}/data_train.npy
Saves: {OUT_DIR}/coverage_results.json, coverage_pulls.png
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hiker import reconstruct_hiker
from hessian_uq import (compute_hessian, make_nll_fn,
                        compute_covariance_regularised, compute_covariance_projected)

# ── Paths ─────────────────────────────────────────────────────────────
OUT_DIR = os.environ.get("HIKER_OUT_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data, model, covariance ──────────────────────────────────────
print("=" * 60)
print("Loading data and model...")
print("=" * 60)

with open(os.path.join(OUT_DIR, "data_config.json")) as f:
    data_config = json.load(f)
DATA_DIR = data_config.get("data_dir", OUT_DIR)
data_train = np.load(os.path.join(DATA_DIR, "data_train.npy"))
x_train = torch.from_numpy(data_train).double().to(DEVICE)
N_TRAIN = x_train.shape[0]

with open(os.path.join(OUT_DIR, "train_config.json")) as f:
    train_config = json.load(f)

d = data_train.shape[1]

# Reconstruct model and load weights
model = reconstruct_hiker(train_config, d=d).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model.pt"), map_location=DEVICE))
model.eval()

cov_w = torch.from_numpy(np.load(os.path.join(OUT_DIR, "covariance.npy"))).double().to(DEVICE)
layer_sizes = [layer.M for layer in model.layers]
M_total = sum(layer_sizes)

print(f"  N_train = {N_TRAIN}, M_total = {M_total}, layers = {layer_sizes}")

# ══════════════════════════════════════════════════════════════════════
# Compute <x0> and its uncertainty from the fitted model
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("First-moment observable <x0>")
print("=" * 60)

with torch.no_grad():
    w_hat = model.get_free_coeffs().detach().clone()

    if model.use_norm_kernel:
        # <x0> = sum_l sum_i w_i mu_i^(0) + w_norm * mu_norm^(0)
        # d<x0>/dw_j = mu_j^(0) - mu_norm^(0)
        c_norm = model.norm_kernel.centroids.double()
        mu_norm_x0 = c_norm[0, 0]
        mean_x0_fit = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
        grad_parts = []
        for layer in model.layers:
            w_l = layer.coeffs.double()
            c_main = layer.kernel_layer.centroids.double()
            mean_x0_fit = mean_x0_fit + (w_l * c_main[:, 0]).sum()
            grad_parts.append(c_main[:, 0] - mu_norm_x0)
        mean_x0_fit = mean_x0_fit + model.w_norm() * mu_norm_x0
    else:
        # <x0> = sum_l sum_i w_i mu_i^(0) / Z
        # d<x0>/dw_j = (mu_j^(0) - <x0>) / Z
        Z = model.Z()
        mean_x0_fit = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
        for layer in model.layers:
            w_l = layer.coeffs.double()
            c_main = layer.kernel_layer.centroids.double()
            mean_x0_fit = mean_x0_fit + (w_l * c_main[:, 0]).sum()
        mean_x0_fit = mean_x0_fit / (Z + 1e-30)
        grad_parts = []
        for layer in model.layers:
            c_main = layer.kernel_layer.centroids.double()
            grad_parts.append((c_main[:, 0] - mean_x0_fit) / (Z + 1e-30))

    grad_mean_x0 = torch.cat(grad_parts)
    var_mean_x0 = (grad_mean_x0 @ cov_w @ grad_mean_x0).item()
    sigma_mean_x0 = np.sqrt(max(var_mean_x0, 0))

# Use the empirical mean of the full training set as proxy for the true <x0>
# (converges to the true value for large N; works for any benchmark)
true_mean_x0 = data_train[:, 0].mean()
pull_central = (mean_x0_fit.item() - true_mean_x0) / (sigma_mean_x0 + 1e-30)

print(f"  Fitted <x0>  = {mean_x0_fit.item():.6f}")
print(f"  True   <x0>  = {true_mean_x0:.6f}")
print(f"  sigma(<x0>)  = {sigma_mean_x0:.6f}")
print(f"  Central pull = {pull_central:.3f}")

# ══════════════════════════════════════════════════════════════════════
# Bootstrap coverage
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
# Read N_BOOTSTRAP from pipeline config if available
_cfg_path = os.path.join(OUT_DIR, "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
    N_BOOTSTRAP = _cfg.get("N_BOOTSTRAP", 50)
else:
    N_BOOTSTRAP = 50

print(f"Bootstrap coverage ({N_BOOTSTRAP} repetitions)")
print("=" * 60)
pulls = []

for b in range(N_BOOTSTRAP):
    np.random.seed(data_config["seed"] + 1000 + b)
    idx_boot = np.random.choice(N_TRAIN, size=N_TRAIN, replace=True)
    x_boot = x_train[idx_boot]

    # Recompute kernel matrices with bootstrap data at the *same* model
    with torch.no_grad():
        K_list_b, K_norm_b = model.get_kernel_matrices(x_boot)

    # Recompute Hessian at the fitted weights with bootstrap data
    nll_fn_b = make_nll_fn(K_list_b, K_norm_b, layer_sizes)
    w_b = w_hat.clone().requires_grad_(True)
    loss_b = nll_fn_b(w_b)
    H_b = compute_hessian(loss_b, w_b)
    if model.use_norm_kernel:
        cov_b, diag_b = compute_covariance_regularised(H_b, eps=1e-6, verbose=False)
    else:
        cov_b, diag_b = compute_covariance_projected(H_b, eig_floor=1e-6, verbose=False)

    # Uncertainty on <x0> for this bootstrap
    with torch.no_grad():
        var_b = (grad_mean_x0 @ cov_b.double() @ grad_mean_x0).item()
        sigma_b = np.sqrt(max(var_b, 0))

    # Empirical <x0> from bootstrap sample
    empirical_mean = x_boot[:, 0].mean().item()
    pull_b = (empirical_mean - mean_x0_fit.item()) / (sigma_b + 1e-30)
    pulls.append(pull_b)

    if (b + 1) % 10 == 0:
        print(f"  [{b+1}/{N_BOOTSTRAP}] pull = {pull_b:.3f}")

pulls = np.array(pulls)
covered_1sigma = np.sum(np.abs(pulls) < 1.0)
covered_2sigma = np.sum(np.abs(pulls) < 2.0)
coverage_1 = covered_1sigma / N_BOOTSTRAP
coverage_2 = covered_2sigma / N_BOOTSTRAP

print(f"\n  1-sigma coverage: {covered_1sigma}/{N_BOOTSTRAP} = {coverage_1*100:.1f}% (target ~68%)")
print(f"  2-sigma coverage: {covered_2sigma}/{N_BOOTSTRAP} = {coverage_2*100:.1f}% (target ~95%)")
print(f"  Pull mean = {pulls.mean():.3f}, std = {pulls.std():.3f} (target: 0, 1)")

# ── Save results ──────────────────────────────────────────────────────
results = {
    "fitted_mean_x0": mean_x0_fit.item(),
    "true_mean_x0": true_mean_x0,
    "sigma_mean_x0": sigma_mean_x0,
    "central_pull": pull_central,
    "n_bootstrap": N_BOOTSTRAP,
    "coverage_1sigma": coverage_1,
    "coverage_2sigma": coverage_2,
    "pull_mean": float(pulls.mean()),
    "pull_std": float(pulls.std()),
}
with open(os.path.join(OUT_DIR, "coverage_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── Plot pull distribution ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(pulls, bins=15, density=True, alpha=0.6, color="C0", edgecolor="C0", label="Bootstrap pulls")
xp = np.linspace(-4, 4, 200)
from scipy.stats import norm as norm_dist
ax.plot(xp, norm_dist.pdf(xp), "k--", lw=1.5, label="N(0,1)")
ax.axvline(-1, color="gray", ls=":", alpha=0.5)
ax.axvline(1, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Pull = (empirical - fitted) / sigma", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title(f"Coverage test: {coverage_1*100:.0f}% within 1-sigma (target 68%)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "coverage_pulls.png"), dpi=150)
plt.close()
print("  Saved coverage_pulls.png, coverage_results.json")
print("\nDone.")
