"""
Script 2: Train HIKER, compute Hessian-based covariance.

Supports ensemble mode: when N_ENSEMBLE > 1, trains N models on bootstrap
resamples, combines them into an averaged ensemble, and builds a block-diagonal
covariance. When N_ENSEMBLE = 1 (default), behaves as before.

Reads: {OUT_DIR}/pipeline_config.json, data_config.json, data_train.npy
Saves: model.pt, covariance.npy, hessian.npy, marginals.npz, train_config.json,
       training_loss.png, training_coeffs.png, training_centroids.png,
       eigenspectrum.png
       (ensemble: seed{i}/ subdirectories + ensemble model/covariance)
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hiker import (HIKER, build_hiker_from_config, train_hiker,
                   build_ensemble_model, build_ensemble_covariance,
                   ensemble_density)
from hessian_uq import (
    compute_hessian, make_nll_fn,
    compute_covariance_regularised, compute_covariance_projected,
    predictive_variance, predictive_variance_ensemble,
)

# ── Output directory from environment ─────────────────────────────────
OUT_DIR = os.environ.get("HIKER_OUT_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))

# ── Load pipeline config (or defaults) ────────────────────────────────
cfg_path = os.path.join(OUT_DIR, "pipeline_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
else:
    cfg = {}

LR = cfg.get("LR", 1e-4)
BATCH_SIZE = cfg.get("BATCH_SIZE", None)
TRAIN_MODE = cfg.get("TRAIN_MODE", "sequential")
EARLY_STOPPING = cfg.get("EARLY_STOPPING", None)
N_ENSEMBLE = cfg.get("N_ENSEMBLE", 1)
N_TRAIN_PER_MODEL = cfg.get("N_TRAIN_PER_MODEL", None)  # None = use full dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ─────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
print("=" * 60)
with open(os.path.join(OUT_DIR, "data_config.json")) as f:
    data_config = json.load(f)
DATA_DIR = data_config.get("data_dir", OUT_DIR)
BENCHMARK = data_config.get("benchmark", "2d_gaussian")
data_train = np.load(os.path.join(DATA_DIR, "data_train.npy"))
N_TRAIN = data_train.shape[0]
d = data_train.shape[1]

x_train_full = torch.from_numpy(data_train).double().to(DEVICE)
print(f"  Benchmark = {BENCHMARK}, N_train = {N_TRAIN}, d = {d}")
print(f"  N_ENSEMBLE = {N_ENSEMBLE}")


# ══════════════════════════════════════════════════════════════════════
# Helper: train one model, compute Hessian, save to a directory
# ══════════════════════════════════════════════════════════════════════

def train_single_model(x_train, seed_idx, save_dir):
    """Train one HIKER model and compute its Hessian covariance.

    Returns (model, cov_w, diag).
    """
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(42 + seed_idx)
    model, layer_train_configs = build_hiker_from_config(x_train, cfg)
    model = model.to(DEVICE)

    print(f"\n  {model.n_layers} layer(s):")
    for l, layer in enumerate(model.layers):
        print(f"    Layer {l}: M={layer.M}, sigma={layer.kernel_layer.sigma:.4f}")

    loss_hist, coeffs_hist, centroids_hist = train_hiker(
        model, x_train, layer_train_configs, lr=LR,
        batch_size=BATCH_SIZE, mode=TRAIN_MODE,
        early_stopping_patience=EARLY_STOPPING, verbose=True)

    # ── Training plots ────────────────────────────────────────────
    _plot_training(loss_hist, coeffs_hist, centroids_hist, model, save_dir)

    # ── Check weights ─────────────────────────────────────────────
    with torch.no_grad():
        for l, layer in enumerate(model.layers):
            w = layer.coeffs.cpu().numpy()
            print(f"  Layer {l}: sum(w)={w.sum():.4f}, n_neg={int((w < 0).sum())}")
        if model.use_norm_kernel:
            print(f"  Global w_norm = {model.w_norm().item():.4f}")
        else:
            print(f"  Z = sum(w) = {model.Z().item():.4f}")

    # ── Save model ────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # ── Hessian + covariance ──────────────────────────────────────
    print(f"\n  Computing Hessian...")
    layer_sizes = [layer.M for layer in model.layers]
    with torch.no_grad():
        K_list, K_norm = model.get_kernel_matrices(x_train)
        w_hat = model.get_free_coeffs().detach().clone()

    M_total = w_hat.numel()
    print(f"  M_total = {M_total}")

    nll_fn = make_nll_fn(K_list, K_norm, layer_sizes)
    w_for_hess = w_hat.clone().requires_grad_(True)
    loss = nll_fn(w_for_hess)
    H = compute_hessian(loss, w_for_hess)

    w_check = w_hat.clone().requires_grad_(True)
    grad_at_opt = torch.autograd.grad(nll_fn(w_check), w_check)[0]
    print(f"  |grad| at optimum: {grad_at_opt.norm().item():.4e}")

    print("  Inverting Hessian...")
    if model.use_norm_kernel:
        cov_w, diag = compute_covariance_regularised(H, eps=1e-6, verbose=True)
    else:
        cov_w, diag = compute_covariance_projected(H, eig_floor=1e-6, verbose=True)

    np.save(os.path.join(save_dir, "hessian.npy"), H.cpu().numpy())
    np.save(os.path.join(save_dir, "covariance.npy"), cov_w.cpu().numpy())
    np.save(os.path.join(save_dir, "hessian_eigenvalues.npy"), diag["eigenvalues"])

    # Eigenspectrum plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.semilogy(np.abs(diag["eigenvalues"]), "o-", ms=5)
    ax.set_xlabel("Index", fontsize=13)
    ax.set_ylabel("|Eigenvalue|", fontsize=13)
    ax.set_title("Hessian eigenspectrum", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eigenspectrum.png"), dpi=150)
    plt.close()

    return model, cov_w, diag


def _plot_training(loss_hist, coeffs_hist, centroids_hist, model, save_dir):
    """Produce training diagnostic plots."""
    layer_sizes = [layer.M for layer in model.layers]
    layer_colors = plt.cm.tab10(np.linspace(0, 1, max(len(layer_sizes), 1)))

    # Loss
    if loss_hist:
        fig, ax = plt.subplots(figsize=(8, 4))
        layers_in_hist = sorted(set(lyr for lyr, _, _ in loss_hist))
        for lyr in layers_in_hist:
            epochs_l = [ep for l, ep, _ in loss_hist if l == lyr]
            losses_l = [v for l, _, v in loss_hist if l == lyr]
            label = "joint" if lyr == -1 else f"layer {lyr}"
            ax.plot(epochs_l, losses_l, "o-", ms=3, lw=0.8, label=label)

        all_epochs_loss = np.array([ep for _, ep, _ in loss_hist])
        all_losses = np.array([v for _, _, v in loss_hist])
        n_fit = min(20, len(all_losses))
        if n_fit >= 3:
            x_fit = all_epochs_loss[-n_fit:]
            y_fit = all_losses[-n_fit:]
            coeffs_fit = np.polyfit(x_fit, y_fit, 1)
            b_slope, a_intercept = coeffs_fit
            y_pred = np.polyval(coeffs_fit, x_fit)
            residuals = y_fit - y_pred
            ndf = n_fit - 2
            if n_fit >= 6 and ndf > 0:
                half = n_fit // 2
                sigma_est = np.std(y_fit[:half] - np.polyval(
                    np.polyfit(x_fit[:half], y_fit[:half], 1), x_fit[:half]))
                if sigma_est > 0:
                    chi2_ndf = np.sum((residuals / sigma_est) ** 2) / ndf
                else:
                    chi2_ndf = 0.0
            else:
                chi2_ndf = 1.0
            x_line = np.linspace(x_fit[0], x_fit[-1], 100)
            ax.plot(x_line, np.polyval(coeffs_fit, x_line), "r--", lw=1.5,
                    label=f"Linear fit (last {n_fit} pts)\n"
                          f"  slope = {b_slope:.2e}\n"
                          f"  $\\chi^2$/ndf = {chi2_ndf:.2f}")

        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("NLL/N", fontsize=13)
        ax.set_title("Training loss", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150)
        plt.close()

    # Coefficients
    if coeffs_hist:
        all_epochs = [ep for _, ep, _ in coeffs_hist]
        all_coeffs = np.array([c for _, _, c in coeffs_hist])
        fig, ax = plt.subplots(figsize=(10, 5))
        offset = 0
        for l, M_l in enumerate(layer_sizes):
            for k in range(M_l):
                ax.plot(all_epochs, all_coeffs[:, offset + k],
                        lw=0.5, alpha=0.6, color=layer_colors[l],
                        label=f"Layer {l}" if k == 0 else None)
            offset += M_l
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Coefficient value", fontsize=13)
        ax.set_title("Coefficient evolution", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_coeffs.png"), dpi=150)
        plt.close()

        # Normalized coefficients (w_i / Z)
        Z_per_step = all_coeffs.sum(axis=1, keepdims=True)  # (n_steps, 1)
        Z_per_step = np.where(np.abs(Z_per_step) > 1e-30, Z_per_step, 1e-30)
        all_coeffs_norm = all_coeffs / Z_per_step

        fig, ax = plt.subplots(figsize=(10, 5))
        offset = 0
        for l, M_l in enumerate(layer_sizes):
            for k in range(M_l):
                ax.plot(all_epochs, all_coeffs_norm[:, offset + k],
                        lw=0.5, alpha=0.6, color=layer_colors[l],
                        label=f"Layer {l}" if k == 0 else None)
            offset += M_l
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("$w_i / Z$", fontsize=13)
        ax.set_title("Normalized coefficient evolution ($w_i / \\Sigma w$)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_coeffs_normalized.png"), dpi=150)
        plt.close()

    # Centroids
    if centroids_hist:
        all_epochs_c = [ep for _, ep, _ in centroids_hist]
        all_c = np.array([c for _, _, c in centroids_hist])
        d_dim = all_c.shape[2]
        fig, axes = plt.subplots(1, d_dim, figsize=(6 * d_dim, 5), squeeze=False)
        for dim in range(d_dim):
            ax = axes[0, dim]
            offset = 0
            for l, M_l in enumerate(layer_sizes):
                for k in range(M_l):
                    ax.plot(all_epochs_c, all_c[:, offset + k, dim],
                            lw=0.5, alpha=0.6, color=layer_colors[l],
                            label=f"Layer {l}" if k == 0 else None)
                offset += M_l
            ax.set_xlabel("Epoch", fontsize=13)
            ax.set_ylabel(rf"$\mu_{{x_{dim}}}$", fontsize=13)
            ax.set_title(rf"Centroid $x_{dim}$ evolution", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_centroids.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# Main: single model or ensemble
# ══════════════════════════════════════════════════════════════════════

if N_ENSEMBLE == 1:
    # ── Single model (original behavior) ──────────────────────────
    print("\n" + "=" * 60)
    print("Training single HIKER model")
    print("=" * 60)

    if N_TRAIN_PER_MODEL is not None and N_TRAIN_PER_MODEL < N_TRAIN:
        np.random.seed(cfg.get("seed", 42))
        idx = np.random.choice(N_TRAIN, size=N_TRAIN_PER_MODEL, replace=False)
        x_train_use = x_train_full[idx]
        print(f"  Using {N_TRAIN_PER_MODEL}/{N_TRAIN} training points")
    else:
        x_train_use = x_train_full
    model, cov_w, diag = train_single_model(x_train_use, seed_idx=0, save_dir=OUT_DIR)
    ensemble_info = None

else:
    # ── Ensemble of N models on bootstrap resamples ───────────────
    print("\n" + "=" * 60)
    print(f"Training ensemble of {N_ENSEMBLE} HIKER models")
    print("=" * 60)

    models = []
    covariances = []

    for s in range(N_ENSEMBLE):
        print(f"\n{'#' * 60}")
        print(f"Seed {s}/{N_ENSEMBLE - 1}")
        print(f"{'#' * 60}")

        # Bootstrap resample
        n_per_model = N_TRAIN_PER_MODEL if N_TRAIN_PER_MODEL is not None else N_TRAIN
        np.random.seed(cfg.get("seed", 42) + s)
        idx = np.random.choice(N_TRAIN, size=n_per_model, replace=True)
        x_boot = x_train_full[idx]
        print(f"  Bootstrap: {n_per_model} points (from {N_TRAIN} total)")

        seed_dir = os.path.join(OUT_DIR, f"seed{s}")
        m, cov_s, diag_s = train_single_model(x_boot, seed_idx=s, save_dir=seed_dir)
        models.append(m)
        covariances.append(cov_s)

    # ── Build ensemble ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building ensemble model")
    print("=" * 60)

    model, ensemble_info = build_ensemble_model(models, device=DEVICE)
    cov_w = build_ensemble_covariance(models, covariances, device=DEVICE)

    print(f"  Ensemble layers: {model.n_layers}")
    print(f"  Ensemble M_total: {sum(l.M for l in model.layers)}")
    print(f"  Ensemble covariance shape: {cov_w.shape}")
    print(f"  Per-seed norm kernels: {ensemble_info['use_norm_per_seed']}")
    print(f"  Seed boundaries: {ensemble_info['seed_boundaries']}")

    # Verify density integrates correctly
    with torch.no_grad():
        from hiker import ensemble_density
        test_pts = x_train_full[:1000]
        f_test = ensemble_density(model, test_pts, ensemble_info)
        print(f"  Density check: min={f_test.min().item():.4e}, "
              f"mean={f_test.mean().item():.4e}, max={f_test.max().item():.4e}")

    # Save ensemble model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
    np.save(os.path.join(OUT_DIR, "covariance.npy"), cov_w.cpu().numpy())

    # Save ensemble info (norm kernels + seed boundaries)
    ens_save = {
        "seed_boundaries": ensemble_info["seed_boundaries"],
        "N_ens": ensemble_info["N_ens"],
        "use_norm_per_seed": ensemble_info["use_norm_per_seed"],
    }
    if ensemble_info["norm_kernels"] is not None:
        for s, nk in enumerate(ensemble_info["norm_kernels"]):
            ens_save[f"norm_kernel_{s}_centroids"] = nk.centroids.detach().cpu().numpy()
            ens_save[f"norm_kernel_{s}_width"] = nk.width.detach().cpu().numpy()
    np.savez(os.path.join(OUT_DIR, "ensemble_info.npz"), **ens_save)
    print("  Saved ensemble_info.npz")

    diag = {"eigenvalues": np.linalg.eigvalsh(cov_w.cpu().numpy())}

    # Eigenspectrum of ensemble covariance
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    eigs = np.sort(np.abs(diag["eigenvalues"]))[::-1]
    ax.semilogy(eigs, "o-", ms=3)
    ax.set_xlabel("Index", fontsize=13)
    ax.set_ylabel("|Eigenvalue|", fontsize=13)
    ax.set_title(f"Ensemble covariance eigenspectrum (N={N_ENSEMBLE})", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eigenspectrum.png"), dpi=150)
    plt.close()

# ── Save training config ─────────────────────────────────────────────
layer_sizes = [layer.M for layer in model.layers]
train_config = dict(cfg)
train_config["layer_sizes"] = layer_sizes
train_config["n_layers"] = model.n_layers
train_config["N_ENSEMBLE"] = N_ENSEMBLE
if N_ENSEMBLE > 1:
    # The ensemble HIKER container has use_norm_kernel=False,
    # but per-seed norm kernels are stored in ensemble_info.npz
    train_config["USE_NORM_KERNEL"] = False
    train_config["USE_NORM_PER_SEED"] = cfg.get("USE_NORM_KERNEL", True)
with open(os.path.join(OUT_DIR, "train_config.json"), "w") as f:
    json.dump(train_config, f, indent=2)
print("  Saved train_config.json")

# ══════════════════════════════════════════════════════════════════════
# Precompute marginals
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Precomputing marginal densities with uncertainty")
print("=" * 60)

from benchmarks import get_marginals

n_grid = 300
n_int = 200
margin = 0.1

grids = []
int_grids = []
for dim in range(d):
    lo = data_train[:, dim].min()
    hi = data_train[:, dim].max()
    pad = margin * (hi - lo)
    grids.append(np.linspace(lo - pad, hi + pad, n_grid))
    int_grids.append(np.linspace(lo - pad, hi + pad, n_int))

assert d == 2, f"Marginal precomputation currently supports d=2, got d={d}"

# Compute ensemble/single marginals
f_marginals = []
sigma_marginals = []

for dim in range(d):
    other = 1 - dim
    x_eval_grid = grids[dim]
    x_int = int_grids[other]
    dx = x_int[1] - x_int[0]

    print(f"  Computing marginal {dim}...")
    f_marg = np.zeros(n_grid)
    var_marg = np.zeros(n_grid)
    for i, xv in enumerate(x_eval_grid):
        if dim == 0:
            pts = torch.tensor(np.column_stack([np.full(n_int, xv), x_int]),
                               dtype=torch.float64).to(DEVICE)
        else:
            pts = torch.tensor(np.column_stack([x_int, np.full(n_int, xv)]),
                               dtype=torch.float64).to(DEVICE)
        if ensemble_info is not None:
            var_f, f_eval = predictive_variance_ensemble(pts, model, cov_w, ensemble_info)
        else:
            var_f, f_eval = predictive_variance(pts, model, cov_w)
        f_marg[i] = f_eval.sum().item() * dx
        var_marg[i] = var_f.sum().item() * dx ** 2

    f_marginals.append(f_marg)
    sigma_marginals.append(np.sqrt(np.maximum(var_marg, 0)))

# Also compute per-seed marginals (ensemble only, for comparison plot)
seed_marginals = {}
if N_ENSEMBLE > 1:
    print("  Computing per-seed marginals...")
    for s, m_s in enumerate(models):
        seed_marginals[s] = {}
        for dim in range(d):
            other = 1 - dim
            x_int = int_grids[other]
            dx = x_int[1] - x_int[0]
            f_marg_s = np.zeros(n_grid)
            for i, xv in enumerate(grids[dim]):
                if dim == 0:
                    pts = torch.tensor(np.column_stack([np.full(n_int, xv), x_int]),
                                       dtype=torch.float64).to(DEVICE)
                else:
                    pts = torch.tensor(np.column_stack([x_int, np.full(n_int, xv)]),
                                       dtype=torch.float64).to(DEVICE)
                with torch.no_grad():
                    f_marg_s[i] = m_s.density(pts).sum().item() * dx
            seed_marginals[s][dim] = f_marg_s

# True marginals
marginals_fn = get_marginals(BENCHMARK)
if marginals_fn is not None:
    true_marginals = marginals_fn(grids)
    print("  Analytic true marginals available.")
else:
    true_marginals = [None] * d

# Save
save_dict = {"d": d, "benchmark": BENCHMARK, "N_ENSEMBLE": N_ENSEMBLE}
for dim in range(d):
    save_dict[f"x{dim}_grid"] = grids[dim]
    save_dict[f"f_marg{dim}"] = f_marginals[dim]
    save_dict[f"sigma_marg{dim}"] = sigma_marginals[dim]
    if true_marginals[dim] is not None:
        save_dict[f"true_marg{dim}"] = true_marginals[dim]
    # Per-seed marginals
    for s in seed_marginals:
        save_dict[f"f_marg{dim}_seed{s}"] = seed_marginals[s][dim]

np.savez(os.path.join(OUT_DIR, "marginals.npz"), **save_dict)
print("  Saved marginals.npz")

print("\nDone.")
