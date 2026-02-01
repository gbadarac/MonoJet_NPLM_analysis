#!/usr/bin/env python

import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from torch.autograd import grad
from torch.autograd.functional import hessian
import gc
import mplhep as hep

from Uncertainty_Modeling.wifi.utils_kernel_wifi import plot_ensemble_marginals_2d_kernel, profile_likelihood_scan, plot_final_marginals_and_ratio

# Use CMS style for plots
hep.style.use("CMS")

# -------------------------------------------------------------------
# Point Python to Sparker_utils so we can import the ensemble builder
# -------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent              # .../Uncertainty_Modeling/wifi/Fit_Weights
REPO_ROOT = THIS_DIR.parents[2]                         # .../MonoJet_NPLM_analysis
TRAIN_ENSEMBLES_DIR = REPO_ROOT / "Train_Ensembles"     # .../MonoJet_NPLM_analysis/Train_Ensembles
TRAIN_MODELS_DIR = TRAIN_ENSEMBLES_DIR / "Train_Models" # .../Train_Ensembles/Train_Models
SPARKER_UTILS_DIR = TRAIN_MODELS_DIR / "Sparker_utils"  # .../Train_Models/Sparker_utils

sys.path.insert(1, str(SPARKER_UTILS_DIR))

from kernel_wifi_ensemble_utils import build_wifi_ensemble

# Where to store WiFi–kernel fit results
RESULTS_BASE_DIR = THIS_DIR / "results_fit_weights_kernel"

# -------------------------------------------------------------------
# Hessian and covariance helpers
# -------------------------------------------------------------------
def get_hessian(loss, weights):
    grad1 = grad(loss, weights, create_graph=True, allow_unused=False)[0]
    H_rows = []
    for i in range(weights.numel()):
        grad2 = grad(grad1[i], weights, retain_graph=True, allow_unused=False)[0]
        H_rows.append(grad2)
    return torch.stack(H_rows)

def compute_sandwich_covariance(H, w, model_probs, lam=1e-6):
    """
    H : (M, M) Hessian of NLL wrt weights
    w : (M,) final weights
    model_probs : (N, M) f_j(x_n)
    lam : L2 contribution from norm regularization
    """
    M = len(w)
    N = model_probs.shape[0]

    w = w.detach().clone().double().requires_grad_(False)
    model_probs = model_probs.double()

    # V = H^{-1}
    eye = torch.eye(M, dtype=torch.float64, device=H.device)
    H_reg = H + lam * eye
    V = torch.linalg.solve(H_reg, eye)

    # f(x) = sum_j w_j f_j(x)
    f_raw = (model_probs * w.view(1, -1)).sum(dim=1)  # (N,)
    f = torch.clamp_min(f_raw, 0.0)

    eps = 1e-12
    # gradient of -log(f + eps) wrt w is -(f_j / (f + eps))
    grads = -(model_probs / (f[:, None] + eps))  # (N, M)

    # BUT: because f = clamp_min(f_raw, 0), gradient is zero where f_raw <= 0
    mask = (f_raw > 0).to(dtype=torch.float64, device=H.device)  # (N,)
    grads = grads * mask[:, None]

    mean_grad = grads.mean(dim=0, keepdim=True)
    U = ((grads - mean_grad).T @ (grads - mean_grad)) / N

    cov_w = V @ U @ V.T
    cov_w = cov_w / N
    return cov_w

def compute_sandwich_covariance_no_penalty(H, w, model_probs):
    M = len(w)
    N = model_probs.shape[0]
    eye = torch.eye(M, dtype=torch.float64, device=H.device)
    V = torch.linalg.solve(H, eye)

    f = (model_probs * w.view(1, -1)).sum(dim=1)
    grads = -(model_probs / (f[:, None] + 1e-12))  # (N, M)

    mean_grad = grads.mean(dim=0, keepdim=True)
    U = ((grads - mean_grad).T @ (grads - mean_grad)) / N

    cov_w = V @ U @ V.T
    return cov_w / N

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit WiFi-style ensemble weights for kernel ensemble"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to kernel training output (where config.json & seed*/ live)",
    )
    parser.add_argument(
        "--n_wifi_components",
        type=int,
        default=10,
        help="Number of ensemble members (seeds) to include",
    )
    parser.add_argument(
        "--epochs", type=int, default=2000, help="Max number of training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Logging & early-stopping patience (in epochs)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-1, help="Learning rate for Adam"
    )
    
    parser.add_argument(
        "--compute_covariance",
        action="store_true",
        help="If set, compute Hessian + sandwich covariance",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="If set, do not save any plots (weights & cov only)",
    )
    parser.add_argument(
        "--seed_bootstrap",
        type=int,
        default=1234,
        help="Seed for bootstrap resampling",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to .npy file with target data of shape (N, d)",
    )

    parser.add_argument(
        "--lambda_norm",
        type=float,
        default=1.0,
        help="Weight-sum penalty strength, notebook uses 1",
    )

    args = parser.parse_args()

    folder_path = Path(args.folder_path).resolve()
    print(f"Using folder_path = {folder_path}")

    # --------------------------------------------------------------
    # Build a NF-style results directory name
    # Example:
    #   folder_path.name      -> N_100000_dim_2_kernels_Soft-SparKer2_M100_Nboot100000_lr0.01
    #   folder_path.parent.name -> 2d_bimodal_gaussian_heavy_tail
    # Final results dir:
    #   results_fit_weights_kernel/N_100000_dim_2_kernels_..._2d_bimodal_gaussian_heavy_tail
    # --------------------------------------------------------------
    trial_name   = folder_path.name
    dataset_tag  = folder_path.parent.name
    results_dir  = RESULTS_BASE_DIR / f"{trial_name}_{dataset_tag}_latest_Sean_corrections_and_no_penalty_only_minimizer"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Storing WiFi fit results in: {results_dir}")

    # Output directory for plots & diagnostics
    plots_dir = results_dir / "wifi_ensemble_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load target dataset from disk
    # ------------------------------------------------------------------
    data_file = args.data_path
    data_train_tot = np.load(data_file).astype("float32")   # shape (N, d)
    print("Loaded target data:", data_train_tot.shape)

    # ------------------------------------------------------------------
    # Build ensemble
    # ------------------------------------------------------------------
    ensemble, config_json = build_wifi_ensemble(
        folder_path=folder_path,
        n_wifi_components=args.n_wifi_components,
        lambda_norm=args.lambda_norm,
        train_centroids=False,
        train_coeffs=False,
        train_widths=False,
        train_weights=True,
        weights_activation=None,
    )

    # ------------------------------------------------------------
    # 1) Keep SparKer on CPU for evaluation (no SPARKutils changes)
    # ------------------------------------------------------------
    device_kernel = torch.device("cpu")
    ensemble = ensemble.to(device=device_kernel, dtype=torch.float64)

    N_train = int(config_json["N"])
    rng = np.random.default_rng(args.seed_bootstrap)
    idx = rng.integers(0, len(data_train_tot), size=N_train)
    bootstrap_np = data_train_tot[idx]
    bootstrap_sample = torch.from_numpy(bootstrap_np).to(device=device_kernel, dtype=torch.float64)
    bootstrap_sample_cpu = bootstrap_sample.detach().cpu().numpy()

    # ------------------------------------------------------------
    # 2) Precompute model_probs ONCE on CPU  (N, M)
    # ------------------------------------------------------------
    print("Precomputing model_probs on CPU ...", flush=True)
    with torch.no_grad():
        model_probs_cpu = ensemble.member_probs(bootstrap_sample).to(dtype=torch.float64, device="cpu").contiguous()

        #----------------------------------------------------------------------
        # --- critical: member "densities" must be nonnegative for a likelihood
        model_probs_cpu = torch.clamp_min(model_probs_cpu, 0.0)

        print(
            "model_probs stats:",
            "min", float(model_probs_cpu.min()),
            "max", float(model_probs_cpu.max()),
            "finite", bool(torch.isfinite(model_probs_cpu).all()),
            flush=True
        )
        #----------------------------------------------------------------------
        
    print(f"model_probs_cpu computed: shape={tuple(model_probs_cpu.shape)}", flush=True)

    # ------------------------------------------------------------
    # 3) Optimize weights on GPU (fast), sum-to-one enforced by construction
    #    (no penalty, no positivity constraint on weights)
    # ------------------------------------------------------------
    device_fit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_probs = model_probs_cpu.to(device=device_fit, dtype=torch.float64, non_blocking=True)

    M = model_probs.shape[1]

    # Initialise from current ensemble weights + small noise
    w_init = ensemble.weights.detach().to(device=device_fit, dtype=torch.float64).clone()
    w_init = w_init + 1e-2 * torch.randn_like(w_init)
    
    print("w_init:", w_init.detach().cpu().numpy(), flush=True)
    print("w_init sum:", float(w_init.sum().detach().cpu().item()), flush=True)

    # Free parameters: u = first M-1 weights
    u0 = w_init[:-1].clone()
    u = torch.nn.Parameter(u0, requires_grad=True)

    opt = torch.optim.Adam([u], lr=args.lr)

    def build_full_weights(u_vec: torch.Tensor) -> torch.Tensor:
        # Enforce sum(w)=1 by defining the last weight as 1 - sum(first M-1)
        w_last = 1.0 - u_vec.sum()
        return torch.cat([u_vec, w_last.view(1)], dim=0)  # (M,)

    def nll_from_probs(u_vec: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        w = build_full_weights(u_vec)  # (M,)

        # raw mixture (can be negative because weights can be negative)
        p_raw = (model_probs * w.view(1, -1)).sum(dim=1)  # (N,)

        # IMPORTANT: keep the density positive and differentiable
        # This does NOT constrain weights to be positive.
        p = F.softplus(p_raw) + eps

        return -torch.log(p).mean()

    loss_hist = []
    best = float("inf")
    bad = 0
    log_every = max(1, args.patience)

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = nll_from_probs(u)
        loss.backward()
        opt.step()

        if epoch % log_every == 0:
            with torch.no_grad():
                loss_after = nll_from_probs(u)
                cur = float(loss_after.detach().cpu().item())

            # gradient norm you print is the grad from the last backward (pre step)
            gnorm = float(u.grad.detach().norm().cpu().item())
            print(f"epoch {epoch} loss {cur:.6e} |dL/du| {gnorm:.3e}", flush=True)

            w_sum = float(build_full_weights(u.detach()).sum().detach().cpu().item())
            print(f"epoch {epoch} sumw {w_sum:.6f}", flush=True)
            
            w_now = build_full_weights(u.detach()).cpu().numpy()
            print(f"epoch {epoch} weights {w_now}", flush=True)

            loss_hist.append(cur)

            if cur < best:
                best = cur
                bad = 0
            else:
                bad += 1
                if bad >= log_every:
                    print(f"early stopping at epoch {epoch} best {best:.6f}", flush=True)
                    break

    with torch.no_grad():
        final_weights = build_full_weights(u.detach()).cpu().numpy()

    np.save(results_dir / "final_weights.npy", final_weights)
    np.save(results_dir / "loss_history_wifi.npy", np.array(loss_hist, dtype=np.float32))
    print("Saved final_weights.npy and loss_history_wifi.npy", flush=True)

    # copy weights back into ensemble (CPU) for any later plotting code that expects ensemble.weights
    with torch.no_grad():
        ensemble.weights.copy_(
            torch.from_numpy(final_weights).to(device=device_kernel, dtype=ensemble.weights.dtype)
        )
    # ------------------------------------------------------------------
    # Hessian + sandwich covariance (same nll as optimisation)
    # ------------------------------------------------------------------
    if args.compute_covariance:
        lambda_norm = float(args.lambda_norm)

        # reuse precomputed model_probs_cpu
        model_probs = model_probs_cpu  # (N, M) on CPU float64

        w_for_hess = torch.from_numpy(final_weights).to(dtype=torch.float64, device="cpu").clone()
        w_for_hess.requires_grad_(True)

        def nll_w(w):
            eps = 1e-12

            p_raw = (model_probs * w.view(1, -1)).sum(dim=1)
            p = torch.clamp_min(p_raw, 0.0)

            data_term = -torch.log(p + eps).mean()

            s = w.sum()
            constraint_term = lambda_norm * (s - 1.0)

            return data_term + constraint_term

        y = nll_w(w_for_hess)
        H = get_hessian(y, w_for_hess).detach()
        np.save(results_dir / "Hessian_weights.npy", H.numpy())

        w_final = w_for_hess.detach()
        cov_w = compute_sandwich_covariance(H, w_final, model_probs, lam=1.0)
        np.save(results_dir / "cov_weights.npy", cov_w.detach().numpy())

    # ------------------------------------------------------------------
    # Build grid & evaluate final ensemble for plots
    # ------------------------------------------------------------------
    device_plot = torch.device("cpu")

    if not args.no_plots:
        pad = 0.05

        x0_lo, x0_hi = bootstrap_sample_cpu[:, 0].min(), bootstrap_sample_cpu[:, 0].max()
        x1_lo, x1_hi = bootstrap_sample_cpu[:, 1].min(), bootstrap_sample_cpu[:, 1].max()

        x0_pad = pad * (x0_hi - x0_lo + 1e-12)
        x1_pad = pad * (x1_hi - x1_lo + 1e-12)

        x0 = torch.arange(x0_lo - x0_pad, x0_hi + x0_pad + 0.5*0.01, 0.01, device=device_plot, dtype=torch.float64)
        x1 = torch.arange(x1_lo - x1_pad, x1_hi + x1_pad + 0.5*0.005, 0.005, device=device_plot, dtype=torch.float64)

        X0, X1 = torch.meshgrid(x0, x1, indexing="xy")
        grid = torch.stack([X0.flatten(), X1.flatten()], dim=1)

        with torch.no_grad():
            Y = ensemble(grid) 
        cov_w_np = np.load(results_dir / "cov_weights.npy") if args.compute_covariance else None

        # Marginals + ratio
        plot_final_marginals_and_ratio(
            ensemble,
            bootstrap_sample,
            grid,
            x0,
            x1,
            Y,
            plots_dir,
            tag="final",
        )

        # 2D heatmap of ensemble density
        fig = plt.figure()
        sc = plt.scatter(
            grid[:, 0].cpu().numpy(),
            grid[:, 1].cpu().numpy(),
            c=Y.cpu().numpy(),
            edgecolors="none",
            s=1,
        )
        plt.colorbar(sc)
        plt.xlim(-2, 3)
        plt.ylim(-1, 2)
        fig.tight_layout()
        fig.savefig(plots_dir / "ensemble_heatmap.png", dpi=200)
        plt.close(fig)

        # 2D histogram of data
        fig = plt.figure()
        x0_h = torch.arange(-2.0, 3.0, 0.09).double().numpy()
        x1_h = torch.arange(-1.0, 2.0, 0.09).double().numpy()
        plt.hist2d(
            bootstrap_sample[:, 0].cpu().numpy(),
            bootstrap_sample[:, 1].cpu().numpy(),
            bins=[x0_h, x1_h],
            density=True,
        )
        plt.colorbar()
        fig.tight_layout()
        fig.savefig(plots_dir / "data_hist2d.png", dpi=200)
        plt.close(fig)

        feature_names = ["Feature 1", "Feature 2"]
        
        plot_ensemble_marginals_2d_kernel(
            kernel_models=ensemble.ensemble,
            x_data=bootstrap_sample.detach().cpu(),                  # (N,2)
            weights=ensemble.weights.detach().cpu(),         # (M,)
            cov_w=cov_w_np,                         # (M,M) or None
            feature_names=feature_names,
            outdir=str(plots_dir),
            bins=40,
        )

    print("All done.")

if __name__ == "__main__":
    main()
