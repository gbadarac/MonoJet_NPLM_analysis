#!/usr/bin/env python

import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib as mpl
mpl.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from torchmin import minimize
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
    from torch.autograd import grad as torch_grad

    grad1 = torch_grad(loss, weights, create_graph=True, allow_unused=False)[0]
    H_rows = []
    for i in range(len(weights)):
        grad2 = torch_grad(grad1[i], weights, retain_graph=True)[0]
        H_rows.append(grad2)
    return torch.stack(H_rows)

def nll(x, weights, ensemble):

    """
    NLL for fixed kernel components and free weights, matching NF WiFi:
        -log p(x; w).mean() + lambda_norm * (sum w_i - 1)
    """
    # model_probs(x) for each ensemble member -> (N, M)
    outs = torch.stack(
        [m.call(x)[-1, :, 0] / m.get_norm()[-1] for m in ensemble.ensemble], dim=1
    )  # (N, M)
    weighted = outs * weights.view(1, -1)
    p = weighted.sum(dim=1)  # (N,)
    lambda_norm=1.0
    reg = lambda_norm * (weights.sum() - 1.0)
    return -torch.log(p + 1e-12).mean() + reg

def compute_sandwich_covariance(H, w, model_probs, lam=1.0):
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
    V = torch.linalg.solve(H, eye)

    # f(x) = sum_j w_j f_j(x)
    f = (model_probs * w.view(1, -1)).sum(dim=1)  # (N,)

    grads = torch.zeros((N, M), dtype=torch.float64, device=H.device)
    for j in range(M):
        f_j = model_probs[:, j]
        grads[:, j] = -(f_j / (f + 1e-12)) + lam

    mean_grad = grads.mean(dim=0, keepdim=True)
    U = ((grads - mean_grad).T @ (grads - mean_grad)) / N

    cov_w = V @ U @ V.T
    cov_w = cov_w / N
    return cov_w

def compute_sandwich_covariance_no_penalty(H, w, model_probs):
    M = len(w)
    N = model_probs.shape[0]
    eye = torch.eye(M, dtype=torch.float64)
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
        default=1000.0,
        help="Weight-sum penalty strength, notebook uses 1000",
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
    results_dir  = RESULTS_BASE_DIR / f"{trial_name}_{dataset_tag}"
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
    # Build ensemble (shared logic)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = ensemble.to(device=device, dtype=torch.float64)

    N_train = config_json["N"]
    print("Training size N from config:", N_train)

    print("Trainable parameters in ensemble:")
    ensemble.count_trainable_parameters(verbose=True)

    # ------------------------------------------------------------------
    # Use full dataset (no bootstrap, like NF WiFi)
    # ------------------------------------------------------------------
    # Bootstrap sample, matches supervisor notebook
    N_train = int(config_json["N"])  # bootstrap size used in training
    rng = np.random.default_rng(args.seed_bootstrap)
    idx = rng.integers(0, len(data_train_tot), size=N_train)
    bootstrap_np = data_train_tot[idx]  # (N_train, d)

    bootstrap_sample = torch.from_numpy(bootstrap_np).to(device=device, dtype=torch.float64)

    # Make sure ensemble is on same device and in float64
    ensemble = ensemble.to(device=device, dtype=torch.float64)

    opt = torch.optim.Adam(ensemble.parameters(), lr=args.lr)

    loss_hist = []
    best = float("inf")
    bad = 0

    log_every = max(1, args.patience)  # notebook prints every `patience`

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)

        # This is the core change: use the Ensemble.loss exactly like notebook
        loss = ensemble.loss(bootstrap_sample)

        loss.backward()
        opt.step()

        if epoch % log_every == 0:
            cur = float(loss.detach().cpu().item())
            w = ensemble.weights.detach().cpu().numpy()
            print(f"epoch {epoch} loss {cur:.6f} weights {w} sumw {w.sum():.6f}", flush=True)
            loss_hist.append(cur)

            tol = getattr(ensemble, "tol", 0.0)
            if cur + tol < best:
                best = cur
                bad = 0
            else:
                bad += 1
                if bad >= log_every:
                    print(f"early stopping at epoch {epoch} best {best:.6f}", flush=True)
                    break

    final_weights = ensemble.weights.detach().cpu().numpy()
    np.save(results_dir / "final_weights.npy", final_weights)
    np.save(results_dir / "loss_history_wifi.npy", np.array(loss_hist, dtype=np.float32))
    print("Saved final_weights.npy and loss_history_wifi.npy")

    # ------------------------------------------------------------------
    # Hessian + sandwich covariance (same nll as optimisation)
    # ------------------------------------------------------------------
    if args.compute_covariance:
        with torch.no_grad():
            model_probs = torch.stack(
                [m.call(bootstrap_sample)[-1, :, 0] / m.get_norm()[-1] for m in ensemble.ensemble],
                dim=1,
            ).detach().cpu().to(dtype=torch.float64)  # (N, M)

        w_final = ensemble.weights.detach().cpu().to(dtype=torch.float64)

        lambda_norm = float(args.lambda_norm)

        def nll_w(w):
            p = (model_probs * w.view(1, -1)).sum(dim=1)
            return -torch.log(p + 1e-12).sum() + lambda_norm * (w.sum() - 1.0) ** 2

        w_for_hess = w_final.detach().clone().requires_grad_(True)
        H = hessian(nll_w, w_for_hess).detach()
        np.save(results_dir / "Hessian_weights.npy", H.numpy())

        # sandwich covariance: event-wise grads WITHOUT penalty term
        cov_w = compute_sandwich_covariance_no_penalty(H, w_final, model_probs)
        cov_w_np = cov_w.detach().cpu().numpy()
        np.save(results_dir / "cov_weights.npy", cov_w_np)

    # ------------------------------------------------------------------
    # Build grid & evaluate final ensemble for plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        bootstrap_sample_cpu = bootstrap_sample.detach().cpu().numpy()
        pad = 0.05

        x0_lo, x0_hi = bootstrap_sample_cpu[:, 0].min(), bootstrap_sample_cpu[:, 0].max()
        x1_lo, x1_hi = bootstrap_sample_cpu[:, 1].min(), bootstrap_sample_cpu[:, 1].max()

        x0_pad = pad * (x0_hi - x0_lo + 1e-12)
        x1_pad = pad * (x1_hi - x1_lo + 1e-12)

        x0 = torch.arange(x0_lo - x0_pad, x0_hi + x0_pad + 0.5*0.01, 0.01, device=device, dtype=torch.float64)
        x1 = torch.arange(x1_lo - x1_pad, x1_hi + x1_pad + 0.5*0.005, 0.005, device=device, dtype=torch.float64)

        X0, X1 = torch.meshgrid(x0, x1, indexing="xy")
        grid = torch.stack([X0.flatten(), X1.flatten()], dim=1)

        with torch.no_grad():
            Y = ensemble(grid) / ensemble.weights.sum()

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
            x_data=bootstrap_sample.float(),                  # (N,2)
            weights=ensemble.weights.detach().cpu(),         # (M,)
            cov_w=cov_w_np,                         # (M,M) or None
            feature_names=feature_names,
            outdir=str(plots_dir),
            bins=40,
        )

    print("All done.")

if __name__ == "__main__":
    main()
