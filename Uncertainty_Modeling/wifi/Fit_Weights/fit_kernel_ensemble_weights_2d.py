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
        lambda_norm=1.0,
        train_centroids=False,
        train_coeffs=False,
        train_widths=False,
        train_weights=True,
        weights_activation=None,
    )

    device = next(ensemble.parameters()).device

    N_train = config_json["N"]
    print("Training size N from config:", N_train)

    print("Trainable parameters in ensemble:")
    ensemble.count_trainable_parameters(verbose=True)

    # ------------------------------------------------------------------
    # Use full dataset (no bootstrap, like NF WiFi)
    # ------------------------------------------------------------------
    x_data = torch.from_numpy(data_train_tot).to(device).double()

    # ------------------------------------------------------------------
    # Precompute model probabilities on bootstrap sample (N, M)
    # This mirrors model_probs in the NF WiFi script
    # ------------------------------------------------------------------
    with torch.no_grad():
        model_probs = torch.stack(
            [
                m.call(x_data)[-1, :, 0] / m.get_norm()[-1]
                for m in ensemble.ensemble
            ],
            dim=1,
        )  # shape (N, M)

    # Move to CPU + float64 for stable Newton optimisation
    model_probs = model_probs.to(dtype=torch.float64, device="cpu")
    N, M = model_probs.shape
    print("model_probs shape:", model_probs.shape)

    print("model_probs min, max:", model_probs.min().item(), model_probs.max().item())
    print("fraction model_probs < 0:", (model_probs < 0).double().mean().item())
    print("fraction non finite:", (~torch.isfinite(model_probs)).double().mean().item())

    # ------------------------------------------------------------------
    # Define NLL(w) as in NF WiFi:
    #   -log( sum_j w_j f_j(x_n) ).mean_n + lambda_norm * (sum_j w_j - 1)
    # ------------------------------------------------------------------

    def nll_weights(w):
        lambda_norm = 1.0
        # w: (M,) float64, requires_grad=True
        p = torch.clamp_min((model_probs * w.view(1, -1)).sum(dim=1), 0.0)  # (N,)
        return -torch.log(p + 1e-12).mean() + lambda_norm * (w.sum() - 1.0)

    # ------------------------------------------------------------------
    # Optimise weights with torchmin (Newton method), like NF WiFi
    # ------------------------------------------------------------------
    w0_np = np.ones(M) / M
    w0 = torch.tensor(w0_np, dtype=torch.float64)
    # add small noise as in NF script
    w0 = (w0 + 1e-2 * torch.randn_like(w0)).detach().clone().requires_grad_()
    print("Initial weights guess:", w0)

    torch.cuda.empty_cache()
    gc.collect()

    attempt = 0
    max_attempts = 50
    best = None

    w_i_initial = np.ones(M) / M

    while attempt < max_attempts:
        w0 = torch.tensor(w_i_initial, dtype=torch.float64, requires_grad=True)
        w0 = (w0 + 1e-2 * torch.randn_like(w0)).detach().clone().requires_grad_()
        print("w0:", w0)

        try:
            loss0 = nll_weights(w0)
            if not torch.isfinite(loss0):
                print(f"Attempt {attempt+1} skipped, non finite loss {loss0.item()}")
                attempt += 1
                continue

            print(f"Attempt {attempt+1}, start loss {loss0.item():.4f}")

            torch.cuda.empty_cache()
            gc.collect()

            res = minimize(
                nll_weights,
                w0,
                method="newton-exact",
                options={"disp": False, "max_iter": 300},
            )

            if not res.success:
                print(f"Attempt {attempt+1} failed, {res.message}")
                attempt += 1
                continue

            w_try = res.x.detach()

            # NF style sanity check, are we relying on clamp to hide p(x)<=0
            p_try = (model_probs * w_try.view(1, -1)).sum(dim=1)
            frac_bad = (p_try <= 0).double().mean().item()
            pmin = p_try.min().item()
            print(
                f"Attempt {attempt+1} done, frac p<=0 {frac_bad:.3e}, pmin {pmin:.3e}, sumw {w_try.sum().item():.6f}"
            )

            if frac_bad < 1e-6:
                best = w_try
                break

            if best is None:
                best = w_try

            attempt += 1

        except Exception as e:
            print(f"Attempt {attempt+1} exception {e}")
            attempt += 1

    if best is None:
        raise RuntimeError("Optimization failed after multiple attempts.")

    w_final = best
    final_loss = nll_weights(w_final).item()
    print("Final loss (NLL):", final_loss)
    print("Final weights:", w_final)
    print("Sum of weights:", w_final.sum().item())

    p_final = (model_probs * w_final.view(1, -1)).sum(dim=1)
    print("p(x) min, mean:", p_final.min().item(), p_final.mean().item())
    print("fraction p(x) <= 0:", (p_final <= 0).double().mean().item())
    print("min weight:", w_final.min().item())

    # ------------------------------------------------------------------
    # Copy final weights into ensemble module and save
    # ------------------------------------------------------------------
    with torch.no_grad():
        ensemble.weights.data = w_final.to(
            dtype=ensemble.weights.dtype,
            device=ensemble.weights.device,
        )

    final_weights = ensemble.weights.detach().cpu().numpy()
    np.save(results_dir / "final_weights.npy", final_weights)
    print("Saved final_weights.npy")

    # ------------------------------------------------------------------
    # Hessian + sandwich covariance (same nll as optimisation)
    # ------------------------------------------------------------------
    if args.compute_covariance:
        print("Computing Hessian of NLL wrt weights (autograd)...")
        w_for_hess = w_final.detach().clone().requires_grad_()
        H = hessian(nll_weights, w_for_hess).detach()
        H_np = H.cpu().numpy()
        np.save(results_dir / "Hessian_weights.npy", H_np)
        print("Saved Hessian_weights.npy")

        print("Computing sandwich covariance...")
        # model_probs already computed above, on CPU float64
        cov_w = compute_sandwich_covariance(
            H, w_final.double(), model_probs, lam=1.0
        )
        cov_w_np = cov_w.detach().cpu().numpy()
        np.save(results_dir / "cov_weights.npy", cov_w_np)
        print("Saved cov_weights.npy")

        if not args.no_plots:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cov_w_np, interpolation="none")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(plots_dir / "cov_weights_matrix.png", dpi=200)
            plt.close(fig)

    # ------------------------------------------------------------------
    # Build grid & evaluate final ensemble for plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        x_data_cpu = x_data.detach().cpu().numpy()
        pad = 0.05

        x0_lo, x0_hi = x_data_cpu[:, 0].min(), x_data_cpu[:, 0].max()
        x1_lo, x1_hi = x_data_cpu[:, 1].min(), x_data_cpu[:, 1].max()

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
            x_data,
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
            x_data[:, 0].cpu().numpy(),
            x_data[:, 1].cpu().numpy(),
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
            x_data=x_data.float(),                  # (N,2)
            weights=w_final.detach().cpu(),         # (M,)
            cov_w=cov_w_np,                         # (M,M) or None
            feature_names=feature_names,
            outdir=str(plots_dir),
            bins=40,
        )

    print("All done.")

if __name__ == "__main__":
    main()
