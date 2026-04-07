"""
Script 4b: Re-generate all GoF diagnostic plots from saved results.

Can be re-run without re-running the GoF test. Reads saved model weights,
loss histories, and results JSONs.

For each variant (constrained, free, frozen):
  - Loss curves
  - Weight pulls
  - Marginal comparisons (top: densities, bottom: binned ratios to test data)

Also re-generates the calibration plot if toy results exist.

Reads: {OUT_DIR}/model.pt, covariance.npy, data_config.json, train_config.json,
       gof_model_den{suffix}.pt, gof_model_num{suffix}.pt,
       gof_loss_hist{suffix}.npz, gof_results{suffix}.json
Saves: gof_loss_curves{suffix}.png, gof_weight_pulls{suffix}.png,
       gof_marginals_x{dim}{suffix}.png, gof_calibration.png
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm as norm_dist
from scipy.interpolate import interp1d

from hiker import reconstruct_hiker
from hessian_uq import predictive_variance
from benchmarks import get_marginals
from gof_plots import make_all_marginal_plots

# ── Paths ─────────────────────────────────────────────────────────────
OUT_DIR = os.environ.get("HIKER_OUT_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Find all gof_* subdirectories
gof_dirs = sorted([os.path.join(OUT_DIR, d) for d in os.listdir(OUT_DIR)
                   if d.startswith("gof_") and os.path.isdir(os.path.join(OUT_DIR, d))])
if not gof_dirs:
    # Backward compat: look in OUT_DIR itself
    gof_dirs = [OUT_DIR]
print(f"  Found {len(gof_dirs)} GoF output dir(s): {[os.path.basename(d) for d in gof_dirs]}")

# ── Load data, model, covariance ──────────────────────────────────────
print("=" * 60)
print("Loading data and model...")
print("=" * 60)

with open(os.path.join(OUT_DIR, "data_config.json")) as f:
    data_config = json.load(f)
DATA_DIR = data_config.get("data_dir", OUT_DIR)
BENCHMARK = data_config.get("benchmark", "2d_gaussian")

data_test = np.load(os.path.join(DATA_DIR, "data_test.npy"))

with open(os.path.join(OUT_DIR, "train_config.json")) as f:
    train_config = json.load(f)

d = data_test.shape[1]
model = reconstruct_hiker(train_config, d=d).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model.pt"), map_location=DEVICE))
model.eval()

cov_w = torch.from_numpy(np.load(os.path.join(OUT_DIR, "covariance.npy"))).double().to(DEVICE)
w_hat = model.get_free_coeffs().detach().clone()
w_hat_np = w_hat.cpu().numpy()
w_sigma = np.sqrt(np.diag(cov_w.cpu().numpy()))
layer_sizes = [layer.M for layer in model.layers]
M_total = sum(layer_sizes)

# Load GoF hyperparameters
_cfg_path = os.path.join(OUT_DIR, "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as f:
        _cfg = json.load(f)
else:
    _cfg = {}
PATIENCE = _cfg.get("PATIENCE_GOF", 2000)
N_TEST_GOF = _cfg.get("N_TEST_GOF", 20000)
N_KERNELS_NUM = _cfg.get("N_KERNELS_NUM", 80)
KERNEL_WIDTH_NUM = _cfg.get("KERNEL_WIDTH_NUM", 0.10)

# Subsample test data (same seed as run_gof.py)
np.random.seed(999)
idx_obs = np.random.choice(len(data_test), size=min(N_TEST_GOF, len(data_test)), replace=False)
x_obs = data_test[idx_obs]

marginals_fn = get_marginals(BENCHMARK)

print(f"  Benchmark = {BENCHMARK}, M_total = {M_total}")
print(f"  N_test used = {len(x_obs)}")


# ══════════════════════════════════════════════════════════════════════
# Helper: evaluate GoF model density at arbitrary points
# ══════════════════════════════════════════════════════════════════════

def gof_density_at_points(gof_weights, x_pts):
    """Evaluate density using GoF weights (tensor) at new points."""
    x = x_pts.double().to(DEVICE)
    with torch.no_grad():
        w = gof_weights
        g = torch.zeros(x.shape[0], dtype=torch.float64, device=DEVICE)
        offset = 0
        for layer in model.layers:
            M_l = layer.M
            w_l = w[offset:offset + M_l]
            K = layer.kernel_layer.kernels(x)
            g = g + K @ w_l
            offset += M_l

        if model.use_norm_kernel:
            K_n = model.norm_kernel.kernels(x)
            w_norm = 1.0 - w.sum()
            p = g + w_norm * K_n.squeeze(1)
        else:
            Z = w.sum()
            p = g / (Z + 1e-30)
    return p.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# Variants
# ══════════════════════════════════════════════════════════════════════

VARIANTS = [
    ("constrained (Hessian cov)", ""),
    ("free (no constraint)",      "_free"),
    ("frozen weights",            "_frozen"),
]

for GOF_DIR in gof_dirs:
    gof_name = os.path.basename(GOF_DIR)
    print(f"\n{'#' * 60}")
    print(f"Processing: {gof_name}")
    print(f"{'#' * 60}")

    for label, suffix in VARIANTS:
        result_path = os.path.join(GOF_DIR, f"gof_results{suffix}.json")
        den_path = os.path.join(GOF_DIR, f"gof_model_den{suffix}.pt")
        num_path = os.path.join(GOF_DIR, f"gof_model_num{suffix}.pt")
        loss_path = os.path.join(GOF_DIR, f"gof_loss_hist{suffix}.npz")

        if not os.path.exists(result_path):
            print(f"\n  Skipping '{label}' — no results found.")
            continue

        print(f"\n  {'=' * 56}")
        print(f"  Plotting: {label}")
        print(f"  {'=' * 56}")

        with open(result_path) as f:
            result = json.load(f)

        # ── Loss curves ───────────────────────────────────────────
        if os.path.exists(loss_path):
            lh = np.load(loss_path, allow_pickle=True)
            loss_den = lh["den"]
            loss_num = lh["num"]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
            if len(loss_den):
                ax1.plot(np.arange(1, len(loss_den) + 1) * PATIENCE, loss_den,
                         "o-", ms=3, lw=0.8, color="C1")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
            ax1.set_title("Denominator (null)"); ax1.grid(True, alpha=0.3)
            if len(loss_num):
                ax2.plot(np.arange(1, len(loss_num) + 1) * PATIENCE, loss_num,
                         "o-", ms=3, lw=0.8, color="C2")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
            ax2.set_title("Numerator (alternative)"); ax2.grid(True, alpha=0.3)
            plt.suptitle(f"GoF training loss [{label}]", fontsize=14)
            plt.tight_layout()
            fig.savefig(os.path.join(GOF_DIR, f"gof_loss_curves{suffix}.png"), dpi=150)
            plt.close()
            print(f"    Saved gof_loss_curves{suffix}.png")

        # ── Load GoF model weights ────────────────────────────────
        if not (os.path.exists(den_path) and os.path.exists(num_path)):
            print(f"    No saved model weights — skipping weight pulls and marginals.")
            continue

        w_den = torch.load(den_path, map_location="cpu")["weights"].to(DEVICE)
        w_num = torch.load(num_path, map_location="cpu")["weights"].to(DEVICE)
        w_den_np = w_den.cpu().numpy()
        w_num_np = w_num.cpu().numpy()

        num_state = torch.load(num_path, map_location="cpu")
        has_pert = any(k.startswith("perturbation_net.") for k in num_state)

        # ── Weight pulls ──────────────────────────────────────────
        pull_den = (w_den_np - w_hat_np) / (w_sigma + 1e-30)
        pull_num = (w_num_np - w_hat_np) / (w_sigma + 1e-30)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        indices = np.arange(M_total)
        for ax, pull, title, color in [
            (ax1, pull_den, "Denominator (null)", "C1"),
            (ax2, pull_num, "Numerator (alternative)", "C2"),
        ]:
            ax.bar(indices, pull, color=color, alpha=0.7, edgecolor=color, linewidth=0.5)
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.axhline(1, color="gray", ls=":", lw=0.6)
            ax.axhline(-1, color="gray", ls=":", lw=0.6)
            ax.set_xlabel("Weight index"); ax.set_ylabel(r"Pull")
            ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")
        plt.suptitle(f"GoF weight pulls [{label}]", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(GOF_DIR, f"gof_weight_pulls{suffix}.png"), dpi=150)
        plt.close()
        print(f"    Saved gof_weight_pulls{suffix}.png")

        # ── Marginal comparisons ──────────────────────────────────
        pert_coeffs = None
        pert_centres = None
        if has_pert:
            pert_centres = num_state["perturbation_net.centers"].to(DEVICE)
            pert_coeffs = num_state["perturbation_net.coefficients"].to(DEVICE)

        margin = 0.1
        for dim in range(d):
            dim_label = rf"$x_{dim}$"
            n_grid = 300
            lo, hi = x_obs[:, dim].min(), x_obs[:, dim].max()
            pad = margin * (hi - lo)
            x_grid = np.linspace(lo - pad, hi + pad, n_grid)

            if marginals_fn is not None:
                grids_for_true = [np.array([0.0])] * d
                grids_for_true[dim] = x_grid
                true_marg = marginals_fn(grids_for_true)[dim]
            else:
                true_marg = None

            other_dim = 1 - dim
            n_int = 200
            lo_o, hi_o = x_obs[:, other_dim].min(), x_obs[:, other_dim].max()
            pad_o = margin * (hi_o - lo_o)
            x_other = np.linspace(lo_o - pad_o, hi_o + pad_o, n_int)
            dx = x_other[1] - x_other[0]

            f_hiker_marg = np.zeros(n_grid)
            var_hiker_marg = np.zeros(n_grid)
            f_den_marg = np.zeros(n_grid)
            f_num_marg = np.zeros(n_grid)

            for i, xv in enumerate(x_grid):
                if dim == 0:
                    pts = torch.tensor(np.column_stack([np.full(n_int, xv), x_other]),
                                       dtype=torch.float64).to(DEVICE)
                else:
                    pts = torch.tensor(np.column_stack([x_other, np.full(n_int, xv)]),
                                       dtype=torch.float64).to(DEVICE)
                var_f, f_eval = predictive_variance(pts, model, cov_w)
                f_hiker_marg[i] = f_eval.sum().item() * dx
                var_hiker_marg[i] = var_f.sum().item() * dx ** 2
                f_den_marg[i] = gof_density_at_points(w_den, pts).sum() * dx

                f_num_base = gof_density_at_points(w_num, pts).sum() * dx
                if pert_coeffs is not None:
                    with torch.no_grad():
                        import math
                        sigma_p = num_state.get("perturbation_net.sigma",
                                                torch.tensor(KERNEL_WIDTH_NUM))
                        sigma_p = sigma_p.item() if hasattr(sigma_p, 'item') else float(sigma_p)
                        d_dim = pts.shape[1]
                        nc = 1.0 / ((2 * math.pi) ** (d_dim / 2) * sigma_p ** d_dim)
                        diff = pts.unsqueeze(1) - pert_centres.unsqueeze(0)
                        dist_sq = (diff ** 2).sum(dim=2)
                        K_pert = nc * torch.exp(-0.5 * dist_sq / sigma_p ** 2)
                        b = pert_coeffs - pert_coeffs.mean()
                        pert_val = (K_pert @ b).sum().item() * dx
                    f_num_marg[i] = f_num_base + pert_val
                else:
                    f_num_marg[i] = f_num_base

            sigma_hiker_marg = np.sqrt(np.maximum(var_hiker_marg, 0))

            make_all_marginal_plots(
                GOF_DIR, dim, dim_label, label, suffix,
                x_obs[:, dim], x_grid,
                f_hiker_marg, sigma_hiker_marg, f_den_marg, f_num_marg, true_marg)


# ══════════════════════════════════════════════════════════════════════
# Calibration plot (if toy results exist)
# ══════════════════════════════════════════════════════════════════════
for GOF_DIR in gof_dirs:
    cal_path = os.path.join(GOF_DIR, "gof_results.json")
    if not os.path.exists(cal_path):
        continue
    with open(cal_path) as f:
        cal = json.load(f)
    if "t_toys" not in cal or not cal["t_toys"]:
        continue
    t_obs = cal["t_obs"]
    t_toys = np.array(cal["t_toys"])
    p_value_emp = cal.get("p_value_empirical", np.mean(t_toys >= t_obs))
    z_score_emp = cal.get("z_score_empirical", 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(t_toys, bins=max(10, len(t_toys) // 3), density=True, alpha=0.6,
            color="C0", edgecolor="C0", label=f"Calibration toys (n={len(t_toys)})")
    ax.axvline(t_obs, color="red", lw=2, ls="--", label=f"$t_{{obs}}$ = {t_obs:.1f}")
    ax.set_xlabel("Test statistic $t$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"GoF calibration — empirical p = {p_value_emp:.3f}, Z = {z_score_emp:.2f}",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GOF_DIR, "gof_calibration.png"), dpi=150)
    plt.close()
    print(f"  Saved gof_calibration.png in {os.path.basename(GOF_DIR)}")

print("\nDone.")
