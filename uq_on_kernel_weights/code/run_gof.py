"""
Script 4: Goodness-of-fit test via learned likelihood ratio (NPLM),
calibrated with toys sampled from the null hypothesis.

Runs three variants of the LRT on observed data:
  - (default)  weights trainable with Gaussian constraint from Hessian covariance
  - (_free)    weights trainable without constraint
  - (_frozen)  weights frozen to their initial (fitted) values

For each variant: loss curves, weight pulls, coefficient evolution,
marginal comparison plots (ratio to data, true, and HIKER).
Then calibration toys for the constrained variant.

Supports ensembles with per-seed norm kernels (loaded from ensemble_info.npz).
Each GoF configuration gets its own subfolder (gof_<label>/).
Skips already-computed variants unless --force is passed.

Reads: {OUT_DIR}/data_config.json, train_config.json, model.pt, covariance.npy
       {OUT_DIR}/ensemble_info.npz (if ensemble with per-seed norm kernels)
       {DATA_DIR}/data_test.npy
Saves: {OUT_DIR}/gof_<label>/gof_results{_free,_frozen}.json,
       gof_model_{den,num}{suffix}.pt, gof_loss_hist{suffix}.npz,
       gof_loss_curves.png, gof_weight_pulls.png, gof_coeffs_evolution.png,
       gof_marginals_x{dim}.png, gof_vs_true_x{dim}.png, gof_vs_hiker_x{dim}.png,
       gof_calibration.png
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

import argparse

from hiker import reconstruct_hiker, sample_from_hiker_auto
from hessian_uq import predictive_variance
from gof import run_gof_test
from benchmarks import get_marginals
from gof_plots import make_all_marginal_plots

# ── Args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true",
                    help="Force re-run of all variants even if results exist.")
_args = parser.parse_args()
FORCE_RERUN = _args.force

# ── Paths ─────────────────────────────────────────────────────────────
OUT_DIR = os.environ.get("HIKER_OUT_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load pipeline config (or defaults) ────────────────────────────────
_cfg_path = os.path.join(OUT_DIR, "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
else:
    _cfg = {}

N_TEST_GOF = _cfg.get("N_TEST_GOF", 20000)
N_KERNELS_NUM = _cfg.get("N_KERNELS_NUM", 80)
KERNEL_WIDTH_NUM = _cfg.get("KERNEL_WIDTH_NUM", 0.10)
CLIP_NUM = _cfg.get("CLIP_NUM", 0.2)
LAMBDA_NUM = _cfg.get("LAMBDA_NUM", 0.0)
EPOCHS_DEN = _cfg.get("EPOCHS_DEN", 30000)
EPOCHS_NUM = _cfg.get("EPOCHS_NUM", 80000)
LR_DEN = _cfg.get("LR_DEN", 5e-5)
LR_NUM = _cfg.get("LR_NUM", 5e-5)
PATIENCE = _cfg.get("PATIENCE_GOF", 2000)
N_TOYS = _cfg.get("N_TOYS", 30)

# Load train config early (needed for GOF_LABEL)
with open(os.path.join(OUT_DIR, "train_config.json")) as f:
    train_config = json.load(f)

# GoF output subfolder: auto-generated from hyperparameters, or explicit label
GOF_LABEL = _cfg.get("GOF_LABEL", None)
if GOF_LABEL is None:
    use_norm = train_config.get("USE_NORM_PER_SEED", train_config.get("USE_NORM_KERNEL", True))
    norm_tag = "norm" if use_norm else "nonorm"
    GOF_LABEL = f"N{N_TEST_GOF}_M{N_KERNELS_NUM}_W{KERNEL_WIDTH_NUM}_lr{LR_NUM}_{norm_tag}"
    if CLIP_NUM is not None:
        GOF_LABEL += f"_c{CLIP_NUM}"

GOF_DIR = os.path.join(OUT_DIR, f"gof_{GOF_LABEL}")
os.makedirs(GOF_DIR, exist_ok=True)
print(f"  GoF output dir: {GOF_DIR}")

# Save GoF config for reference
gof_config = {
    "N_TEST_GOF": N_TEST_GOF, "N_KERNELS_NUM": N_KERNELS_NUM,
    "KERNEL_WIDTH_NUM": KERNEL_WIDTH_NUM, "CLIP_NUM": CLIP_NUM,
    "LAMBDA_NUM": LAMBDA_NUM, "EPOCHS_DEN": EPOCHS_DEN, "EPOCHS_NUM": EPOCHS_NUM,
    "LR_DEN": LR_DEN, "LR_NUM": LR_NUM, "PATIENCE_GOF": PATIENCE, "N_TOYS": N_TOYS,
    "GOF_LABEL": GOF_LABEL,
}
with open(os.path.join(GOF_DIR, "gof_config.json"), "w") as f:
    json.dump(gof_config, f, indent=2)

# ── Load ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data, model, and covariance...")
print("=" * 60)

with open(os.path.join(OUT_DIR, "data_config.json")) as f:
    data_config = json.load(f)
DATA_DIR = data_config.get("data_dir", OUT_DIR)
BENCHMARK = data_config.get("benchmark", "2d_gaussian")
data_test = np.load(os.path.join(DATA_DIR, "data_test.npy"))

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

with torch.no_grad():
    all_w = model.get_all_coeffs().cpu().numpy()
    n_neg = (all_w < 0).sum()
    print(f"  M_total = {M_total}, layers = {layer_sizes}, "
          f"weights: min={all_w.min():.4f}, max={all_w.max():.4f}")
    if n_neg:
        print(f"  WARNING: {n_neg} negative weight(s) — will use hit-or-miss for toy generation")
print(f"  N_test available = {len(data_test)}, will use {N_TEST_GOF} per LRT")

# ── Select observed test data ─────────────────────────────────────────
np.random.seed(999)
idx_obs = np.random.choice(len(data_test), size=N_TEST_GOF, replace=False)
x_obs = data_test[idx_obs]
x_obs_t = torch.from_numpy(x_obs).double().to(DEVICE)

with torch.no_grad():
    K_obs_list, K_norm_obs = model.get_kernel_matrices(x_obs_t)

# ── Load ensemble info if available ───────────────────────────────────
ens_info_path = os.path.join(OUT_DIR, "ensemble_info.npz")
gof_ensemble_info = None
if os.path.exists(ens_info_path):
    ens_data = np.load(ens_info_path, allow_pickle=True)
    if bool(ens_data.get("use_norm_per_seed", False)):
        N_ens = int(ens_data["N_ens"])
        seed_boundaries = [tuple(x) for x in ens_data["seed_boundaries"]]
        # Precompute norm kernel evaluations at test data
        norm_kernels_eval = []
        from hiker import KernelLayer
        for s in range(N_ens):
            nc = torch.from_numpy(ens_data[f"norm_kernel_{s}_centroids"]).double().to(DEVICE)
            nw = torch.from_numpy(ens_data[f"norm_kernel_{s}_width"]).double().to(DEVICE)
            nk = KernelLayer(nc, nw.item(), train_centroids=False).to(DEVICE)
            with torch.no_grad():
                norm_kernels_eval.append(nk.kernels(x_obs_t))
        gof_ensemble_info = {
            "seed_boundaries": seed_boundaries,
            "N_ens": N_ens,
            "norm_kernels_eval": norm_kernels_eval,
        }
        print(f"  Loaded ensemble info: {N_ens} seeds with per-seed norm kernels")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def gof_density_at_points(gof_model, x_pts, hiker_model):
    """Evaluate the GoF model density at new points x_pts."""
    x = x_pts.double().to(next(hiker_model.parameters()).device)
    with torch.no_grad():
        w = gof_model.weights
        g = torch.zeros(x.shape[0], dtype=torch.float64, device=w.device)
        offset = 0
        for layer in hiker_model.layers:
            M_l = layer.M
            w_l = w[offset:offset + M_l]
            K = layer.kernel_layer.kernels(x)
            g = g + K @ w_l
            offset += M_l

        if hiker_model.use_norm_kernel:
            K_n = hiker_model.norm_kernel.kernels(x)
            w_norm = 1.0 - w.sum()
            p = g + w_norm * K_n.squeeze(1)
        else:
            Z = w.sum()
            p = g / (Z + 1e-30)

        if gof_model.perturbation_net is not None:
            p = p + gof_model.perturbation_net(x)
    return p.cpu().numpy()


def make_diagnostic_plots(result, label, suffix):
    """
    Produce loss curves, weight pulls, and marginal comparison plots
    for one GoF variant.
    """
    model_den = result["model_den"]
    model_num = result["model_num"]
    loss_den = result["loss_hist_den"]
    loss_num = result["loss_hist_num"]
    train_w = model_den.train_weights

    # ── Loss curves ───────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    if loss_den:
        ax1.plot(np.arange(1, len(loss_den) + 1) * PATIENCE, loss_den,
                 "o-", ms=3, lw=0.8, color="C1")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Denominator (null)", fontsize=13)
    ax1.grid(True, alpha=0.3)

    if loss_num:
        ax2.plot(np.arange(1, len(loss_num) + 1) * PATIENCE, loss_num,
                 "o-", ms=3, lw=0.8, color="C2")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Numerator (alternative)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"GoF training loss [{label}]", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(GOF_DIR, f"gof_loss_curves{suffix}.png"), dpi=150)
    plt.close()
    print(f"  Saved gof_loss_curves{suffix}.png")

    # ── Weight pulls ──────────────────────────────────────────────
    with torch.no_grad():
        w_den = model_den.weights.cpu().numpy()
        w_num = model_num.weights.cpu().numpy()

    pull_den = (w_den - w_hat_np) / (w_sigma + 1e-30)
    pull_num = (w_num - w_hat_np) / (w_sigma + 1e-30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    indices = np.arange(M_total)

    ax1.bar(indices, pull_den, color="C1", alpha=0.7, edgecolor="C1", linewidth=0.5)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.axhline(1, color="gray", ls=":", lw=0.6)
    ax1.axhline(-1, color="gray", ls=":", lw=0.6)
    ax1.set_xlabel("Weight index", fontsize=12)
    ax1.set_ylabel(r"Pull $(w_{\mathrm{fit}} - \hat{w}) / \sigma_w$", fontsize=12)
    ax1.set_title("Denominator (null)", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(indices, pull_num, color="C2", alpha=0.7, edgecolor="C2", linewidth=0.5)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.axhline(1, color="gray", ls=":", lw=0.6)
    ax2.axhline(-1, color="gray", ls=":", lw=0.6)
    ax2.set_xlabel("Weight index", fontsize=12)
    ax2.set_ylabel(r"Pull $(w_{\mathrm{fit}} - \hat{w}) / \sigma_w$", fontsize=12)
    ax2.set_title("Numerator (alternative)", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"GoF weight pulls [{label}]", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(GOF_DIR, f"gof_weight_pulls{suffix}.png"), dpi=150)
    plt.close()
    print(f"  Saved gof_weight_pulls{suffix}.png")

    # ── Coefficient evolution over GoF training ───────────────────
    has_hist = ("w_hist_num" in result and result["w_hist_num"])
    has_pert_hist = ("pert_hist_num" in result and result["pert_hist_num"])

    if has_hist or has_pert_hist:
        n_panels = (1 if has_hist else 0) + (1 if has_pert_hist else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False)
        panel = 0
        epoch_axis = np.arange(1, len(result["loss_hist_num"]) + 1) * PATIENCE

        if has_hist:
            ax = axes[0, panel]
            w_hist = np.array(result["w_hist_num"])  # (n_steps, M_total)
            for j in range(w_hist.shape[1]):
                ax.plot(epoch_axis[:len(w_hist)], w_hist[:, j],
                        lw=0.4, alpha=0.5, color="C2")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Weight value", fontsize=12)
            ax.set_title("Numerator: HIKER weight evolution", fontsize=12)
            ax.grid(True, alpha=0.3)
            panel += 1

        if has_pert_hist:
            ax = axes[0, panel]
            p_hist = np.array(result["pert_hist_num"])  # (n_steps, N_kernels)
            for j in range(p_hist.shape[1]):
                ax.plot(epoch_axis[:len(p_hist)], p_hist[:, j],
                        lw=0.4, alpha=0.5, color="C3")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Perturbation coeff", fontsize=12)
            ax.set_title("Numerator: perturbation coeff evolution", fontsize=12)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"GoF coefficient evolution [{label}]", fontsize=13)
        plt.tight_layout()
        fname = f"gof_coeffs_evolution{suffix}.png"
        fig.savefig(os.path.join(GOF_DIR, fname), dpi=150)
        plt.close()
        print(f"  Saved {fname}")

    # ── Marginal comparisons (3 plots per dimension) ────────────
    d = x_obs.shape[1]
    margin = 0.1
    marginals_fn = get_marginals(BENCHMARK)

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
            f_den_marg[i] = gof_density_at_points(model_den, pts, model).sum() * dx
            f_num_marg[i] = gof_density_at_points(model_num, pts, model).sum() * dx

        sigma_hiker_marg = np.sqrt(np.maximum(var_hiker_marg, 0))

        make_all_marginal_plots(
            GOF_DIR, dim, dim_label, label, suffix,
            x_obs[:, dim], x_grid,
            f_hiker_marg, sigma_hiker_marg, f_den_marg, f_num_marg, true_marg)


def save_gof_result(result, suffix):
    """Save scalar results to JSON and model state dicts for replotting."""
    skip = ("model", "loss_hist", "w_hist", "pert_hist")
    out = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in result.items()
           if not any(k.startswith(s) for s in skip)}
    with open(os.path.join(GOF_DIR, f"gof_results{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)
    # Save only trainable parameters (not the large kernel matrix buffers)
    if "model_den" in result:
        den_save = {k: v for k, v in result["model_den"].state_dict().items()
                    if not k.startswith("K_") and not k.startswith("K_norm")}
        num_save = {k: v for k, v in result["model_num"].state_dict().items()
                    if not k.startswith("K_") and not k.startswith("K_norm")}
        torch.save(den_save, os.path.join(GOF_DIR, f"gof_model_den{suffix}.pt"))
        torch.save(num_save, os.path.join(GOF_DIR, f"gof_model_num{suffix}.pt"))
    # Save loss and coefficient histories
    if "loss_hist_den" in result:
        save_data = {
            "loss_den": np.array(result["loss_hist_den"]),
            "loss_num": np.array(result["loss_hist_num"]),
        }
        if result.get("w_hist_den"):
            save_data["w_den"] = np.array(result["w_hist_den"])
        if result.get("w_hist_num"):
            save_data["w_num"] = np.array(result["w_hist_num"])
        if result.get("pert_hist_num"):
            save_data["pert_num"] = np.array(result["pert_hist_num"])
        np.savez(os.path.join(GOF_DIR, f"gof_loss_hist{suffix}.npz"), **save_data)


# ══════════════════════════════════════════════════════════════════════
# Run the three LRT variants on observed data
# ══════════════════════════════════════════════════════════════════════

VARIANTS = [
    # (label, suffix, train_weights, use_constraint)
    ("constrained (Hessian cov)", "",        True,  True),
    ("free (no constraint)",      "_free",   True,  False),
    ("frozen weights",            "_frozen", False, True),
]

obs_results = {}

for label, suffix, tw, uc in VARIANTS:
    result_path = os.path.join(GOF_DIR, f"gof_results{suffix}.json")
    if os.path.exists(result_path) and not FORCE_RERUN:
        print(f"\n  Skipping '{label}' — {os.path.basename(result_path)} already exists. "
              f"Use --force to re-run.")
        with open(result_path) as f:
            obs_results[suffix] = json.load(f)
        continue

    print("\n" + "=" * 60)
    print(f"LRT on observed data — {label}")
    print("=" * 60)

    torch.manual_seed(123)
    result = run_gof_test(
        K_obs_list, K_norm_obs, layer_sizes, x_data=x_obs_t,
        weights_hat=w_hat, cov_hat=cov_w,
        n_kernels_num=N_KERNELS_NUM, kernel_width_num=KERNEL_WIDTH_NUM,
        clip_num=CLIP_NUM, lambda_num=LAMBDA_NUM,
        epochs_den=EPOCHS_DEN, epochs_num=EPOCHS_NUM,
        lr_den=LR_DEN, lr_num=LR_NUM, patience=PATIENCE,
        train_weights=tw, use_constraint=uc, verbose=True,
        ensemble_info=gof_ensemble_info,
    )

    t_val = result["test_statistic"]
    print(f"\n  t{suffix or ''} = {t_val:.4f}")

    print(f"\nMaking diagnostic plots [{label}]...")
    make_diagnostic_plots(result, label, suffix)
    save_gof_result(result, suffix)
    obs_results[suffix] = result

# Print summary of all variants before starting calibration
print("\n" + "=" * 60)
print("Test statistic summary (before calibration)")
print("=" * 60)
for label, suffix, _, _ in VARIANTS:
    if suffix in obs_results:
        t_v = obs_results[suffix].get("test_statistic", "N/A")
        print(f"  {label:35s}  t = {t_v}")

t_obs = obs_results[""]["test_statistic"]


# ══════════════════════════════════════════════════════════════════════
# Calibration toys (default variant only: constrained)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Running {N_TOYS} calibration toys (constrained variant)")
print("=" * 60)

t_toys = []

for toy in range(N_TOYS):
    print(f"\n--- Toy {toy+1}/{N_TOYS} ---")
    toy_data = sample_from_hiker_auto(model, N_TEST_GOF, seed=7000 + toy)
    x_toy = torch.from_numpy(toy_data).double().to(DEVICE)

    with torch.no_grad():
        K_toy_list, K_norm_toy = model.get_kernel_matrices(x_toy)

    # Recompute per-seed norm kernel evaluations for this toy
    toy_ens_info = None
    if gof_ensemble_info is not None:
        toy_norm_evals = []
        ens_data_reload = np.load(ens_info_path, allow_pickle=True)
        for s in range(gof_ensemble_info["N_ens"]):
            nc = torch.from_numpy(ens_data_reload[f"norm_kernel_{s}_centroids"]).double().to(DEVICE)
            nw_val = float(ens_data_reload[f"norm_kernel_{s}_width"])
            nk = KernelLayer(nc, nw_val, train_centroids=False).to(DEVICE)
            with torch.no_grad():
                toy_norm_evals.append(nk.kernels(x_toy))
        toy_ens_info = {
            "seed_boundaries": gof_ensemble_info["seed_boundaries"],
            "N_ens": gof_ensemble_info["N_ens"],
            "norm_kernels_eval": toy_norm_evals,
        }

    torch.manual_seed(8000 + toy)
    result_toy = run_gof_test(
        K_toy_list, K_norm_toy, layer_sizes, x_data=x_toy,
        weights_hat=w_hat, cov_hat=cov_w,
        n_kernels_num=N_KERNELS_NUM, kernel_width_num=KERNEL_WIDTH_NUM,
        clip_num=CLIP_NUM, lambda_num=LAMBDA_NUM,
        epochs_den=EPOCHS_DEN, epochs_num=EPOCHS_NUM,
        lr_den=LR_DEN, lr_num=LR_NUM, patience=PATIENCE,
        train_weights=True, use_constraint=True, verbose=False,
        ensemble_info=toy_ens_info,
    )
    t_toy = result_toy["test_statistic"]
    t_toys.append(t_toy)
    print(f"  t_toy = {t_toy:.4f}")

t_toys = np.array(t_toys)


# ══════════════════════════════════════════════════════════════════════
# Empirical p-value and Z-score
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Results (constrained variant)")
print("=" * 60)

p_value_emp = np.mean(t_toys >= t_obs)
if p_value_emp > 0:
    z_score_emp = norm_dist.ppf(1.0 - p_value_emp)
else:
    z_score_emp = float("inf")

compatible = z_score_emp < 2.0

print(f"  t_obs = {t_obs:.4f}")
print(f"  Toy distribution: mean={t_toys.mean():.2f}, std={t_toys.std():.2f}, "
      f"median={np.median(t_toys):.2f}")
print(f"  Empirical p-value = {p_value_emp:.4f}  ({(t_toys >= t_obs).sum()}/{N_TOYS} toys above)")
print(f"  Empirical Z-score = {z_score_emp:.2f}")
print(f"  Compatible (Z < 2): {compatible}")

# Compare all three variants
print("\n  --- Test statistic comparison ---")
for label, suffix, _, _ in VARIANTS:
    t_v = obs_results[suffix]["test_statistic"]
    print(f"    {label:30s}  t = {t_v:.4f}")

# ── Save full results ─────────────────────────────────────────────────
full_results = {
    "t_obs": float(t_obs),
    "t_toys": t_toys.tolist(),
    "n_toys": N_TOYS,
    "n_test_per_lrt": N_TEST_GOF,
    "p_value_empirical": float(p_value_emp),
    "z_score_empirical": float(z_score_emp),
    "compatible": bool(compatible),
}
with open(os.path.join(GOF_DIR, "gof_results.json"), "w") as f:
    json.dump(full_results, f, indent=2)

# ── Plot calibration ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(t_toys, bins=max(10, N_TOYS // 3), density=True, alpha=0.6, color="C0",
        edgecolor="C0", label=f"Calibration toys (n={N_TOYS})")
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
print("  Saved gof_calibration.png, gof_results.json")
print("\nDone.")
