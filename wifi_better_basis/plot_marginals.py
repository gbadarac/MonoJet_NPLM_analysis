"""
Marginal plots for a wifi-ensemble run.

Reads artifacts from runs/<run_name>/ (mu_q, Sigma_q, w_hat, Σ_w_sandwich,
basis state dicts) and the cached training data, then plots the 1D marginals
of the model density p_hat(x) ∝ r_hat(x) * q(x) against the data and (if
available) the analytic marginal for the benchmark.

Marginals come from SIR: draw M samples from q, importance-weight by
r_hat(x_i) = exp(features(x_i) · ŵ), and bin in each dimension. The ±1σ
uncertainty band is computed analytically via delta-method propagation of
Σ_w through the SIR-bin estimator (no bootstrap):

    p_i(w)        = exp(F_i · w) / Σ_k exp(F_k · w)        (SIR weights)
    M_d_j(w)      = (Σ_{i ∈ bin_j} p_i) / width_j           (marginal density)
    ∂ M_d_j / ∂w  = M_d_j · (⟨f⟩_{bin_j} − ⟨f⟩_p̂)
    σ_M_d_j²     = (∂ M_d_j / ∂w)^T Σ_w (∂ M_d_j / ∂w)

Bins whose feature signature ⟨f⟩_{bin_j} matches the overall ⟨f⟩_p̂ have
tight σ; bins that pull strongly on the wifi reweighter have looser σ.

Usage:
    # Re-plot an existing run (no retraining)
    python plot_marginals.py --name wifi_2d_gmm_skew_K10_N100000_s7

    # Or imported and called from run.py at the end of the pipeline
    from plot_marginals import plot_marginals
    plot_marginals(out_dir)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = PACKAGE_DIR
sys.path.insert(0, PACKAGE_DIR)

from benchmarks import get_marginals
from reference import sample_reference
from basis import MLPLogit, evaluate_features


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def _load_basis(basis_dir, d_in, hidden):
    """Reload all frozen MLP basis members in numerical order."""
    fnames = sorted(f for f in os.listdir(basis_dir) if f.endswith(".pt"))
    models = []
    for f in fnames:
        m = MLPLogit(d_in, hidden=tuple(hidden))
        m.load_state_dict(torch.load(os.path.join(basis_dir, f), weights_only=True))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    return models


def _load_run(out_dir):
    with open(os.path.join(out_dir, "wifi_config.json")) as f:
        cfg = json.load(f)
    mu_q    = torch.from_numpy(np.load(os.path.join(out_dir, "mu_q.npy"))).double()
    Sigma_q = torch.from_numpy(np.load(os.path.join(out_dir, "Sigma_q.npy"))).double()
    w_hat   = torch.from_numpy(np.load(os.path.join(out_dir, "w_hat.npy"))).double()
    Sigma_w = np.load(os.path.join(out_dir, "Sigma_w_sandwich.npy"))

    basis_dir = os.path.join(out_dir, "basis")
    d_in = mu_q.numel()
    models = _load_basis(basis_dir, d_in, cfg["MLP_HIDDEN"])
    if len(models) != cfg["K"]:
        print(f"  warn: found {len(models)} basis members on disk, "
              f"config says K={cfg['K']}")
    return cfg, mu_q, Sigma_q, w_hat, Sigma_w, models


def _load_data(cfg):
    # Decoupled N_train / N_test: cache directory encodes both.
    name = (
        f"{cfg['benchmark']}_Ntrain{cfg['N_train']}_Ntest{cfg['N_test']}"
        f"_seed{cfg['seed']}"
    )
    return np.load(os.path.join(PROJECT_ROOT, "data", name, "data_train.npy"))


# ──────────────────────────────────────────────────────────────────────
# Analytic marginal density and ±1σ band via delta method on Σ_w
# ──────────────────────────────────────────────────────────────────────

def _analytic_marginal_band(F, Xq, w_hat, Sigma_w, edges, dim):
    """
    Compute (centres, M_d, sigma_d) for one dimension's marginal.

    M_d_j     = SIR bin estimator of the marginal density at bin j.
    sigma_d_j = closed-form 1σ from delta-method propagation of Σ_w.

    Notes
    -----
    - Samples that fall outside the displayed bin range are dropped from
      both M_d and the band (so the curve+band integrates to ≈ 1 over the
      display window, matching the data histogram drawn with density=True).
    """
    s = F @ w_hat
    s = s - s.max()
    p = torch.exp(s); p = p / p.sum()
    p_np = p.cpu().numpy()
    F_np = F.cpu().numpy()
    Sw = np.asarray(Sigma_w)

    K1 = F_np.shape[1]
    n_bins = len(edges) - 1
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    x_d = Xq[:, dim].cpu().numpy() if isinstance(Xq, torch.Tensor) else Xq[:, dim]
    bin_idx = np.digitize(x_d, edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_in = bin_idx[in_range]
    p_in = p_np[in_range]
    F_in = F_np[in_range]

    # Renormalise weights to sum to 1 over in-range samples so the curve and
    # the data histogram (drawn with density=True over the same window) live
    # on the same vertical scale.
    p_in = p_in / max(p_in.sum(), 1e-30)

    # Overall feature mean under the in-range portion of p̂.
    mean_f_overall = (p_in[:, None] * F_in).sum(axis=0)

    sum_w = np.zeros(n_bins)
    sum_wF = np.zeros((n_bins, K1))
    np.add.at(sum_w, bin_in, p_in)
    np.add.at(sum_wF, bin_in, p_in[:, None] * F_in)

    M_d = sum_w / widths

    mean_f_bin = np.zeros((n_bins, K1))
    mask = sum_w > 0
    mean_f_bin[mask] = sum_wF[mask] / sum_w[mask, None]

    delta = mean_f_bin - mean_f_overall  # (n_bins, K+1)
    var = (M_d ** 2) * np.einsum("ij,jk,ik->i", delta, Sw, delta)
    sigma_d = np.sqrt(np.clip(var, 0.0, None))
    return centres, M_d, sigma_d


# ──────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────

def plot_marginals(out_dir, M_sir=200_000, n_bins=80, seed=0):
    """
    Generate one figure with one panel per dimension, saved to out_dir as
    marginals.png.

    Parameters
    ----------
    out_dir : path to runs/<run_name>/
    M_sir   : number of importance samples from q
    n_bins  : histogram bin count
    """
    print(f"[plot_marginals] reading run {out_dir}")
    cfg, mu_q, Sigma_q, w_hat, Sigma_w, models = _load_run(out_dir)
    X_data = _load_data(cfg)
    d = X_data.shape[1]

    print(f"[plot_marginals] drawing {M_sir} SIR samples from q and"
          f" evaluating the frozen basis")
    Xq = sample_reference(mu_q, Sigma_q, M_sir, seed=int(seed) + 10001)
    F = evaluate_features(models, Xq)         # (M_sir, K+1)

    marginals_fn = get_marginals(cfg["benchmark"])

    fig, axes = plt.subplots(1, d, figsize=(5 * d, 4), squeeze=False)
    axes = axes[0]

    for dim in range(d):
        lo, hi = np.quantile(X_data[:, dim], [0.001, 0.999])
        pad = 0.05 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, n_bins + 1)

        ax = axes[dim]

        # 1) data histogram
        ax.hist(
            X_data[:, dim], bins=edges, density=True,
            histtype="step", color="black", lw=1.5, label="data",
        )

        # 2) analytic ±1σ band from Σ_w via delta method
        centres, M_d, sigma_d = _analytic_marginal_band(
            F, Xq, w_hat, Sigma_w, edges, dim,
        )
        ax.fill_between(
            centres, M_d - sigma_d, M_d + sigma_d,
            alpha=0.30, color="C0", step="mid",
            label=r"$\hat r\cdot q$, $\pm 1\sigma$ via $\Sigma_w$",
        )

        # 3) headline curve from ŵ
        ax.step(
            centres, M_d, where="mid", color="C0", lw=1.5,
            label=r"$\hat r\cdot q$ at $\hat w$",
        )

        # 4) analytic marginal if available
        if marginals_fn is not None:
            try:
                grids = [np.linspace(*np.quantile(X_data[:, j], [0.001, 0.999]), 200)
                         for j in range(d)]
                analytic = marginals_fn(grids)
                ax.plot(grids[dim], analytic[dim], color="C3", lw=1.5,
                        ls="--", label="analytic truth")
            except Exception as e:
                print(f"  warn: analytic marginals unavailable: {e}")

        ax.set_xlabel(rf"$x_{dim}$")
        ax.set_ylabel("density")
        ax.set_title(rf"marginal $x_{dim}$  ({cfg['benchmark']})")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"wifi run: {os.path.basename(out_dir)}   "
        f"(K={cfg['K']}, N_train={cfg['N_train']}, seed={cfg['seed']})",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "marginals.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_marginals] wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True,
                   help="run folder name under runs/")
    p.add_argument("--M_sir", type=int, default=200_000)
    p.add_argument("--bins", type=int, default=80)
    args = p.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "runs", args.name)
    plot_marginals(out_dir, M_sir=args.M_sir, n_bins=args.bins)


if __name__ == "__main__":
    main()
