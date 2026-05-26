"""
Frequentist coverage test for <x_dim> under the wifi-reweighted model
p_hat(x) = r_hat(x)*q(x)/Z, via fresh DGP-sample pseudoexperiments.

For each pseudoexperiment k = 1..N_PE:
  1. Draw a fresh independent sample of size N from the benchmark DGP,
     a fresh reference sample of size N_ref from q, AND a fresh SIR pool
     of size M_sir from q.
  2. Refit ŵ_k from scratch (BCE / scipy trust-exact) on the fresh sample,
     basis frozen.
  3. Recompute the sandwich Σ_w_k from the same fresh sample.
  4. Compute <x_dim>_k under p_hat_k = r(.;ŵ_k) * q via SIR on the fresh
     pool, plus the closed-form gradient ∂<x_dim>/∂w and the SNIS variance
     σ²_pool of the SIR estimator.
  5. σ_k² = grad^T Σ_w_k grad + σ²_pool.
     pull_k = (<x_dim>_k − truth) / σ_k.

The σ_k decomposition has two independent contributions:
  - σ²_sandwich = grad^T Σ_w grad — finite-N data variance through ŵ_k.
  - σ²_pool    = Σ p_i_norm² (x_i − mean)² — finite-M_sir SIR MC variance.
The pool term survives the limit Σ_w → 0; with a fixed pool across PEs it
would be a constant offset that doesn't show up in pull spread, so the
pool is resampled per PE so coverage actually verifies σ_total.

`truth` is the empirical mean over a large independent DGP sample (limit
of the population mean). Coverage = fraction of pulls within 1σ / 2σ;
target ~68% / 95% if σ_total is correctly calibrated.

Notes
-----
- The basis is held FROZEN across pseudoexperiments (refitting K MLPs
  per PE would be prohibitive). Coverage is therefore conditional on the
  basis. To get unconditional coverage you'd retrain the basis too.
- Each PE uses fresh independent DGP draws — not bootstraps of the
  original training data — so there is no residual sample bias.

Reads the wifi run dir written by run.py.

Usage:
    python coverage.py --name <wifi_run_name>
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
from scipy.stats import norm as norm_dist

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = PACKAGE_DIR
sys.path.insert(0, PACKAGE_DIR)

from benchmarks import get_benchmark
from reference import sample_reference
from basis import MLPLogit, evaluate_features
from wifi_train import fit_linear_head, sandwich_cov


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def _load_basis(basis_dir, d_in, hidden):
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
    basis_dir = os.path.join(out_dir, "basis")
    d_in = mu_q.numel()
    models = _load_basis(basis_dir, d_in, cfg["MLP_HIDDEN"])
    return cfg, mu_q, Sigma_q, w_hat, models


# ──────────────────────────────────────────────────────────────────────
# Observable: <x_dim> under p_hat = r * q, via SIR on a q-pool
# ──────────────────────────────────────────────────────────────────────

def _sir_weights(F, w):
    s = F @ w
    s = s - s.max()
    p = torch.exp(s)
    return p / p.sum()


def model_mean_grad_and_pool_var(F_q, Xq, w, dim=0):
    """
    Return (<x_dim>, ∇_w <x_dim>, σ²_pool) under p_hat = r(·;w) · q, evaluated
    on a q-pool (Xq, F_q) with self-normalised IS weights.

        d <x_dim> / dw = E_p̂[(x_dim − <x_dim>) · features(x)]

    σ²_pool is the asymptotic SNIS variance estimate of the mean estimator
    — the residual MC error from finite pool size. Even with ŵ known
    exactly, the SIR estimator still has this variance from approximating
    the integral over q with M_pool samples; it scales as 1/M_pool through
    p_norm = O(1/M_pool):

        σ²_pool ≈ Σ_i p_i_norm² (x_i − <x_dim>)²
                = (1/M) E_q[w(x)² (x − μ)²] / (E_q[w])²

    where w(x) ∝ exp(F·w_k) is the importance ratio. Independent of the
    sandwich Σ_w (which captures only Var(ŵ_k)), so total
    Var(mean_k − truth) = grad^T Σ_w grad + σ²_pool.
    """
    p = _sir_weights(F_q, w)
    x = Xq[:, dim]
    mean = (p * x).sum()
    centred = x - mean
    grad = (p.unsqueeze(1) * centred.unsqueeze(1) * F_q).sum(dim=0)
    var_pool = (p ** 2 * centred ** 2).sum()
    return float(mean), grad, float(var_pool)


# ──────────────────────────────────────────────────────────────────────
# Pseudoexperiment loop
# ──────────────────────────────────────────────────────────────────────

def run_coverage(out_dir, dim=0, verbose=True):
    cfg, mu_q, Sigma_q, w_hat_central, models = _load_run(out_dir)
    seed = int(cfg.get("seed", 0))

    # Pseudoexperiment hyperparameters. The legacy key COVERAGE_N_BOOTSTRAP is
    # retained so existing configs keep working.
    N_PE = int(cfg.get("COVERAGE_N_PSEUDOEXP",
                       cfg.get("COVERAGE_N_BOOTSTRAP", 100)))
    M_sir = int(cfg.get("COVERAGE_M_SIR", 200_000))

    # Match the original linhead training scale.
    N_data = int(cfg.get("N_train", 100_000)) // 2
    cfg_n_ref = cfg.get("N_REF_LINHEAD", None)
    N_ref = int(cfg_n_ref) if cfg_n_ref else N_data

    lh_max_iter = int(cfg.get("LINHEAD_MAX_ITER", 500))
    ridge_rel = float(cfg.get("SANDWICH_RIDGE_REL", 1e-8))
    lam_ridge = float(cfg.get("LINHEAD_RIDGE", 0.0))

    if verbose:
        print(f"[coverage] {N_PE} pseudoexperiments, "
              f"N_data={N_data}, N_ref={N_ref}, M_sir={M_sir}, dim=x_{dim}")
        print(f"[coverage] basis frozen; refitting ŵ via fit_linear_head per PE")
        print(f"[coverage] q-pool resampled per PE so σ_total verifies "
              f"σ²_sandwich + σ²_pool")

    # ── Truth: empirical mean over a large independent DGP sample ───
    gen = get_benchmark(cfg["benchmark"])
    N_truth = max(10_000_000, 50 * N_data)
    X_truth = gen(N_truth, seed=seed + 999_999)
    truth = float(X_truth[:, dim].mean())
    if verbose:
        print(f"[coverage] truth: empirical <x_{dim}> on {N_truth} fresh DGP "
              f"samples = {truth:.6f}")
    del X_truth  # free memory

    # ── Pseudoexperiment loop ────────────────────────────────────────
    # Each PE draws fresh data, fresh reference, AND a fresh q-pool. The
    # last is so Var(mean_k) over PEs picks up both the sandwich-style
    # finite-N variance via ŵ_k AND the SIR-pool MC variance σ²_pool —
    # otherwise σ²_pool is a constant offset across PEs and the sandwich-
    # only σ_k passes coverage by silently leaving σ²_pool out of the test.
    pulls = np.zeros(N_PE)
    sigmas = np.zeros(N_PE)
    sigmas_sandwich = np.zeros(N_PE)
    sigmas_pool = np.zeros(N_PE)
    means_fit = np.zeros(N_PE)
    bce_losses = np.zeros(N_PE)

    for k in range(N_PE):
        # Fresh independent draws from DGP (data), q (reference), and the
        # SIR pool used to evaluate <x_dim>. Three distinct seed offsets so
        # they don't accidentally collide.
        seed_data = seed + 100_000 + k * 13
        seed_ref  = seed + 200_000 + k * 17
        seed_pool = seed + 300_000 + k * 19
        X_k = gen(N_data, seed=seed_data)
        Xt_k = torch.from_numpy(X_k).double()
        F_data_k = evaluate_features(models, Xt_k)

        X_ref_k = sample_reference(mu_q, Sigma_q, N_ref, seed=seed_ref)
        F_ref_k = evaluate_features(models, X_ref_k)

        Xq_pool_k = sample_reference(mu_q, Sigma_q, M_sir, seed=seed_pool)
        F_q_pool_k = evaluate_features(models, Xq_pool_k)

        # Refit linear head from scratch (no warm start, no prior).
        w_k, bce_k = fit_linear_head(
            F_data_k, F_ref_k, max_iter=lh_max_iter,
            lam_ridge=lam_ridge, verbose=False,
        )
        bce_losses[k] = bce_k

        # Sandwich covariance on the fresh sample.
        Sigma_k, _, _ = sandwich_cov(
            w_k, F_data_k, F_ref_k, ridge_rel=ridge_rel,
            lam_ridge=lam_ridge,
        )

        # Observable + delta-method σ_sandwich + analytic σ_pool.
        mean_k, grad_k, var_pool_k = model_mean_grad_and_pool_var(
            F_q_pool_k, Xq_pool_k, w_k, dim=dim)
        var_sandwich = float(grad_k @ Sigma_k @ grad_k)
        sigma_sandwich = np.sqrt(max(var_sandwich, 0.0))
        sigma_pool = np.sqrt(max(var_pool_k, 0.0))
        # σ_pool and σ_sandwich are independent by construction (the SIR pool
        # is drawn independently of (X_data, X_ref) on which Σ_w is estimated).
        sigma_k = np.sqrt(var_sandwich + var_pool_k)

        means_fit[k] = mean_k
        sigmas[k] = sigma_k
        sigmas_sandwich[k] = sigma_sandwich
        sigmas_pool[k] = sigma_pool
        pulls[k] = (mean_k - truth) / (sigma_k + 1e-30)

        if verbose and (k + 1) % max(1, N_PE // 10) == 0:
            print(f"  [{k+1}/{N_PE}] <x_{dim}>={mean_k:+.4f}  "
                  f"σ_sw={sigma_sandwich:.4f}  σ_pool={sigma_pool:.4f}  "
                  f"σ={sigma_k:.4f}  pull={pulls[k]:+.3f}  bce={bce_k:.4f}")

    # ── Coverage stats ──────────────────────────────────────────────
    cov1 = float(np.mean(np.abs(pulls) < 1.0))
    cov2 = float(np.mean(np.abs(pulls) < 2.0))
    if verbose:
        print(f"\n  1σ coverage: {cov1*100:.1f}%  (target ~68%)")
        print(f"  2σ coverage: {cov2*100:.1f}%  (target ~95%)")
        print(f"  pull mean = {pulls.mean():+.3f}, std = {pulls.std():.3f}  "
              f"(target 0, 1)")
        print(f"  fitted <x_{dim}>: mean = {means_fit.mean():+.4f}, "
              f"std = {means_fit.std():.4f}")
        print(f"  σ_sandwich : mean = {sigmas_sandwich.mean():.4f}, "
              f"std = {sigmas_sandwich.std():.4f}")
        print(f"  σ_pool     : mean = {sigmas_pool.mean():.4f}, "
              f"std = {sigmas_pool.std():.4f}")
        print(f"  σ_total    : mean = {sigmas.mean():.4f}, "
              f"std = {sigmas.std():.4f}")
        # σ_pool fraction of total variance — telling us whether the pool
        # MC error is non-trivial relative to the data-N variance.
        var_pool_frac = float((sigmas_pool ** 2).mean()
                              / max((sigmas ** 2).mean(), 1e-30))
        print(f"  Var(σ_pool) / Var(σ_total) = {var_pool_frac:.3f}  "
              f"(0 = pool negligible; 1 = pool dominates)")
        # Useful diagnostic: ratio of mean σ to the actual std of the mean.
        sigma_ratio = sigmas.mean() / max(means_fit.std(), 1e-30)
        print(f"  σ̄ / std(<x_{dim}>) = {sigma_ratio:.3f}  "
              f"(target 1.0; <1 = under-cover, >1 = over-cover)")

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "dim": dim,
        "truth": truth,
        "truth_n_samples": int(N_truth),
        "n_pseudoexperiments": int(N_PE),
        "N_data_per_pe": int(N_data),
        "N_ref_per_pe": int(N_ref),
        "M_sir_per_pe": int(M_sir),
        "coverage_1sigma": cov1,
        "coverage_2sigma": cov2,
        "pull_mean": float(pulls.mean()),
        "pull_std": float(pulls.std()),
        "fitted_mean_mean": float(means_fit.mean()),
        "fitted_mean_std": float(means_fit.std()),
        "sigma_total_mean": float(sigmas.mean()),
        "sigma_total_std": float(sigmas.std()),
        "sigma_sandwich_mean": float(sigmas_sandwich.mean()),
        "sigma_sandwich_std": float(sigmas_sandwich.std()),
        "sigma_pool_mean": float(sigmas_pool.mean()),
        "sigma_pool_std": float(sigmas_pool.std()),
        "var_pool_fraction_of_total": float(
            (sigmas_pool ** 2).mean() / max((sigmas ** 2).mean(), 1e-30)),
        "sigma_ratio": float(sigmas.mean() / max(means_fit.std(), 1e-30)),
        "bce_loss_mean": float(bce_losses.mean()),
    }
    with open(os.path.join(out_dir, "coverage_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(out_dir, "coverage_pulls.npy"), pulls)
    np.save(os.path.join(out_dir, "coverage_sigmas.npy"), sigmas)
    np.save(os.path.join(out_dir, "coverage_sigmas_sandwich.npy"), sigmas_sandwich)
    np.save(os.path.join(out_dir, "coverage_sigmas_pool.npy"), sigmas_pool)
    np.save(os.path.join(out_dir, "coverage_means.npy"), means_fit)

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.hist(pulls, bins=20, density=True, alpha=0.6, color="C0",
            edgecolor="C0", label=f"pulls (N={N_PE})")
    xp = np.linspace(-4, 4, 200)
    ax.plot(xp, norm_dist.pdf(xp), "k--", lw=1.5, label="N(0, 1)")
    for s in (-2, -1, 1, 2):
        ax.axvline(s, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel(rf"pull = ($<x_{dim}>_k$ − truth) / $\sigma_k$")
    ax.set_ylabel("density")
    ax.set_title(
        f"coverage: {cov1*100:.0f}% 1σ / {cov2*100:.0f}% 2σ "
        f"(target 68 / 95);  μ={pulls.mean():+.2f} σ={pulls.std():.2f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # σ_k vs <x_dim>_k — should look like a horizontal cloud if the σ
    # estimator is well-calibrated. Truth marked as a vertical line.
    ax = axes[1]
    ax.scatter(means_fit, sigmas, s=14, alpha=0.6, color="C0")
    ax.axvline(truth, color="C3", lw=1.5, label=f"truth = {truth:.4f}")
    ax.axhline(means_fit.std(), color="k", ls="--", lw=1.0,
               label=f"std($<x_{dim}>_k$) = {means_fit.std():.4f}")
    ax.axhline(sigmas.mean(), color="gray", ls=":", lw=1.0,
               label=f"mean σ_k = {sigmas.mean():.4f}")
    ax.set_xlabel(rf"$<x_{dim}>_k$ (fit value per PE)")
    ax.set_ylabel(r"$\sigma_k$ (delta-method)")
    ax.set_title(rf"$\sigma_k$ vs $<x_{dim}>_k$  "
                 rf"(σ̄/std = {results['sigma_ratio']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coverage_pulls.png"), dpi=150)
    plt.close(fig)

    if verbose:
        print(f"  wrote coverage_results.json, coverage_pulls.{{npy,png}}, "
              f"coverage_{{sigmas,means}}.npy")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run folder under runs/")
    p.add_argument("--dim", type=int, default=0)
    args = p.parse_args()
    out_dir = os.path.join(PROJECT_ROOT, "runs", args.name)
    run_coverage(out_dir, dim=args.dim)


if __name__ == "__main__":
    main()
