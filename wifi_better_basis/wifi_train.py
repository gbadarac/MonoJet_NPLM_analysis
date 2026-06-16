"""
wifi_train.py — Linear head fit and weight covariance estimation.

ROLE IN THE PIPELINE
--------------------
This file implements Stage 1b of the density ratio estimation. The K=128 MLP
classifiers from basis.py are now frozen — each l_k(x) is a fixed function
approximating log(p(x)/q(x)). This file fits the BEST LINEAR COMBINATION of
those K functions on the held-out half_B of the training data:

    log r̂(x) = w_0 * 1 + w_1 * l_1(x) + ... + w_K * l_K(x)
             = ŵ · features(x)

and then estimates the uncertainty on ŵ via three methods:
  - Sandwich estimator  Cov(ŵ) = H^{-1} J H^{-1}   ← the key one for GoF
  - Naive H^{-1}                                    ← diagnostic only
  - Bootstrap (empirical covariance of B=200 refits) ← cross-check

WHY THIS SEPARATION MATTERS
----------------------------
The two-stage structure (nonlinear basis training on half_A, linear head fit
on half_B) is essential:
  1. The MLP classifiers are trained only on half_A. They never see half_B.
  2. The linear head ŵ is fit only on half_B. Because the basis is already
     frozen, this is a CONVEX optimisation problem (logistic regression with
     fixed features) — one global minimum, no local minima, clean analytic
     Hessian and gradient.
  3. Convexity is what makes Cov(ŵ) analytically tractable via the sandwich
     estimator. With NF or kernel basis members, the loss is non-convex and
     the sandwich estimator breaks down — this is why those approaches fail
     to give a well-calibrated GoF test.

THE SANDWICH ESTIMATOR Cov(ŵ) = H^{-1} J H^{-1}
-------------------------------------------------
ŵ is an M-estimator: the minimiser of a sum of per-sample loss functions.
For M-estimators, the asymptotic covariance is the sandwich formula where:

  H = expected Hessian of the loss = F^T diag(sigma(s)(1-sigma(s))) F
      (the "bread": curvature of the loss surface at ŵ)

  J = variance of the score = F^T diag((Y - sigma(s))^2) F
      (the "meat": variability of individual gradients at ŵ)

  Cov(ŵ) = H^{-1} J H^{-1}

If the model is correctly specified (i.e. the linear combination of MLP
logits perfectly captures log(p/q)), then J = H/n and Cov(ŵ) = H^{-1}/n.
The sandwich is more conservative: it remains valid even under misspecification,
which is the typical case with finite K. This is the correct covariance to
use for the GoF test.

The naive H^{-1} assumes correct specification. It is provided for comparison
only — using it when the model is misspecified gives artificially small
uncertainties and an anti-conservative GoF test.
"""

import numpy as np
import torch
from scipy.optimize import minimize as _scipy_minimize
from scipy.special import expit as _expit


# ──────────────────────────────────────────────────────────────────────
# Linear head fit (BCE)
# ──────────────────────────────────────────────────────────────────────

def fit_linear_head(F_data, F_ref, max_iter=500, tol=1e-9, lam_ridge=0.0,
                    verbose=False):
    """
    Fit the wifi linear head ŵ by minimising the BCE loss on the held-out
    half_B data (F_data) and a matched fresh reference draw (F_ref).

    This is logistic regression with fixed features F — the linear combination
    of frozen MLP logits. The loss is:

        L(w) = (sum_data softplus(-s_d) + sum_ref softplus(s_r) + 0.5*lam*||w||^2) / n

    where s = F @ w are the logits, n = N_data + N_ref, and softplus(t) = log(1+e^t).
    This is a convex function of w → unique global minimum → the sandwich
    estimator's asymptotic guarantees hold exactly.

    OPTIMISER: scipy trust-exact (Newton trust-region with exact Hessian)
    The exact Hessian H(w) = (F_d^T D_d F_d + F_r^T D_r F_r) / n is cheap to
    compute (it's just a matrix product). trust-exact uses it directly to take
    near-Newton steps, converging in far fewer iterations than gradient descent.
    The same optimiser is used in classifier_gof.py so the linear head and GoF
    fits are consistent.

    RIDGE REGULARISATION (lam_ridge)
    At K=128, the feature matrix F has 129 columns and the basis members are
    increasingly correlated (many directions of F are nearly redundant). This
    makes the Hessian H near-singular, causing trust-exact to wander along the
    flat directions ("squirrely optimisation"). A small L2 ridge lam_ridge > 0
    adds lam*I to H, regularising away these flat directions.
    Default lam_ridge=0.0 (no regularisation). If set, pass the SAME value to
    sandwich_cov and bootstrap_cov so Cov(ŵ) reflects the regularised estimator.

    Parameters
    ----------
    F_data : (N_data, K+1) tensor — feature matrix for half_B data events
    F_ref  : (N_ref,  K+1) tensor — feature matrix for matched reference events
    lam_ridge : float — L2 ridge coefficient (0 = no regularisation)

    Returns
    -------
    w_hat : (K+1,) torch.float64 tensor — fitted weight vector
    loss  : float — final BCE loss value at ŵ
    """
    F_data_np = F_data.double().cpu().numpy()
    F_ref_np = F_ref.double().cpu().numpy()
    K1 = F_data_np.shape[1]   # = K+1 (K MLP logits + 1 bias)
    n = float(F_data_np.shape[0] + F_ref_np.shape[0])
    lam = float(lam_ridge)

    def fun(w):
        # Logits: s_d = F_data @ w for data events, s_r = F_ref @ w for reference
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        # BCE: -log sigma(s_d) for Y=1 = softplus(-s_d); -log(1-sigma(s_r)) for Y=0 = softplus(s_r)
        bce = (np.logaddexp(0.0, -s_d).sum()
               + np.logaddexp(0.0, s_r).sum())
        if lam > 0.0:
            bce = bce + 0.5 * lam * float(w @ w)  # L2 ridge penalty
        return float(bce) / n   # normalise by total sample size

    def jac(w):
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        gd = _expit(s_d) - 1.0   # gradient contribution from data: sigma(s_d) - 1
        gr = _expit(s_r)          # gradient contribution from ref:  sigma(s_r)
        g = F_data_np.T @ gd + F_ref_np.T @ gr
        if lam > 0.0:
            g = g + lam * w
        return g / n

    def hess(w):
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        sd = _expit(s_d); sr = _expit(s_r)
        # Per-sample sigmoid variance: sigma(s) * (1 - sigma(s)) ∈ (0, 0.25]
        # This is the curvature of the BCE loss for each sample.
        # Maximum at s=0 (p=q), decreasing as |s| grows.
        d_d = sd * (1.0 - sd)
        d_r = sr * (1.0 - sr)
        # H = F^T diag(d) F for data + ref (the expected Fisher information matrix)
        H = ((F_data_np.T * d_d) @ F_data_np
             + (F_ref_np.T * d_r) @ F_ref_np)
        if lam > 0.0:
            H = H + lam * np.eye(K1, dtype=H.dtype)
        return H / n

    # Initialise at w=0: equivalent to r̂(x) = 1 everywhere (no correction to q)
    w0 = np.zeros(K1, dtype=np.float64)
    res = _scipy_minimize(
        fun, w0, jac=jac, hess=hess, method="trust-exact",
        options={"maxiter": max_iter, "gtol": tol, "disp": False},
    )

    w_hat = torch.from_numpy(np.ascontiguousarray(res.x, dtype=np.float64))
    loss = float(res.fun)
    if verbose:
        grad_norm = float(np.linalg.norm(res.jac))
        ridge_tag = f"  λ_ridge={lam:.2e}" if lam > 0.0 else ""
        print(f"  linear head fit (trust-exact): final loss = {loss:.6f}  "
              f"|∇| = {grad_norm:.2e}  nit = {int(res.nit)}  "
              f"nfev = {int(res.nfev)}{ridge_tag}  msg = {str(res.message)[:60]}")
    return w_hat, loss


# ──────────────────────────────────────────────────────────────────────
# Sandwich covariance
# ──────────────────────────────────────────────────────────────────────

def _stack_with_labels(F_data, F_ref):
    """Stack feature matrices and create corresponding Y=1/Y=0 labels."""
    F_all = torch.cat([F_data, F_ref], dim=0).double()
    Y = torch.cat([
        torch.ones(F_data.shape[0], dtype=torch.float64),   # data → Y=1
        torch.zeros(F_ref.shape[0], dtype=torch.float64),   # ref  → Y=0
    ])
    return F_all, Y


def sandwich_cov(w_hat, F_data, F_ref, ridge_rel=1e-8, lam_ridge=0.0):
    """
    Compute the sandwich covariance Cov(ŵ) = H^{-1} J H^{-1}.

    This is the correct asymptotic covariance for ŵ as an M-estimator,
    valid even when the linear-combination model is misspecified (i.e. even
    if no linear combination of the K logits perfectly captures log(p/q)).

    THE TWO MATRICES
    H (the Hessian / "bread"):
        H = F^T diag(sigma(s)(1-sigma(s))) F
        Measures the curvature of the BCE loss at ŵ. Large H means the loss
        is sharply peaked → ŵ is precisely determined by the data.

    J (the score variance / "meat"):
        J = F^T diag((Y - sigma(s))^2) F
        Measures the variability of individual per-sample gradients at ŵ.
        Large J means individual events contribute very different gradients →
        ŵ is sensitive to which specific events were in the dataset.

    SANDWICH vs NAIVE H^{-1}
    If the model is correctly specified, the per-sample gradients are
    uncorrelated and their variance equals the expected Hessian: J = H/n.
    Then H^{-1} J H^{-1} = H^{-1}/n.
    With misspecification (the typical case), J ≠ H/n and the sandwich gives
    larger, more conservative uncertainties — which is what we want for a
    valid GoF test. Using H^{-1} alone would underestimate the uncertainty
    and make the GoF test anti-conservative (too many false positives).

    RIDGE REGULARISATION
    A small relative ridge (eps = ridge_rel * trace(H) / K1) is added to H
    before inversion for numerical stability. This is separate from lam_ridge
    (which regularises the BCE loss itself). If lam_ridge > 0, pass the same
    value here so H includes the ridge term from the regularised estimator.

    Parameters
    ----------
    w_hat     : (K+1,) tensor — fitted weights from fit_linear_head
    F_data    : (N_data, K+1) tensor
    F_ref     : (N_ref,  K+1) tensor
    ridge_rel : float — relative ridge for H inversion (numerical stability)
    lam_ridge : float — same value as used in fit_linear_head (if any)

    Returns
    -------
    Sigma_w : (K+1, K+1) sandwich covariance tensor
    H, J    : raw Hessian and score-variance matrices (for diagnostics)
    """
    F_all, Y = _stack_with_labels(F_data, F_ref)
    s = F_all @ w_hat               # logits at ŵ: s_i = ŵ · f_i
    sig = torch.sigmoid(s)          # predicted probabilities sigma(s_i)
    var = sig * (1.0 - sig)         # per-sample BCE curvature (N,)
    resid = (Y - sig)               # per-sample residual: Y_i - sigma(s_i) (N,)

    # H = F^T diag(var) F  — the expected Hessian (information matrix)
    H = (F_all * var.unsqueeze(1)).T @ F_all
    K1 = H.shape[0]
    if lam_ridge > 0.0:
        # Include ridge term from the regularised estimator
        H = H + float(lam_ridge) * torch.eye(K1, dtype=H.dtype)

    # J = F^T diag(resid^2) F  — the empirical score variance
    J = (F_all * (resid ** 2).unsqueeze(1)).T @ F_all

    # Add a small relative ridge to H before inversion for numerical stability
    # (guards against near-singular H when basis members are highly correlated)
    eps = ridge_rel * (torch.trace(H) / K1)
    H_reg = H + eps * torch.eye(K1, dtype=H.dtype)
    H_inv = torch.linalg.inv(H_reg)

    # Sandwich: Cov(ŵ) = H^{-1} J H^{-1}
    Sigma = H_inv @ J @ H_inv
    return Sigma, H, J


def naive_inv_hessian_cov(w_hat, F_data, F_ref, ridge_rel=1e-8, lam_ridge=0.0):
    """
    Naive covariance estimate H^{-1} — FOR DIAGNOSTIC COMPARISON ONLY.

    This assumes the linear model is correctly specified (J = H/n), which is
    almost never exactly true. It gives smaller uncertainty estimates than the
    sandwich and should NOT be used for the GoF test — use sandwich_cov instead.
    Provided here to quantify how much the misspecification matters: if
    sandwich ≈ naive, the model is a good fit; if sandwich >> naive, there is
    significant misspecification and the sandwich is doing real work.
    """
    F_all, _ = _stack_with_labels(F_data, F_ref)
    s = F_all @ w_hat
    sig = torch.sigmoid(s)
    var = sig * (1.0 - sig)
    H = (F_all * var.unsqueeze(1)).T @ F_all
    K1 = H.shape[0]
    if lam_ridge > 0.0:
        H = H + float(lam_ridge) * torch.eye(K1, dtype=H.dtype)
    eps = ridge_rel * (torch.trace(H) / K1)
    return torch.linalg.inv(H + eps * torch.eye(K1, dtype=H.dtype))


# ──────────────────────────────────────────────────────────────────────
# Bootstrap covariance
# ──────────────────────────────────────────────────────────────────────

def bootstrap_cov(F_data, F_ref, B=200, max_iter=200, lam_ridge=0.0, seed=0,
                  verbose=False):
    """
    Bootstrap covariance: resample (data, ref) B=200 times, refit ŵ each time,
    take the empirical covariance of the B weight vectors.

    WHAT THIS CAPTURES VS THE SANDWICH
    Both sandwich and bootstrap estimate Cov(ŵ) due to finite sample size.
    The sandwich does it analytically from the Hessian and score variance at
    the observed ŵ. The bootstrap does it empirically by re-running the fit
    on perturbed datasets. They should agree when the sample size is large;
    disagreement at smaller N is informative about higher-order effects.

    The MLP basis is implicitly fixed across all bootstrap replicates — we
    work directly with the pre-computed feature matrices F_data and F_ref.
    This is consistent with the sandwich, which also treats the basis as fixed.
    Refitting the full basis (K=128 MLPs) per replicate would be prohibitively
    expensive and would measure something different (unconditional uncertainty
    including basis training variance).

    Parameters
    ----------
    F_data : (N_data, K+1) tensor — feature matrix for half_B data
    F_ref  : (N_ref,  K+1) tensor — feature matrix for matched reference
    B      : int — number of bootstrap replicates (default 200)
    lam_ridge : float — forwarded to fit_linear_head for consistency

    Returns
    -------
    Sigma : (K+1, K+1) empirical covariance of the B weight vectors
    Ws    : (B, K+1) all B bootstrap weight estimates (for diagnostics)
    mean  : (K+1,) mean of the B bootstrap estimates (should ≈ ŵ)
    """
    F_data = F_data.double()
    F_ref = F_ref.double()
    N_d, N_r = F_data.shape[0], F_ref.shape[0]
    K1 = F_data.shape[1]

    gen = torch.Generator().manual_seed(int(seed))
    Ws = torch.zeros(B, K1, dtype=torch.float64)
    for b in range(B):
        # Resample data and reference independently with replacement
        idx_d = torch.randint(0, N_d, (N_d,), generator=gen)
        idx_r = torch.randint(0, N_r, (N_r,), generator=gen)
        Fb_d = F_data[idx_d]
        Fb_r = F_ref[idx_r]
        # Refit the linear head on this bootstrap resample
        w_b, _ = fit_linear_head(
            Fb_d, Fb_r, max_iter=max_iter, lam_ridge=lam_ridge, verbose=False,
        )
        Ws[b] = w_b
        if verbose and (b % max(1, B // 10) == 0):
            print(f"    boot {b+1}/{B}")

    mean = Ws.mean(dim=0)
    centred = Ws - mean
    # Unbiased sample covariance (divide by B-1)
    Sigma = (centred.T @ centred) / (B - 1)
    return Sigma, Ws, mean
