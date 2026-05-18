"""
Wifi linear-head fit and weight covariance.

The basis is frozen upstream (basis.py). Here we:
  1. Build feature matrix F = [1 | l_1(x) ... l_K(x)] for held-out data and
     a fresh reference sample.
  2. Fit w in R^{K+1} via BCE classifier loss (logistic regression on F),
     using scipy's trust-exact (Newton trust-region with the exact Hessian)
     so the optimiser matches the GoF fit.
  3. Estimate Cov(w_hat) two ways:
       - sandwich:  Sigma_w = H^{-1} J H^{-1}
                    H = sum_i sigma(s_i)(1 - sigma(s_i)) f_i f_i^T
                    J = sum_i (Y_i - sigma(s_i))^2 f_i f_i^T
       - bootstrap: resample (data, ref) B times, refit w on each, take the
                    empirical covariance of the bootstrap estimates.
     The basis is held fixed across bootstrap replicates.

Notes:
- BCE here is an M-estimator for the density ratio; the sandwich is the
  correct asymptotic covariance, not a robustness option.
- The Hessian is regularised with a small Tikhonov ridge before inversion to
  guard against near-singular F^T F (basis logits can be highly correlated).
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
    BCE fit for w via scipy trust-exact (Newton trust-region with the exact
    Hessian). Same optimiser as classifier_gof.fit_classifier so the linear
    head and the GoF classifier are minimised consistently. Returns
    (w_hat, final_loss) with loss = mean BCE (+ ridge if lam_ridge > 0).

    Closed forms (per-sample BCE, then summed; Y=1 for data, Y=0 for ref;
    optional sum-form Tikhonov ridge ½·λ·‖w‖² on the BCE_sum, then divided
    by n so units stay clean):
        L(w)   = (Σ_d softplus(-s_d) + Σ_r softplus(s_r) + ½·λ·‖w‖²) / n
        ∇L(w)  = (F_d^T (σ(s_d) - 1) + F_r^T σ(s_r) + λ·w) / n
        H(w)   = (F_d^T diag(σ(s_d)(1-σ(s_d))) F_d
                  + F_r^T diag(σ(s_r)(1-σ(s_r))) F_r + λ·I) / n

    The ridge defaults to 0.0 (no regularisation, matches the original wifi
    behaviour). When the reference is good enough that BCE → log(2),
    optimal w_hat ≈ 0 and the BCE Hessian becomes near-singular along
    redundant feature directions; trust-exact then wanders to absurd values
    in those flat directions. A small λ kills that wander without biasing
    fits where there's real signal — sandwich_cov / naive_inv_hessian_cov /
    bootstrap_cov in this module accept the matching `lam_ridge` so the
    posterior covariance reflects the regularised estimator.
    """
    F_data_np = F_data.double().cpu().numpy()
    F_ref_np = F_ref.double().cpu().numpy()
    K1 = F_data_np.shape[1]
    n = float(F_data_np.shape[0] + F_ref_np.shape[0])
    lam = float(lam_ridge)

    def fun(w):
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        bce = (np.logaddexp(0.0, -s_d).sum()
               + np.logaddexp(0.0, s_r).sum())
        if lam > 0.0:
            bce = bce + 0.5 * lam * float(w @ w)
        return float(bce) / n

    def jac(w):
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        gd = _expit(s_d) - 1.0       # data: σ - 1
        gr = _expit(s_r)              # ref:  σ
        g = F_data_np.T @ gd + F_ref_np.T @ gr
        if lam > 0.0:
            g = g + lam * w
        return g / n

    def hess(w):
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        sd = _expit(s_d); sr = _expit(s_r)
        d_d = sd * (1.0 - sd)
        d_r = sr * (1.0 - sr)
        H = ((F_data_np.T * d_d) @ F_data_np
             + (F_ref_np.T * d_r) @ F_ref_np)
        if lam > 0.0:
            H = H + lam * np.eye(K1, dtype=H.dtype)
        return H / n

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
    F_all = torch.cat([F_data, F_ref], dim=0).double()
    Y = torch.cat([
        torch.ones(F_data.shape[0], dtype=torch.float64),
        torch.zeros(F_ref.shape[0], dtype=torch.float64),
    ])
    return F_all, Y


def sandwich_cov(w_hat, F_data, F_ref, ridge_rel=1e-8, lam_ridge=0.0):
    """
    Sigma_w = H^{-1} J H^{-1}, with a small relative Tikhonov ridge added to H
    before inversion (epsilon = ridge_rel * trace(H) / dim). When the linhead
    was fit with a ridge penalty (`lam_ridge` in fit_linear_head's sum-form
    units), pass the same value here so H reflects the regularised estimator's
    Hessian (H_data + λ·I); J is unchanged because the ridge contributes
    deterministically to the score and so doesn't enter the score variance.

    Returns
    -------
    Sigma_w : (K+1, K+1) tensor
    H, J    : raw matrices (for diagnostics; H already includes lam_ridge·I
              if lam_ridge > 0)
    """
    F_all, Y = _stack_with_labels(F_data, F_ref)
    s = F_all @ w_hat
    sig = torch.sigmoid(s)
    var = sig * (1.0 - sig)                       # (N,)
    resid = (Y - sig)                             # (N,)
    # H = F^T diag(var) F  (+ λ·I when the linhead was regularised)
    H = (F_all * var.unsqueeze(1)).T @ F_all
    K1 = H.shape[0]
    if lam_ridge > 0.0:
        H = H + float(lam_ridge) * torch.eye(K1, dtype=H.dtype)
    # J = F^T diag(resid^2) F
    J = (F_all * (resid ** 2).unsqueeze(1)).T @ F_all

    eps = ridge_rel * (torch.trace(H) / K1)
    H_reg = H + eps * torch.eye(K1, dtype=H.dtype)
    H_inv = torch.linalg.inv(H_reg)
    Sigma = H_inv @ J @ H_inv
    return Sigma, H, J


def naive_inv_hessian_cov(w_hat, F_data, F_ref, ridge_rel=1e-8, lam_ridge=0.0):
    """
    Standard inverse-Hessian H^{-1} for diagnostic comparison only. Adds the
    same `lam_ridge·I` to H as sandwich_cov when the linhead was regularised.
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
    Resample (data, ref) with replacement B times, refit w on each, return
    the empirical covariance of the bootstrap w-estimates. `lam_ridge` is
    forwarded to fit_linear_head so each bootstrap fit uses the same
    regulariser as the headline fit.

    Basis is implicitly fixed (we work directly with feature matrices).
    """
    F_data = F_data.double()
    F_ref = F_ref.double()
    N_d, N_r = F_data.shape[0], F_ref.shape[0]
    K1 = F_data.shape[1]

    gen = torch.Generator().manual_seed(int(seed))
    Ws = torch.zeros(B, K1, dtype=torch.float64)
    for b in range(B):
        idx_d = torch.randint(0, N_d, (N_d,), generator=gen)
        idx_r = torch.randint(0, N_r, (N_r,), generator=gen)
        Fb_d = F_data[idx_d]
        Fb_r = F_ref[idx_r]
        w_b, _ = fit_linear_head(
            Fb_d, Fb_r, max_iter=max_iter, lam_ridge=lam_ridge, verbose=False,
        )
        Ws[b] = w_b
        if verbose and (b % max(1, B // 10) == 0):
            print(f"    boot {b+1}/{B}")

    mean = Ws.mean(dim=0)
    centred = Ws - mean
    Sigma = (centred.T @ centred) / (B - 1)
    return Sigma, Ws, mean
