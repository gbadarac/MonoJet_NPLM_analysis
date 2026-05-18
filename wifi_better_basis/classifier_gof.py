"""
Classifier goodness-of-fit test for the wifi-reweighted model.

Given the trained wifi pipeline (frozen MLP basis, linear head w_hat with
covariance Sigma_w), this module fits two BCE classifiers between observed
data (Y=1) and reference samples drawn from q (Y=0):

  Denominator: f_den(x; w) = w · features(x)              (just the wifi
                                                           reweighter, with
                                                           a Gaussian prior
                                                           on (w − w_hat))
  Numerator:   f_num(x; w, b) = w · features(x)
                              + sum_j b_j G_j(x)          (perturbations
                                                           added in
                                                           log-density space)

Test statistic:
    t = 2 · [ L_den(w_den*) − L_num(w_num*, b_num*) ]

where L is the BCE objective plus the prior penalties:
    L = − sum_data log σ(f) − sum_ref log(1 − σ(f))
        + ½ (w − w_hat)^T Σ_w_eff^{-1} (w − w_hat)
        + ½ λ_pert ‖b‖²

Three variants control how much the wifi weights can drift:
  - constrained: Σ_w_eff = Σ_w
  - free:        Σ_w_eff = c · Σ_w   (c = FREE_COV_INFLATE, looser prior)
  - frozen:      w ≡ w_hat            (no drift; only b is fit on the num side)

Calibration: replace "data" with toys drawn from p_hat = r_hat·q (via SIR on
q), keep the reference draws from q, run the same num/den fits per toy. The
empirical distribution of t over toys is the null distribution.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize as _scipy_minimize
from scipy.special import expit as _expit


# Heuristic threshold for declaring a fit "well-converged" — the objective is
# convex and well-conditioned, so a healthy fit should land far below this.
# Anything above is a red flag worth surfacing.
GRAD_NORM_OK = 1e-4


# ──────────────────────────────────────────────────────────────────────
# Perturbation kernels (Gaussian RBF in log-density space)
# ──────────────────────────────────────────────────────────────────────

class PerturbationRBF(nn.Module):
    """
    Frozen-shape isotropic Gaussian kernels at fixed centroids. Used as
    additive perturbations to the classifier logit. Each kernel integrates
    to 1 in density space, but for an MLC GoF the normalisation only
    matters for downstream interpretation; we keep it for parity.
    """

    def __init__(self, centroids, sigma):
        super().__init__()
        c = torch.as_tensor(centroids, dtype=torch.float64)
        self.register_buffer("centroids", c)        # (M, d)
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float64))
        d = c.shape[1]
        norm_const = 1.0 / ((2 * math.pi) ** (d / 2) * float(sigma) ** d)
        self.register_buffer("norm_const",
                             torch.tensor(norm_const, dtype=torch.float64))

    @property
    def M(self):
        return int(self.centroids.shape[0])

    def matrix(self, x):
        """Return G[i, j] = G_j(x_i), shape (N, M)."""
        x = x.double()
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)        # (N, M, d)
        d2 = (diff ** 2).sum(dim=2)
        return self.norm_const * torch.exp(-0.5 * d2 / self.sigma ** 2)


# ──────────────────────────────────────────────────────────────────────
# BCE loss with prior penalties
# ──────────────────────────────────────────────────────────────────────

def _ridge_penalty_inv(Sigma, ridge_rel):
    """Return (Sigma + eps I)^{-1} with a small relative Tikhonov ridge."""
    K1 = Sigma.shape[0]
    eps = ridge_rel * (torch.trace(Sigma) / K1)
    return torch.linalg.inv(Sigma + eps * torch.eye(K1, dtype=Sigma.dtype))


# ──────────────────────────────────────────────────────────────────────
# Single-fit optimiser: BCE + prior on Δw + L2 ridge on b
# ──────────────────────────────────────────────────────────────────────

def fit_classifier(F_data, F_ref, w_hat, Sigma_w_eff,
                   G_data=None, G_ref=None, lam_pert=0.0,
                   frozen=False, max_iter=500, tol=1e-9,
                   ridge_rel=1e-8, verbose=False):
    """
    Optimise the BCE-with-priors loss for one (variant, num/den) configuration.

    The objective
        L(w, b) = Σ_data softplus(−s_d) + Σ_ref softplus(s_r)
                + ½ (w − w_hat)^T Σ_w_eff^{-1} (w − w_hat)
                + ½ λ_pert ‖b‖²
    with logit s_i = w · F_i + b · G_i, is jointly convex in (w, b) and the
    Hessian has a clean closed form, so we use scipy's trust-exact (Newton-
    style trust-region with the exact Hessian).

    Closed forms used by the optimiser:
        ∇_w L = F_d^T (σ(s_d) − 1) + F_r^T σ(s_r) + Σ_w_eff^{-1} (w − w_hat)
        ∇_b L = G_d^T (σ(s_d) − 1) + G_r^T σ(s_r) + λ_pert · b
        H_ww  = F_d^T D_d F_d + F_r^T D_r F_r + Σ_w_eff^{-1}
        H_bb  = G_d^T D_d G_d + G_r^T D_r G_r + λ_pert · I
        H_wb  = F_d^T D_d G_d + F_r^T D_r G_r
    where D_x = diag(σ(s_x)(1 − σ(s_x))).

    If G_data/G_ref are None, fits the denominator (no perturbation).
    If frozen=True, fixes w at w_hat and only optimises b (if a perturbation
    is provided — otherwise the loss is constant and we evaluate it once).

    Returns a dict with keys:
        loss        : final objective value, float
        w           : final w, (K+1,) tensor
        b           : final b, (M_pert,) tensor or None
        loss_hist   : per-fun-call loss values (np.ndarray)
        grad_norm   : ‖∇L‖_2 at the final point
        n_eval      : scipy.optimize nfev (number of fun calls)
        n_iter      : scipy.optimize nit  (number of accepted iterations)
        converged   : grad_norm < GRAD_NORM_OK
        hit_max_iter: True if scipy terminated by maxiter rather than gtol
    """
    F_data = F_data.double()
    F_ref = F_ref.double()
    F_data_np = F_data.cpu().numpy()
    F_ref_np = F_ref.cpu().numpy()
    K1 = F_data_np.shape[1]

    has_pert = G_data is not None
    if has_pert:
        G_data = G_data.double()
        G_ref = G_ref.double()
        G_data_np = G_data.cpu().numpy()
        G_ref_np = G_ref.cpu().numpy()
        M = G_data_np.shape[1]

    w_hat_np = w_hat.cpu().numpy().astype(np.float64)

    if not frozen:
        Sw_inv_t = _ridge_penalty_inv(Sigma_w_eff.double(), ridge_rel)
        Sw_inv = Sw_inv_t.cpu().numpy()

    # Trivial case: frozen + no perturbation -> single-evaluation result.
    if frozen and not has_pert:
        s_d = F_data_np @ w_hat_np
        s_r = F_ref_np @ w_hat_np
        L = float(np.logaddexp(0.0, -s_d).sum() + np.logaddexp(0.0, s_r).sum())
        return {
            "loss": L,
            "w": torch.from_numpy(w_hat_np.copy()),
            "b": None,
            "loss_hist": np.array([L], dtype=np.float64),
            "grad_norm": 0.0,
            "n_eval": 1, "n_iter": 0,
            "converged": True, "hit_max_iter": False,
        }

    # Parameter vector layout depends on which blocks are variable.
    if frozen:                       # only b
        x0 = np.zeros(M, dtype=np.float64)
    elif not has_pert:               # only w (denominator)
        x0 = w_hat_np.copy()
    else:                            # both w and b (numerator)
        x0 = np.concatenate([w_hat_np, np.zeros(M, dtype=np.float64)])

    def _split(x):
        if frozen:
            return w_hat_np, x
        if not has_pert:
            return x, None
        return x[:K1], x[K1:]

    loss_hist = []

    def fun(x):
        w, b = _split(x)
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        if b is not None:
            s_d = s_d + G_data_np @ b
            s_r = s_r + G_ref_np @ b
        loss = float(np.logaddexp(0.0, -s_d).sum()
                     + np.logaddexp(0.0, s_r).sum())
        if not frozen:
            dw = w - w_hat_np
            loss += 0.5 * float(dw @ Sw_inv @ dw)
        if has_pert and lam_pert > 0:
            loss += 0.5 * lam_pert * float(b @ b)
        loss_hist.append(loss)
        return loss

    def jac(x):
        w, b = _split(x)
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        if b is not None:
            s_d = s_d + G_data_np @ b
            s_r = s_r + G_ref_np @ b
        gd = _expit(s_d) - 1.0           # data: σ−1
        gr = _expit(s_r)                  # ref:  σ
        parts = []
        if not frozen:
            gw = F_data_np.T @ gd + F_ref_np.T @ gr + Sw_inv @ (w - w_hat_np)
            parts.append(gw)
        if has_pert:
            gb = G_data_np.T @ gd + G_ref_np.T @ gr
            if lam_pert > 0:
                gb = gb + lam_pert * b
            parts.append(gb)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def hess(x):
        w, b = _split(x)
        s_d = F_data_np @ w
        s_r = F_ref_np @ w
        if b is not None:
            s_d = s_d + G_data_np @ b
            s_r = s_r + G_ref_np @ b
        sd = _expit(s_d); sr = _expit(s_r)
        d_d = sd * (1.0 - sd)
        d_r = sr * (1.0 - sr)
        if not frozen and has_pert:
            n = K1 + M
            H = np.zeros((n, n))
            H[:K1, :K1] = ((F_data_np.T * d_d) @ F_data_np
                           + (F_ref_np.T * d_r) @ F_ref_np
                           + Sw_inv)
            H[K1:, K1:] = ((G_data_np.T * d_d) @ G_data_np
                           + (G_ref_np.T * d_r) @ G_ref_np)
            if lam_pert > 0:
                H[K1:, K1:] += lam_pert * np.eye(M)
            FwG = ((F_data_np.T * d_d) @ G_data_np
                   + (F_ref_np.T * d_r) @ G_ref_np)
            H[:K1, K1:] = FwG
            H[K1:, :K1] = FwG.T
            return H
        if not frozen:
            return ((F_data_np.T * d_d) @ F_data_np
                    + (F_ref_np.T * d_r) @ F_ref_np
                    + Sw_inv)
        # frozen, only b
        H = ((G_data_np.T * d_d) @ G_data_np
             + (G_ref_np.T * d_r) @ G_ref_np)
        if lam_pert > 0:
            H = H + lam_pert * np.eye(M)
        return H

    res = _scipy_minimize(
        fun, x0, jac=jac, hess=hess, method="trust-exact",
        options={"maxiter": max_iter, "gtol": tol},
    )

    w_final, b_final = _split(res.x)
    if frozen:
        w_final_t = torch.from_numpy(w_hat_np.copy())
    else:
        w_final_t = torch.from_numpy(w_final.copy())
    b_final_t = torch.from_numpy(b_final.copy()) if b_final is not None else None

    grad_norm = float(np.linalg.norm(jac(res.x)))
    n_iter = int(getattr(res, "nit", 0))
    hit_max_iter = (n_iter >= max_iter)

    if verbose:
        flag = "OK" if grad_norm < GRAD_NORM_OK else "WARN"
        msg = str(getattr(res, "message", ""))[:60]
        print(f"      trust-exact [{flag}] loss={float(res.fun):.6f}  "
              f"||∇||={grad_norm:.2e}  n_iter={n_iter}  "
              f"n_eval={len(loss_hist)}  msg={msg}")

    return {
        "loss": float(res.fun),
        "w": w_final_t,
        "b": b_final_t,
        "loss_hist": np.array(loss_hist, dtype=np.float64),
        "grad_norm": grad_norm,
        "n_eval": len(loss_hist),
        "n_iter": n_iter,
        "converged": grad_norm < GRAD_NORM_OK,
        "hit_max_iter": hit_max_iter,
    }


# ──────────────────────────────────────────────────────────────────────
# Test statistic for one (data, ref) pair, one variant at a time
# ──────────────────────────────────────────────────────────────────────

# Two variants with distinct semantics. With the wifi_better_basis run_gof
# orchestration, they also use DIFFERENT toy ensembles (constrained: posterior-
# predictive null with w_k ~ N(ŵ, Σ_w); frozen: plug-in null at w = ŵ), so
# there is no longer a cross-variant ordering to check.
VARIANTS = ("constrained", "frozen")


def _variant_settings(variant, Sigma_w):
    """Map variant name to (Sigma_w_eff, frozen) for fit_classifier."""
    if variant == "constrained":
        return Sigma_w, False
    if variant == "frozen":
        # Sigma_w_eff is unused when frozen=True (w is fixed at w_hat).
        return Sigma_w, True
    raise ValueError(f"unknown variant: {variant!r}")


def test_stat_one_variant(variant, F_data, F_ref, G_data, G_ref,
                          w_hat, Sigma_w,
                          lam_pert=0.0, max_iter=500, tol=1e-9,
                          ridge_rel=1e-8, verbose=False):
    """
    Run num and den fits for a single variant on a single (data, ref) pair.

    Returns
    -------
    dict with keys:
        L_den, L_num, t,
        w_den, w_num, b_num   (numpy arrays; b_num may be None),
        den, num              (full fit-result dicts from fit_classifier).
    """
    Sw_eff, frozen = _variant_settings(variant, Sigma_w)

    if verbose:
        print(f"    [{variant}] denominator fit")
    den = fit_classifier(
        F_data, F_ref, w_hat, Sw_eff, G_data=None, G_ref=None,
        frozen=frozen, max_iter=max_iter, tol=tol,
        ridge_rel=ridge_rel, verbose=verbose,
    )
    if verbose:
        print(f"    [{variant}] numerator fit")
    num = fit_classifier(
        F_data, F_ref, w_hat, Sw_eff, G_data=G_data, G_ref=G_ref,
        lam_pert=lam_pert, frozen=frozen, max_iter=max_iter, tol=tol,
        ridge_rel=ridge_rel, verbose=verbose,
    )
    t = 2.0 * (den["loss"] - num["loss"])
    if verbose:
        print(f"    [{variant}] t = {t:.4f}")
    return {
        "L_den": den["loss"],
        "L_num": num["loss"],
        "t": float(t),
        "w_den": den["w"].cpu().numpy(),
        "w_num": num["w"].cpu().numpy(),
        "b_num": (num["b"].cpu().numpy() if num["b"] is not None else None),
        "den": den,
        "num": num,
    }


def test_stat_all_variants(F_data, F_ref, G_data, G_ref,
                           w_hat, Sigma_w,
                           lam_pert=0.0, max_iter=500, tol=1e-9,
                           ridge_rel=1e-8, verbose=False):
    """
    Convenience wrapper for the OBSERVED test statistic only — runs both
    variants on the same (F_data, F_ref). The toy loop in run_gof.py uses
    test_stat_one_variant directly because the two variants now consume
    different toy DGPs.
    """
    return {
        v: test_stat_one_variant(
            v, F_data, F_ref, G_data, G_ref, w_hat, Sigma_w,
            lam_pert=lam_pert, max_iter=max_iter, tol=tol,
            ridge_rel=ridge_rel, verbose=verbose,
        )
        for v in VARIANTS
    }


# ──────────────────────────────────────────────────────────────────────
# Toy generation: SIR on q
# ──────────────────────────────────────────────────────────────────────

def sir_toy(F_q_pool, X_q_pool, w_hat, N_target, gen_seed):
    """
    Multinomial-resample N_target points from a fixed q-pool, weighted by
    r_hat(x_i) = exp(features(x_i) · w_hat). Returns (X_toy, F_toy).
    """
    s = F_q_pool @ w_hat
    s = s - s.max()
    p = torch.exp(s)
    p = p / p.sum()
    gen = torch.Generator().manual_seed(int(gen_seed))
    idx = torch.multinomial(p, num_samples=N_target, replacement=True, generator=gen)
    return X_q_pool[idx], F_q_pool[idx]
