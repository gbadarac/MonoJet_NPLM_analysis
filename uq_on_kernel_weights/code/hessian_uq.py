"""
Hessian-based uncertainty quantification for HIKER.

Supports two gauge-fixing modes:
  1. Normalization kernel (K_norm is not None):
     f(x) = sum_l K^(l) @ w^(l) + (1 - sum(w)) * K_norm(x)
     Hessian is full-rank. Inverted via compute_covariance_regularised().

  2. Post-hoc normalization (K_norm is None):
     f(x) = sum_l K^(l) @ w^(l) / Z,  Z = sum(w)
     Hessian has one flat direction (1,...,1). Inverted via
     compute_covariance_projected() which projects out the null mode.

Pointwise predictive variance:
  - predictive_variance(): for single models (both gauge modes)
  - predictive_variance_ensemble(): for ensembles with per-seed norm kernels
"""

import torch
from torch.autograd import grad
import numpy as np


# ---------------------------------------------------------------------------
# Hessian computation
# ---------------------------------------------------------------------------

def compute_hessian(loss, params):
    """Row-by-row Hessian via nested autograd."""
    M = params.numel()
    grad1 = grad(loss, params, create_graph=True)[0]
    rows = []
    for i in range(M):
        row = grad(grad1[i], params, retain_graph=True)[0]
        rows.append(row.detach())
    return torch.stack(rows).double()


# ---------------------------------------------------------------------------
# NLL as a function of free weights
# ---------------------------------------------------------------------------

def nll_gauge(w, K_list, K_norm, layer_sizes):
    """
    NLL for HIKER. Handles both norm-kernel and post-hoc modes.

    Parameters
    ----------
    w : Tensor (M_total,)
    K_list : list of Tensor (N, M_l)
    K_norm : Tensor (N, 1) or None
    layer_sizes : list of int
    """
    g = torch.zeros(K_list[0].shape[0], dtype=torch.float64, device=w.device)
    offset = 0
    for l, M_l in enumerate(layer_sizes):
        w_l = w[offset:offset + M_l]
        g = g + K_list[l] @ w_l
        offset += M_l

    if K_norm is not None:
        # Norm kernel mode
        w_n = 1.0 - w.sum()
        f = g + w_n * K_norm.squeeze(1)
    else:
        # Post-hoc normalization
        Z = w.sum()
        f = g / (Z + 1e-30)

    return -torch.log(f + 1e-10).sum()


def make_nll_fn(K_list, K_norm, layer_sizes):
    """Build a closure NLL(w) for autograd."""
    K_list_ = [K.double().detach() for K in K_list]
    Kn_ = K_norm.double().detach() if K_norm is not None else None

    def nll_fn(w):
        return nll_gauge(w, K_list_, Kn_, layer_sizes)

    return nll_fn


# ---------------------------------------------------------------------------
# Covariance from Hessian inversion
# ---------------------------------------------------------------------------

def compute_covariance(H, verbose=True):
    """Direct Hessian inversion (for norm-kernel mode where H is full-rank)."""
    M = H.shape[0]
    H = H.double()
    H_sym = 0.5 * (H + H.T)
    evals = torch.linalg.eigvalsh(H_sym)

    if verbose:
        print(f"  Hessian eigenvalues: min={evals[0]:.4e}, max={evals[-1]:.4e}")
        cond = evals[-1] / evals[0] if evals[0] > 0 else float("inf")
        print(f"  Condition number: {cond:.4e}")
        n_neg = (evals < 0).sum().item()
        if n_neg > 0:
            print(f"  WARNING: {n_neg} negative eigenvalue(s) detected")

    eye = torch.eye(M, dtype=torch.float64, device=H.device)
    cov = torch.linalg.solve(H_sym, eye)
    cov = 0.5 * (cov + cov.T)

    diagnostics = {
        "eigenvalues": evals.cpu().numpy(),
        "condition_number": (evals[-1] / evals[0]).item() if evals[0] > 0 else float("inf"),
        "n_negative_eigenvalues": (evals < 0).sum().item(),
    }
    return cov, diagnostics


def compute_covariance_regularised(H, eps=1e-6, verbose=True):
    """Eigenvalue-regularised inversion (robust, works for norm-kernel mode)."""
    M = H.shape[0]
    H = 0.5 * (H + H.T).double()
    evals, evecs = torch.linalg.eigh(H)

    if verbose:
        print(f"  Hessian eigenvalues: min={evals[0]:.4e}, max={evals[-1]:.4e}")

    evals_reg = torch.clamp(evals, min=eps)
    cov = evecs @ torch.diag(1.0 / evals_reg) @ evecs.T
    cov = 0.5 * (cov + cov.T)

    diagnostics = {
        "eigenvalues": evals.cpu().numpy(),
        "eigenvalues_regularised": evals_reg.cpu().numpy(),
        "condition_number": (evals_reg[-1] / evals_reg[0]).item(),
        "n_regularised": (evals < eps).sum().item(),
    }
    if verbose:
        print(f"  Regularised {diagnostics['n_regularised']} eigenvalue(s) (floor={eps:.1e})")
        print(f"  Condition number after regularisation: {diagnostics['condition_number']:.4e}")

    return cov, diagnostics


def compute_covariance_projected(H, eig_floor=1e-6, verbose=True):
    """
    Projected covariance for post-hoc normalization (no norm kernel).

    Projects out the (1,...,1) flat direction and pseudo-inverts.
    """
    M = H.shape[0]
    H = 0.5 * (H + H.T).double()
    device = H.device

    if verbose:
        raw_eigs = torch.linalg.eigvalsh(H)
        print(f"  Raw Hessian eigenvalues: min={raw_eigs[0]:.4e}, max={raw_eigs[-1]:.4e}")

    # Projector P = I - 11^T/M
    ones = torch.ones(M, dtype=torch.float64, device=device)
    P = torch.eye(M, dtype=torch.float64, device=device) - torch.outer(ones, ones) / M

    H_proj = P @ H @ P
    evals, evecs = torch.linalg.eigh(H_proj)

    if verbose:
        print(f"  Projected eigenvalues: min={evals[0]:.4e}, max={evals[-1]:.4e}")

    mask = evals > eig_floor
    n_null = (~mask).sum().item()
    evals_kept = evals[mask]
    evecs_kept = evecs[:, mask]

    if verbose:
        print(f"  Discarded {n_null} null eigenvalue(s)")
        if evals_kept.numel() > 1:
            print(f"  Condition number: {(evals_kept[-1] / evals_kept[0]).item():.4e}")

    cov = evecs_kept @ torch.diag(1.0 / evals_kept) @ evecs_kept.T
    cov = P @ cov @ P
    cov = 0.5 * (cov + cov.T)

    diagnostics = {
        "eigenvalues": torch.linalg.eigvalsh(H).cpu().numpy(),
        "projected_eigenvalues": evals.cpu().numpy(),
        "kept_eigenvalues": evals_kept.cpu().numpy(),
        "n_null_removed": n_null,
        "condition_number": (evals_kept[-1] / evals_kept[0]).item() if evals_kept.numel() > 1 else float("inf"),
        "n_regularised": 0,
    }
    return cov, diagnostics


# ---------------------------------------------------------------------------
# Pointwise predictive variance
# ---------------------------------------------------------------------------

def predictive_variance(x_eval, model, cov_w):
    """
    Pointwise predictive variance of the density.

    With norm kernel:   df/dw_j = K_j(x) - K_norm(x)
    Without norm kernel: df/dw_j = (K_j(x) - f(x)) / Z
    """
    with torch.no_grad():
        f = model.density(x_eval)
        J_parts = []

        if model.use_norm_kernel:
            K_norm = model.norm_kernel.kernels(x_eval)  # (N, 1)
            for layer in model.layers:
                K = layer.kernel_layer.kernels(x_eval)
                J_parts.append(K - K_norm)
        else:
            Z = model.Z()
            for layer in model.layers:
                K = layer.kernel_layer.kernels(x_eval)
                J_parts.append((K - f.unsqueeze(1)) / (Z + 1e-30))

        J = torch.cat(J_parts, dim=1)
        JC = J @ cov_w.double()
        var_f = (JC * J).sum(dim=1)

    return var_f, f


def predictive_variance_ensemble(x_eval, model, cov_w, ensemble_info):
    """
    Predictive variance for an ensemble with per-seed norm kernels.

    For seed s with norm kernel:
        f_s(x) = sum_i w_i^(s) K_i^(s)(x) + (1/N - sum(w_s)) K_norm^(s)(x)
        df/dw_j^(s) = K_j^(s)(x) - K_norm^(s)(x)
    """
    from hiker import ensemble_density

    with torch.no_grad():
        f = ensemble_density(model, x_eval, ensemble_info)
        J_parts = []

        if ensemble_info["use_norm_per_seed"] and ensemble_info["norm_kernels"] is not None:
            # Per-seed norm kernel Jacobian
            layer_idx = 0
            for s, (start, end) in enumerate(ensemble_info["seed_boundaries"]):
                K_norm_s = ensemble_info["norm_kernels"][s].kernels(x_eval)  # (N, 1)
                M_s = end - start
                # Collect kernels for this seed's layers
                while layer_idx < len(model.layers):
                    seed_of_layer, _ = ensemble_info["layer_map"][layer_idx]
                    if seed_of_layer != s:
                        break
                    K = model.layers[layer_idx].kernel_layer.kernels(x_eval)
                    J_parts.append(K - K_norm_s)
                    layer_idx += 1
        else:
            # Standard no-norm-kernel Jacobian
            Z = model.Z()
            for layer in model.layers:
                K = layer.kernel_layer.kernels(x_eval)
                J_parts.append((K - f.unsqueeze(1)) / (Z + 1e-30))

        J = torch.cat(J_parts, dim=1)
        JC = J @ cov_w.double()
        var_f = (JC * J).sum(dim=1)

    return var_f, f


def predictive_bands(x_eval, model, cov_w, n_sigma=2):
    var_f, f_eval = predictive_variance(x_eval, model, cov_w)
    sigma_f = torch.sqrt(var_f.clamp(min=0)).cpu().numpy()
    f_np = f_eval.cpu().numpy()
    return f_np, f_np + n_sigma * sigma_f, f_np - n_sigma * sigma_f, sigma_f
