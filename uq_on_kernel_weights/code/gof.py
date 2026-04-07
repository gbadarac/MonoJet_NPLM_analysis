"""
Learned likelihood-ratio goodness-of-fit test (NPLM framework).

Supports all HIKER modes: single model with/without norm kernel, and
ensembles with per-seed norm kernels.

The test compares two hypotheses via their likelihood ratio:

Null (denominator): the data is described by the fitted HIKER density,
  with trainable weights constrained by a Gaussian prior centred at the
  best-fit values w_hat with covariance Cov(w_hat) from the Hessian.
  The density is f(x) = g(x)/Z (or with norm kernel). The constraint
  penalises deviations from w_hat, encoding the model's uncertainty.

Alternative (numerator): same as the null, plus a perturbation term
  parameterised by extra Gaussian kernels with trainable coefficients.
  The perturbation is added AFTER normalization:
      p(x) = f(x) + sum_j b_j G_j(x)
  so its magnitude is relative to the normalized density and independent
  of the overall scale Z. If the null is correct, the perturbation
  coefficients stay near zero and the test statistic is small.

Both numerator and denominator use the normalized density f(x) = g(x)/Z
(or f(x) with norm kernel) for the log-likelihood.

Test statistic: t = 2 * (log L_alt - log L_null), calibrated empirically
via toy pseudo-experiments.

Classes:
  PerturbationKernelLayer — extra kernels for the numerator
  GoFModel — LRT model (handles norm kernel, per-seed norm, post-hoc modes)

Functions:
  train_gof_model() — train a GoFModel with coefficient history tracking
  run_gof_test() — full LRT pipeline (den + num + test statistic)
"""

import math
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import chi2, norm


# ──────────────────────────────────────────────────────────────────────
# Perturbation kernels for the numerator
# ──────────────────────────────────────────────────────────────────────

class PerturbationKernelLayer(nn.Module):
    """Extra Gaussian kernels for the numerator model's perturbation term."""

    def __init__(self, centers, sigma, train_centers=False, clip_coeffs=None):
        super().__init__()
        self.centers = nn.Parameter(centers.double(), requires_grad=train_centers)
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float64))
        d = centers.shape[1]
        self.norm_const = 1.0 / ((2 * math.pi) ** (d / 2) * self.sigma ** d)
        M = centers.shape[0]
        self.coefficients = nn.Parameter(torch.zeros(M, dtype=torch.float64), requires_grad=True)
        self.clip = clip_coeffs

    def kernels(self, x):
        x = x.double()
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return self.norm_const * torch.exp(-0.5 * dist_sq / self.sigma ** 2)

    def forward(self, x):
        """Perturbation: sum_j b_j G_j(x), gauge-fixed: b_j -> b_j - mean(b)."""
        K = self.kernels(x)
        b = self.coefficients - self.coefficients.mean()
        return (K * b.unsqueeze(0)).sum(dim=1)

    def forward_from_kernels(self, kern):
        """Same as forward but from precomputed kernel matrix."""
        b = self.coefficients - self.coefficients.mean()
        return kern @ b

    def clip_coefficients(self):
        if self.clip is not None:
            self.coefficients.data.clamp_(-self.clip, self.clip)


# ──────────────────────────────────────────────────────────────────────
# GoF model (TAU)
# ──────────────────────────────────────────────────────────────────────

class GoFModel(nn.Module):
    """
    Likelihood-ratio GoF model with single global normalization kernel.

    The density is:
        p(x) = sum_l K^(l) @ w^(l) + (1 - sum(w)) * K_norm  [+ perturbation]

    Parameters
    ----------
    K_list : list of Tensor (N, M_l) — precomputed main kernels per layer
    K_norm : Tensor (N, 1) — single global normalization kernel
    layer_sizes : list of int — [M_0, M_1, ...]
    weights_init : Tensor (M_total,) — concatenated fitted weights
    weights_cov : Tensor (M_total, M_total)
    perturbation_net : PerturbationKernelLayer or None
    lambda_net : float
    train_weights : bool
    use_constraint : bool
    """

    def __init__(self, K_list, K_norm, layer_sizes, weights_init, weights_cov,
                 perturbation_net=None, lambda_net=0.0, train_weights=True,
                 use_constraint=True, ensemble_info=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.has_norm_kernel = K_norm is not None
        for l, K in enumerate(K_list):
            self.register_buffer(f"K_{l}", K.double())
        if self.has_norm_kernel:
            self.register_buffer("K_norm", K_norm.double())
        self.n_layers = len(layer_sizes)

        # Per-seed norm kernels (ensemble with USE_NORM_KERNEL=True per seed)
        self.ensemble_info = ensemble_info
        if ensemble_info is not None and ensemble_info.get("norm_kernels_eval") is not None:
            # Store precomputed norm kernel evaluations at data points
            for s, Kn_s in enumerate(ensemble_info["norm_kernels_eval"]):
                self.register_buffer(f"K_norm_seed_{s}", Kn_s.double())
            self.seed_boundaries = ensemble_info["seed_boundaries"]
            self.N_ens = ensemble_info["N_ens"]
            self.has_per_seed_norm = True
        else:
            self.has_per_seed_norm = False

        M = weights_init.shape[0]
        self.weights = nn.Parameter(weights_init.clone().double(), requires_grad=train_weights)
        self.lambda_net = lambda_net
        self.perturbation_net = perturbation_net
        self.train_weights = train_weights
        self.use_constraint = use_constraint

        # Build Gaussian prior with robust Cholesky
        cov = 0.5 * (weights_cov + weights_cov.T).double()
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        eye = torch.eye(M, dtype=torch.float64, device=cov.device)
        jitter = 1e-6 * (cov.diagonal().abs().mean() + 1.0)
        L = None
        for _ in range(7):
            try:
                L = torch.linalg.cholesky(cov + jitter * eye)
                break
            except RuntimeError:
                jitter *= 10.0
        if L is None:
            evals, evecs = torch.linalg.eigh(cov)
            evals = torch.clamp(evals, min=1e-8)
            cov = (evecs * evals) @ evecs.T
            L = torch.linalg.cholesky(cov)
        self.prior = torch.distributions.MultivariateNormal(
            weights_init.clone().double(), scale_tril=L
        )

    def _base_density(self):
        """Base density g(x) = sum_l K_l @ w_l [+ norm kernel(s)], no perturbation."""
        w = self.weights
        N = getattr(self, "K_0").shape[0]
        g = torch.zeros(N, dtype=torch.float64, device=w.device)
        offset = 0
        for l in range(self.n_layers):
            M_l = self.layer_sizes[l]
            w_l = w[offset:offset + M_l]
            g = g + getattr(self, f"K_{l}") @ w_l
            offset += M_l

        if self.has_per_seed_norm:
            # Per-seed norm kernels: each seed's norm weight = 1/N_ens - sum(w_s)
            for s, (start, end) in enumerate(self.seed_boundaries):
                w_s = w[start:end]
                w_norm_s = 1.0 / self.N_ens - w_s.sum()
                Kn_s = getattr(self, f"K_norm_seed_{s}")  # (N, 1)
                g = g + w_norm_s * Kn_s.squeeze(1)
        elif self.has_norm_kernel:
            w_norm = 1.0 - w.sum()
            g = g + w_norm * self.K_norm.squeeze(1)
        return g

    def _perturbation(self, pert_kernels=None, x_data=None):
        """Perturbation term (0 if no perturbation net)."""
        if self.perturbation_net is None:
            return 0.0
        if pert_kernels is not None:
            return self.perturbation_net.forward_from_kernels(pert_kernels)
        elif x_data is not None:
            return self.perturbation_net(x_data)
        return 0.0

    def density(self, x_data=None, pert_kernels=None):
        """
        Normalized density: p(x) = f(x) + perturbation(x).
        Perturbation is added AFTER normalization so its magnitude
        is relative to the normalized density, independent of Z.
        """
        g = self._base_density()
        if self.has_norm_kernel or self.has_per_seed_norm:
            p = g  # already normalized by norm kernel(s)
        else:
            Z = self.weights.sum()
            p = g / (Z + 1e-30)
        p = p + self._perturbation(pert_kernels, x_data)
        return p

    def log_constraint(self):
        return self.prior.log_prob(self.weights)

    def loglik(self, x_data=None, pert_kernels=None):
        """
        Log-likelihood.
        - With norm kernel(s): uses normalized density directly
        - Without norm kernel and no perturbation: uses raw g(x) (normalization
          cancels in the LR ratio)
        - Without norm kernel + perturbation: uses g(x)/Z + perturbation
        """
        # Always use the normalized density for the log-likelihood.
        # This ensures denominator and numerator are on the same scale.
        p = self.density(x_data, pert_kernels)
        ll = torch.log(p + 1e-10).sum()
        if self.train_weights and self.use_constraint:
            constraint = self.log_constraint()
            if torch.isnan(constraint) or torch.isinf(constraint):
                print(f"  WARNING: constraint is {constraint.item():.4e}, "
                      f"w range=[{self.weights.min().item():.4e}, {self.weights.max().item():.4e}], "
                      f"w sum={self.weights.sum().item():.4e}")
            ll = ll + constraint
        if torch.isnan(ll):
            print(f"  WARNING: loglik is NaN. "
                  f"w range=[{self.weights.min().item():.4e}, {self.weights.max().item():.4e}]")
        return ll

    def loss(self, x_data=None, pert_kernels=None):
        nll = -self.loglik(x_data, pert_kernels)
        if self.perturbation_net is not None and self.lambda_net > 0:
            b = self.perturbation_net.coefficients - self.perturbation_net.coefficients.mean()
            nll = nll + self.lambda_net * (b ** 2).sum()
        return nll


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_gof_model(model, x_data=None, lr=1e-4, epochs=50000, patience=1000,
                    verbose=True):
    """Train a GoFModel (numerator or denominator)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        if verbose:
            print("  No trainable parameters — skipping training.")
        return [], [], []

    opt = torch.optim.Adam(trainable, lr=lr)
    loss_hist = []
    weights_hist = []   # snapshots of the HIKER weights
    pert_hist = []      # snapshots of the perturbation coefficients (if any)

    # Precompute perturbation kernels if centres are fixed
    pert_kernels = None
    if (model.perturbation_net is not None
            and not model.perturbation_net.centers.requires_grad
            and x_data is not None):
        pert_kernels = model.perturbation_net.kernels(x_data).detach()

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = model.loss(x_data, pert_kernels)
        loss.backward()
        opt.step()
        if model.perturbation_net is not None:
            model.perturbation_net.clip_coefficients()
        if epoch % patience == 0:
            val = loss.item()
            loss_hist.append(val)
            with torch.no_grad():
                weights_hist.append(model.weights.detach().cpu().numpy().copy())
                if model.perturbation_net is not None:
                    pert_hist.append(
                        model.perturbation_net.coefficients.detach().cpu().numpy().copy())
            if verbose:
                print(f"  [{epoch:>6d}] loss = {val:.4f}")

    return loss_hist, weights_hist, pert_hist


# ──────────────────────────────────────────────────────────────────────
# Full GoF test
# ──────────────────────────────────────────────────────────────────────

def run_gof_test(K_list, K_norm, layer_sizes, x_data, weights_hat, cov_hat,
                 n_kernels_num=100, kernel_width_num=0.08, clip_num=0.1,
                 lambda_num=0.0, epochs_den=20000, epochs_num=100000,
                 lr_den=1e-4, lr_num=1e-4, patience=1000,
                 train_weights=True, use_constraint=True, verbose=True,
                 ensemble_info=None):
    """
    Run the full likelihood-ratio GoF test.

    Parameters
    ----------
    K_list : list of Tensor (N_test, M_l) — main kernels per layer
    K_norm : Tensor (N_test, 1) — single global normalization kernel
    layer_sizes : list of int — [M_0, M_1, ...]
    x_data : Tensor (N_test, d)
    weights_hat : Tensor (M_total,) — concatenated fitted weights
    cov_hat : Tensor (M_total, M_total)
    n_kernels_num : int — number of extra perturbation kernels
    kernel_width_num : float
    clip_num : float or None
    lambda_num : float
    epochs_den, epochs_num : int
    lr_den, lr_num : float
    patience : int
    train_weights : bool — if False, weights are frozen to w_hat
    use_constraint : bool — if False, no Gaussian penalty on weights
    verbose : bool

    Returns
    -------
    result : dict
    """
    device = x_data.device
    N_test = x_data.shape[0]

    # --- Denominator (null) ---
    if verbose:
        print("Training denominator (null)...")
    model_den = GoFModel(
        K_list, K_norm, layer_sizes, weights_hat, cov_hat,
        perturbation_net=None, train_weights=train_weights,
        use_constraint=use_constraint, ensemble_info=ensemble_info,
    ).to(device)
    loss_hist_den, w_hist_den, _ = train_gof_model(
        model_den, x_data, lr=lr_den, epochs=epochs_den,
        patience=patience, verbose=verbose)
    with torch.no_grad():
        ll_den = model_den.loglik(x_data).item()
    if verbose:
        print(f"  Denominator log-lik: {ll_den:.4f}")

    # --- Numerator (alternative) ---
    if verbose:
        print("Training numerator (alternative)...")
    idx = torch.randperm(N_test)[:n_kernels_num]
    centers_num = x_data[idx].clone()
    pert_net = PerturbationKernelLayer(centers_num, kernel_width_num, clip_coeffs=clip_num)

    model_num = GoFModel(
        K_list, K_norm, layer_sizes, weights_hat, cov_hat,
        perturbation_net=pert_net, lambda_net=lambda_num,
        train_weights=train_weights, use_constraint=use_constraint,
        ensemble_info=ensemble_info,
    ).to(device)
    loss_hist_num, w_hist_num, pert_hist_num = train_gof_model(
        model_num, x_data, lr=lr_num, epochs=epochs_num,
        patience=patience, verbose=verbose)
    with torch.no_grad():
        ll_num = model_num.loglik(x_data).item()
    if verbose:
        print(f"  Numerator log-lik: {ll_num:.4f}")

    # --- Test statistic ---
    t = 2.0 * (ll_num - ll_den)
    t = max(t, 0.0)
    dof = n_kernels_num
    p_value = 1.0 - chi2.cdf(t, dof)
    z_score = norm.ppf(1.0 - p_value) if p_value > 1e-15 else float("inf")

    if verbose:
        print(f"\n  Test statistic t = {t:.4f}")
        print(f"  Approx. DoF = {dof}")
        print(f"  p-value = {p_value:.4f}")
        print(f"  Z-score = {z_score:.2f}")

    return {
        "test_statistic": t,
        "numerator_ll": ll_num,
        "denominator_ll": ll_den,
        "p_value": p_value,
        "z_score": z_score,
        "dof": dof,
        "model_den": model_den,
        "model_num": model_num,
        "loss_hist_den": loss_hist_den,
        "loss_hist_num": loss_hist_num,
        "w_hist_den": w_hist_den,
        "w_hist_num": w_hist_num,
        "pert_hist_num": pert_hist_num,
    }
