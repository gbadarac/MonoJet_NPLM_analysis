"""
Bootstrapped MLP basis for wifi_better_basis.

Same scheme as code/wifi/basis.py — train K small MLP classifiers, each on
its own bootstrap of the data plus a fresh draw from the Gaussian reference
q, then freeze and use each member's final logit as a basis function. The
full feature vector for the downstream linear head is

    features(x) = (1, l_1(x), …, l_K(x))   in R^{K+1}.

Diff vs the original:
  - Cosine LR decay (with `BASIS_LR_SCHEDULE = "cosine"`) on top of Adam,
    eta_min = lr / 100. "constant" preserves the original behaviour.
  - train_bootstrap_basis takes (X_train_pool, X_val_pool) and bootstraps
    only from the train pool. Each member is also evaluated on the val pool
    with a fresh-q ref draw of matching size; per-member train/val BCE and
    val AUC are returned alongside the model so run.py can surface
    undertraining vs capacity issues.
  - ref_oversample inflates the y=0 reference pool to oversample × |bootstrap|.
    The constant logit shift this introduces (log(N_d/N_r)) is absorbed by
    the bias column in the head, so this is a free quality bump for the BCE
    fits.

MLPLogit / evaluate_basis / evaluate_features are byte-for-byte the same as
the wifi versions (so saved basis state dicts are interchangeable).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

from reference import sample_reference


# ──────────────────────────────────────────────────────────────────────
# Model — same as wifi/basis.MLPLogit
# ──────────────────────────────────────────────────────────────────────

class MLPLogit(nn.Module):
    """Small MLP -> single scalar logit. float64 throughout."""

    def __init__(self, d_in, hidden=(32, 32, 16)):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers).double()

    def forward(self, x):
        return self.net(x.double()).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _balanced_bce_sum(logit, y, sample_weights):
    """Weighted BCE-with-logits, sum reduction. Caller passes per-sample
    weights (typically 1/(2·N_d) for Y=1 and 1/(2·N_r) for Y=0) so the
    population minimizer is the unscaled log density ratio log r(x), no
    matter what the data:ref ratio is. Without this weighting, oversampled
    refs would give s*(x) = log r(x) − log(N_r / N_d), and weight_decay
    would penalize the constant offset back toward 0, biasing the basis
    members' shape. Weights baked in here cancel both effects."""
    return nn.functional.binary_cross_entropy_with_logits(
        logit, y, weight=sample_weights, reduction="sum",
    )


def _auc(s_data_np, s_ref_np):
    """Mann-Whitney U based AUC. Larger means data scores rank above ref."""
    n_d = len(s_data_np)
    n_r = len(s_ref_np)
    if n_d == 0 or n_r == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([s_data_np, s_ref_np]))
    rank_data_sum = ranks[:n_d].sum()
    return float((rank_data_sum - n_d * (n_d + 1) / 2.0) / (n_d * n_r))


@torch.no_grad()
def _eval_bce_and_auc(model, X_data, X_ref, device):
    """Class-balanced BCE and AUC for a frozen model on (X_data, X_ref).

    BCE is reported as 0.5·(mean_d BCE + mean_r BCE) so chance-level (random
    predictor) is log(2) ≈ 0.693 regardless of the data:ref ratio. This makes
    val numbers across different BASIS_REF_OVERSAMPLE settings directly
    comparable, and it matches the population-loss objective optimised in
    train_basis_member."""
    s_d = model(X_data.to(device))
    s_r = model(X_ref.to(device))
    bce_d_mean = nn.functional.binary_cross_entropy_with_logits(
        s_d, torch.ones_like(s_d), reduction="mean",
    )
    bce_r_mean = nn.functional.binary_cross_entropy_with_logits(
        s_r, torch.zeros_like(s_r), reduction="mean",
    )
    bce_balanced = 0.5 * (bce_d_mean + bce_r_mean)
    auc = _auc(s_d.cpu().numpy(), s_r.cpu().numpy())
    return float(bce_balanced.item()), auc


# ──────────────────────────────────────────────────────────────────────
# Training a single basis member
# ──────────────────────────────────────────────────────────────────────

def train_basis_member(X_data, X_ref, hidden, epochs, lr, weight_decay,
                       batch_size=None, lr_schedule="constant",
                       device="cpu", seed=0,
                       X_val_data=None, X_val_ref=None,
                       verbose=False):
    """
    Train one MLP classifier to discriminate X_data (Y=1) from X_ref (Y=0).

    Parameters
    ----------
    X_data, X_ref : training tensors (Y=1 / Y=0).
    hidden        : MLP layer widths.
    epochs, lr, weight_decay, batch_size : Adam hyperparameters.
    lr_schedule   : "constant" (no schedule) or "cosine"
                    (CosineAnnealingLR, eta_min = lr / 100).
    X_val_data, X_val_ref : optional held-out tensors. If both provided,
                    val BCE/AUC are evaluated at start and end and stored
                    in the diag dict.

    Returns
    -------
    model : trained MLPLogit (eval mode, frozen, on `device`).
    diag  : dict with at least
              {"train_bce_first", "train_bce_final",
               "val_bce_first",   "val_bce_final",
               "val_auc_first",   "val_auc_final",
               "lr_final"}.
            val_* are NaN when no validation tensors are provided.
    """
    torch.manual_seed(int(seed))
    d_in = X_data.shape[1]
    model = MLPLogit(d_in, hidden=hidden).to(device)

    X_data_dev = X_data.double().to(device)
    X_ref_dev  = X_ref.double().to(device)
    X = torch.cat([X_data_dev, X_ref_dev], dim=0)
    N_d = X_data_dev.shape[0]
    N_r = X_ref_dev.shape[0]
    y = torch.cat([
        torch.ones(N_d,  dtype=torch.float64, device=device),
        torch.zeros(N_r, dtype=torch.float64, device=device),
    ])
    # Per-sample weights so the BCE is class-balanced. With these, sum_i
    # w_i · BCE_i = ½·(mean_d BCE + mean_r BCE), whose population minimizer
    # is s*(x) = log r(x) — independent of the data:ref ratio. See the
    # docstring on _balanced_bce_sum for why this matters with weight_decay.
    sample_weights = torch.cat([
        torch.full((N_d,), 1.0 / (2.0 * N_d), dtype=torch.float64, device=device),
        torch.full((N_r,), 1.0 / (2.0 * N_r), dtype=torch.float64, device=device),
    ])
    N = X.shape[0]

    has_val = (X_val_data is not None) and (X_val_ref is not None)
    if has_val:
        X_val_data = X_val_data.double()
        X_val_ref = X_val_ref.double()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr / 100.0
        )
    elif lr_schedule == "constant":
        scheduler = None
    else:
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")

    train_bce_first, _ = _eval_bce_and_auc(model, X_data_dev, X_ref_dev, device)
    if has_val:
        val_bce_first, val_auc_first = _eval_bce_and_auc(
            model, X_val_data, X_val_ref, device,
        )
    else:
        val_bce_first = float("nan")
        val_auc_first = float("nan")

    if batch_size is None or batch_size >= N:
        for ep in range(epochs):
            opt.zero_grad()
            # Full-batch balanced BCE: sum_i w_i · BCE_i = ½·(mean_d + mean_r).
            loss = _balanced_bce_sum(model(X), y, sample_weights)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if verbose and (ep % max(1, epochs // 10) == 0):
                lr_now = opt.param_groups[0]["lr"]
                print(f"      epoch {ep:5d}  loss={loss.item():.5f}  lr={lr_now:.2e}")
    else:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        for ep in range(epochs):
            perm = torch.randperm(N, generator=gen)
            tot, n_seen = 0.0, 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size].to(device)
                opt.zero_grad()
                # Minibatch is a uniform random subset; sum_{i in mb} w_i · BCE_i
                # is an unbiased estimator of the full-batch sum (= balanced
                # BCE) up to a |mb|/N factor. Scale by N/|mb| so the gradient
                # magnitude per step matches the full-batch case.
                bce = _balanced_bce_sum(
                    model(X[idx]), y[idx], sample_weights[idx],
                )
                loss = bce * (N / float(idx.numel()))
                loss.backward()
                opt.step()
                tot += loss.item() * idx.numel()
                n_seen += idx.numel()
            if scheduler is not None:
                scheduler.step()
            if verbose and (ep % max(1, epochs // 10) == 0):
                lr_now = opt.param_groups[0]["lr"]
                print(f"      epoch {ep:5d}  loss={tot / n_seen:.5f}  lr={lr_now:.2e}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    train_bce_final, _ = _eval_bce_and_auc(model, X_data_dev, X_ref_dev, device)
    if has_val:
        val_bce_final, val_auc_final = _eval_bce_and_auc(
            model, X_val_data, X_val_ref, device,
        )
    else:
        val_bce_final = float("nan")
        val_auc_final = float("nan")

    diag = {
        "train_bce_first": float(train_bce_first),
        "train_bce_final": float(train_bce_final),
        "val_bce_first":   float(val_bce_first),
        "val_bce_final":   float(val_bce_final),
        "val_auc_first":   float(val_auc_first),
        "val_auc_final":   float(val_auc_final),
        "lr_final":        float(opt.param_groups[0]["lr"]),
    }
    return model, diag


# ──────────────────────────────────────────────────────────────────────
# Bootstrap ensemble of basis members
# ──────────────────────────────────────────────────────────────────────

def train_bootstrap_basis(X_train_pool, X_val_pool, mu_q, Sigma_q,
                          K, hidden, epochs, lr, weight_decay,
                          batch_size=None, lr_schedule="constant",
                          ref_oversample=1, device="cpu", seed=0,
                          verbose=False):
    """
    Train K MLP classifiers, each on a fresh bootstrap of X_train_pool plus
    an i.i.d. sample from N(mu_q, Sigma_q) of size ref_oversample × |bootstrap|.
    Each member is also evaluated on a fixed val pool (X_val_pool + matched
    fresh q-draw) and per-member train/val metrics are returned.

    Parameters
    ----------
    X_train_pool : (N_train_pool, d) — bootstrap source for Y=1.
    X_val_pool   : (N_val, d) or None — fixed held-out Y=1 for val metrics.
    mu_q, Sigma_q : Gaussian reference moments (Y=0 source).
    K, hidden, epochs, lr, weight_decay, batch_size, lr_schedule : per-member
        Adam hyperparameters and architecture.
    ref_oversample : int >= 1.
    device : passed to each member's training loop and to the frozen model.

    Returns
    -------
    models : list[MLPLogit] of length K (frozen, eval mode).
    diags  : list[dict] of length K (per-member train/val BCE/AUC).
    """
    if isinstance(X_train_pool, torch.Tensor):
        X_train_pool = X_train_pool.double()
    else:
        X_train_pool = torch.from_numpy(X_train_pool).double()
    if X_val_pool is not None:
        if isinstance(X_val_pool, torch.Tensor):
            X_val_pool = X_val_pool.double()
        else:
            X_val_pool = torch.from_numpy(X_val_pool).double()

    N_pool = X_train_pool.shape[0]
    boot_gen = torch.Generator().manual_seed(int(seed))

    # Fixed validation reference draw (same across members so val numbers are
    # directly comparable). Size matches val_pool × ref_oversample to mirror
    # the train side's class ratio.
    if X_val_pool is not None and X_val_pool.shape[0] > 0:
        N_val_ref = max(1, ref_oversample * X_val_pool.shape[0])
        X_val_ref = sample_reference(mu_q, Sigma_q, N_val_ref,
                                     seed=int(seed) + 77_777)
    else:
        X_val_pool = None
        X_val_ref = None

    models = []
    diags = []
    for k in range(K):
        idx = torch.randint(0, N_pool, (N_pool,), generator=boot_gen)
        Xb = X_train_pool[idx]
        N_ref_k = max(1, ref_oversample * Xb.shape[0])
        Xr = sample_reference(mu_q, Sigma_q, N_ref_k,
                              seed=int(seed) + 10_000 + k)
        if verbose:
            print(f"  basis {k+1}/{K}: bootstrap |D|={Xb.shape[0]}  "
                  f"|R|={Xr.shape[0]}  (oversample={ref_oversample}x)")
        model, diag = train_basis_member(
            Xb, Xr, hidden=hidden, epochs=epochs, lr=lr,
            weight_decay=weight_decay, batch_size=batch_size,
            lr_schedule=lr_schedule, device=device,
            seed=int(seed) + 100 + k,
            X_val_data=X_val_pool, X_val_ref=X_val_ref,
            verbose=verbose,
        )
        if verbose:
            v_bce = diag["val_bce_final"]
            t_bce = diag["train_bce_final"]
            v_auc = diag["val_auc_final"]
            v_str = (f"  val BCE={v_bce:.4f}  AUC={v_auc:.3f}"
                     if not math.isnan(v_bce) else "  (no val)")
            print(f"    [{k+1}/{K} done]  train BCE={t_bce:.4f}{v_str}")
        models.append(model.cpu())
        diags.append(diag)
    return models, diags


# ──────────────────────────────────────────────────────────────────────
# Feature evaluation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_basis(models, X, device="cpu"):
    """Stack final logits from each basis member into (N, K) tensor."""
    if isinstance(X, torch.Tensor):
        Xt = X.double().to(device)
    else:
        Xt = torch.from_numpy(X).double().to(device)
    cols = [m.to(device)(Xt) for m in models]
    return torch.stack(cols, dim=1)


@torch.no_grad()
def evaluate_features(models, X, device="cpu"):
    """Return the full (N, K+1) feature matrix including the leading bias column."""
    L = evaluate_basis(models, X, device=device)
    ones = torch.ones(L.shape[0], 1, dtype=torch.float64, device=L.device)
    return torch.cat([ones, L], dim=1)
