"""
basis.py — Bootstrapped MLP classifier ensemble for density ratio estimation.

ROLE IN THE PIPELINE
--------------------
This file implements Stage 1a of the density ratio estimation: training K
independent MLP classifiers, each of which approximates log(p(x)/q(x)) —
the log density ratio between the MC data distribution p and the Gaussian
reference q.

THE CLASSIFIER TRICK (why BCE training gives you log(p/q))
-----------------------------------------------------------
Given:
  - Y=1 events: actual MC data samples drawn from the unknown p(x)
  - Y=0 events: samples drawn from the known Gaussian q(x)

A binary classifier trained to distinguish these two populations by minimising
the Binary Cross Entropy (BCE) loss converges, in the infinite-data limit, to
the Bayes-optimal classifier:

    sigma(s*(x)) = p(x) / (p(x) + q(x))

where sigma is the sigmoid function. Rearranging:

    s*(x) = log(p(x) / q(x))   ← the log density ratio

So the optimal logit output of the classifier IS the log density ratio, without
ever estimating p or q individually. The classifier does not make a per-point
binary decision; it learns a continuous map across all of feature space by
comparing where the two populations are dense.

WHY AN ENSEMBLE OF K=128 CLASSIFIERS?
--------------------------------------
A single classifier trained on one dataset has:
  - High variance: different random initialisations give different estimates
  - Potential overfitting: the logit can memorise training-set noise

Training K=128 classifiers, each on a different bootstrap resample of the data
(resample with replacement), gives K slightly different estimates of log(p/q).
Their diversity encodes the finite-sample uncertainty in the density ratio.
The downstream wifi linear head (wifi_train.py) finds the best linear combination
of these K estimates, and the sandwich estimator gives Cov(ŵ) — the uncertainty
on those combination weights. This is what makes the GoF self-consistency check
well-calibrated. Empirically: K=64 gives a p-value distribution that is too
peaked at 0; K=96 is better; K=128 is the current best setting.

THE FEATURE VECTOR
------------------
After training, each MLP k is frozen and its logit output l_k(x) is used as a
basis function. The full feature vector for the linear head is:

    features(x) = (1, l_1(x), l_2(x), ..., l_K(x))   in R^{K+1}

The leading 1 is a bias column. The linear head then fits:

    log r̂(x) = w_0 * 1 + w_1 * l_1(x) + ... + w_K * l_K(x)
    r̂(x)     = exp(w · features(x))
    p̂(x)     = r̂(x) * q(x) / Z
"""

import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

from reference import sample_reference


# ──────────────────────────────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────────────────────────────

class MLPLogit(nn.Module):
    """
    Small MLP that maps an input event x to a single scalar logit s(x).

    Architecture: d_in → [hidden layers with ReLU] → 1 (no activation)

    The output s(x) is an unbounded real number. After BCE training against
    MC data (Y=1) vs Gaussian reference (Y=0), s(x) approximates log(p(x)/q(x)):
      - s(x) >> 0: region where p >> q (data much denser than reference)
      - s(x) ≈ 0 : region where p ≈ q (data and reference equally dense)
      - s(x) << 0: region where p << q (reference much denser than data)

    float64 throughout for numerical precision in the downstream linear head fit.
    """

    def __init__(self, d_in, hidden=(32, 32, 16)):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            # Linear projection followed by ReLU non-linearity
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # Final linear layer — no activation, output is an unbounded logit
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers).double()

    def forward(self, x):
        # squeeze(-1): remove the trailing dimension-1 so output is (N,) not (N,1)
        return self.net(x.double()).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _balanced_bce_sum(logit, y, sample_weights):
    """
    Weighted BCE loss (sum reduction) with class-balancing weights.

    WHY CLASS-BALANCED WEIGHTS?
    Without weighting, the BCE loss is dominated by whichever class has more
    samples. If N_ref >> N_data (e.g. with ref_oversample > 1), the loss would
    push s(x) → -infinity for all x to satisfy the many Y=0 examples, biasing
    the logit away from log(p/q). The weight_decay penalty would then pull the
    bias component back toward 0, further distorting the shape of s*(x).

    With per-sample weights 1/(2*N_d) for Y=1 and 1/(2*N_r) for Y=0, the
    weighted sum equals 0.5*(mean_d BCE + mean_r BCE), whose population minimiser
    is exactly s*(x) = log(p(x)/q(x)) — independent of the data:ref ratio.
    This means ref_oversample can be set freely without biasing the learned ratio.
    """
    return nn.functional.binary_cross_entropy_with_logits(
        logit, y, weight=sample_weights, reduction="sum",
    )


def _auc(s_data_np, s_ref_np):
    """
    Mann-Whitney U AUC: probability that a randomly chosen data event scores
    higher than a randomly chosen reference event.

    AUC = 1.0 : perfect separation (classifier always assigns higher logit to data)
    AUC = 0.5 : chance level (classifier cannot distinguish data from reference)
    AUC < 0.5 : classifier is inverted (should not happen with correct training)

    A val AUC well above 0.5 confirms the classifier is learning real structure
    in log(p/q), not just memorising noise.
    """
    n_d = len(s_data_np)
    n_r = len(s_ref_np)
    if n_d == 0 or n_r == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([s_data_np, s_ref_np]))
    rank_data_sum = ranks[:n_d].sum()
    return float((rank_data_sum - n_d * (n_d + 1) / 2.0) / (n_d * n_r))


@torch.no_grad()
def _eval_bce_and_auc(model, X_data, X_ref, device):
    """
    Evaluate class-balanced BCE and AUC for a (possibly frozen) model.

    BCE is reported as 0.5*(mean_d BCE + mean_r BCE) so that chance level
    (a random, uninformative classifier) always gives log(2) ≈ 0.693,
    regardless of the data:ref ratio. This makes diagnostics comparable
    across runs with different ref_oversample settings.

    HEALTHY DIAGNOSTIC VALUES:
      - val BCE well below log(2) ≈ 0.693: classifier is learning
      - val BCE ≈ log(2): classifier at chance — basis member is useless
      - val BCE ≈ train BCE: no overfitting (good generalisation)
      - val BCE >> train BCE: overfitting — consider more regularisation
      - val AUC > 0.6: meaningful separation between p and q
    """
    s_d = model(X_data.to(device))
    s_r = model(X_ref.to(device))
    bce_d_mean = nn.functional.binary_cross_entropy_with_logits(
        s_d, torch.ones_like(s_d), reduction="mean",
    )
    bce_r_mean = nn.functional.binary_cross_entropy_with_logits(
        s_r, torch.zeros_like(s_r), reduction="mean",
    )
    # Class-balanced BCE: 0.5*(mean over data + mean over ref)
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

    After training, the model's logit output l(x) approximates log(p(x)/q(x)).
    The model is returned in eval mode with all parameters frozen — it is never
    updated again after this function returns.

    TRAINING OBJECTIVE
    The class-balanced BCE loss is minimised:

        L(w) = 0.5 * (mean over data of -log sigma(s(x))
                    + mean over ref  of -log(1 - sigma(s(x))))

    In the infinite-data, infinite-capacity limit, the minimiser satisfies:
        s*(x) = log(p(x)/q(x))

    With finite data and a small MLP, s(x) is an approximation to this.
    Each bootstrap member sees a different data resample (from train_bootstrap_basis),
    so the K members give K different approximations, capturing the uncertainty
    in the density ratio estimate.

    LR SCHEDULE
    Cosine annealing decays the learning rate from lr down to lr/100 over the
    full training run. This improves convergence: the large initial lr allows
    fast progress toward the minimum; the small final lr refines the solution
    without oscillating around it.

    Parameters
    ----------
    X_data, X_ref : training tensors (Y=1 / Y=0).
    hidden        : MLP layer widths, e.g. (32, 32, 16).
    epochs, lr, weight_decay, batch_size : Adam hyperparameters.
    lr_schedule   : "constant" or "cosine" (cosine → eta_min = lr/100).
    X_val_data, X_val_ref : optional held-out tensors for val BCE/AUC diagnostics.
                    These are the 10% val pool carved from half_A in run.py.
                    They are NEVER used for gradient updates — only for monitoring.

    Returns
    -------
    model : trained MLPLogit (eval mode, all params frozen).
    diag  : dict of training diagnostics:
              train/val BCE at start and end of training,
              val AUC at start and end,
              final learning rate.
    """
    torch.manual_seed(int(seed))
    d_in = X_data.shape[1]
    model = MLPLogit(d_in, hidden=hidden).to(device)

    X_data_dev = X_data.double().to(device)
    X_ref_dev  = X_ref.double().to(device)
    # Concatenate data (Y=1) and reference (Y=0) into a single training set
    X = torch.cat([X_data_dev, X_ref_dev], dim=0)
    N_d = X_data_dev.shape[0]
    N_r = X_ref_dev.shape[0]
    y = torch.cat([
        torch.ones(N_d,  dtype=torch.float64, device=device),   # data → label 1
        torch.zeros(N_r, dtype=torch.float64, device=device),   # ref  → label 0
    ])

    # Class-balancing weights: 1/(2*N_d) for each data event, 1/(2*N_r) for each
    # reference event. These ensure the loss is 0.5*(mean_d + mean_r) regardless
    # of the data:ref ratio, so the optimal logit is log(p/q) unconditionally.
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
        # Cosine annealing: lr decays smoothly from lr to lr/100 over T_max epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr / 100.0
        )
    elif lr_schedule == "constant":
        scheduler = None
    else:
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")

    # Record BCE/AUC before any training (epoch 0 diagnostics)
    train_bce_first, _ = _eval_bce_and_auc(model, X_data_dev, X_ref_dev, device)
    if has_val:
        val_bce_first, val_auc_first = _eval_bce_and_auc(
            model, X_val_data, X_val_ref, device,
        )
    else:
        val_bce_first = float("nan")
        val_auc_first = float("nan")

    if batch_size is None or batch_size >= N:
        # Full-batch training: use all events each step
        for ep in range(epochs):
            opt.zero_grad()
            loss = _balanced_bce_sum(model(X), y, sample_weights)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if verbose and (ep % max(1, epochs // 10) == 0):
                lr_now = opt.param_groups[0]["lr"]
                print(f"      epoch {ep:5d}  loss={loss.item():.5f}  lr={lr_now:.2e}")
    else:
        # Mini-batch training: shuffle and iterate over batches each epoch.
        # Each minibatch gradient is an unbiased estimator of the full-batch
        # gradient (scaled by N/|mb| to keep gradient magnitude consistent).
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        for ep in range(epochs):
            perm = torch.randperm(N, generator=gen)
            tot, n_seen = 0.0, 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size].to(device)
                opt.zero_grad()
                bce = _balanced_bce_sum(
                    model(X[idx]), y[idx], sample_weights[idx],
                )
                # Scale by N/|mb| so the gradient magnitude matches full-batch
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

    # Freeze the model: switch to eval mode and disable all gradients.
    # From this point on, the model is a fixed function x → l(x) ≈ log(p/q).
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Record final BCE/AUC after training
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
    Train K independent MLP classifiers to form the basis for log r̂(x).

    BOOTSTRAP RESAMPLING
    Each of the K members trains on a fresh bootstrap resample of X_train_pool:
    sample N_pool indices with replacement from {0, ..., N_pool-1}. Different
    members see different random subsets (with repetition), so they converge to
    slightly different approximations of log(p/q). This diversity is what the
    wifi linear head exploits: it finds the combination ŵ that best fits the
    held-out half_B data, and Cov(ŵ) captures how sensitive that combination
    is to which bootstrap was drawn — i.e. the finite-sample uncertainty in
    the density ratio estimate.

    INDEPENDENT SEEDS
    Each member k uses a distinct seed for both its data bootstrap and its
    reference draw, ensuring all K members are statistically independent.
    The validation reference draw uses a single fixed seed (shared across
    members) so that val BCE/AUC numbers are directly comparable.

    Parameters
    ----------
    X_train_pool : (N_train_pool, d) — bootstrap source for Y=1 events.
                   This is the 90% training portion of half_A (after carving
                   out the 10% val pool in run.py).
    X_val_pool   : (N_val, d) — fixed held-out Y=1 events for val diagnostics.
                   Never used for gradient updates.
    mu_q, Sigma_q : Gaussian reference moments from fit_gaussian_reference.
    K             : number of classifiers. Currently K=128 (empirically best).
    ref_oversample: int >= 1. Each member trains against ref_oversample * N_pool
                   reference events. The class-balancing weights in
                   _balanced_bce_sum ensure this does not bias the logit.

    Returns
    -------
    models : list of K frozen MLPLogit instances (eval mode, on CPU).
    diags  : list of K per-member diagnostic dicts (train/val BCE/AUC).
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
    # Single generator for all bootstrap index draws — deterministic given seed
    boot_gen = torch.Generator().manual_seed(int(seed))

    # Draw a fixed validation reference once, shared across all K members.
    # Size matches val_pool × ref_oversample to mirror the train-side class ratio,
    # keeping val BCE directly comparable to train BCE.
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
        # Bootstrap resample: draw N_pool indices with replacement
        idx = torch.randint(0, N_pool, (N_pool,), generator=boot_gen)
        Xb = X_train_pool[idx]  # bootstrap data sample for member k

        # Fresh reference draw for this member — distinct seed per member
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
            seed=int(seed) + 100 + k,       # distinct seed per member
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

        # Move to CPU after training to free GPU memory before next member
        models.append(model.cpu())
        diags.append(diag)
    return models, diags


# ──────────────────────────────────────────────────────────────────────
# Feature evaluation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_basis(models, X, device="cpu"):
    """
    Evaluate all K frozen classifiers on X and stack their logits.

    Returns an (N, K) tensor where entry [i, k] = l_k(x_i) ≈ log(p(x_i)/q(x_i))
    as estimated by the k-th bootstrap member.
    """
    if isinstance(X, torch.Tensor):
        Xt = X.double().to(device)
    else:
        Xt = torch.from_numpy(X).double().to(device)
    cols = [m.to(device)(Xt) for m in models]
    return torch.stack(cols, dim=1)  # shape (N, K)


@torch.no_grad()
def evaluate_features(models, X, device="cpu"):
    """
    Build the full (N, K+1) feature matrix used by the wifi linear head.

    The feature vector for event x is:
        features(x) = (1, l_1(x), l_2(x), ..., l_K(x))

    The leading column of ones is the bias term. The linear head then fits:
        log r̂(x) = w_0 * 1 + w_1 * l_1(x) + ... + w_K * l_K(x)

    The bias w_0 absorbs any constant offset in the ratio (including the
    normalisation constant Z and any constant shift from ref_oversample).
    """
    L = evaluate_basis(models, X, device=device)           # (N, K)
    ones = torch.ones(L.shape[0], 1, dtype=torch.float64, device=L.device)
    return torch.cat([ones, L], dim=1)                     # (N, K+1)
