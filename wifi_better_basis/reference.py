"""
reference.py — Gaussian reference distribution q(x).

ROLE IN THE PIPELINE
--------------------
The overall goal is to augment a limited MC simulation dataset by learning its
distribution p(x) well enough to generate more synthetic events. Rather than
estimating p(x) directly (as normalizing flows or kernel mixtures do), this
pipeline estimates the DENSITY RATIO r(x) = p(x)/q(x), where q(x) is a simple
known reference distribution. The target density is then recovered as:

    p̂(x) = r̂(x) · q(x) / Z

where Z ≈ 1 in the population limit and is handled implicitly by SIR reweighting.

This file provides q(x): a multivariate Gaussian fitted to the training data.
q is fixed once at the start and treated as a deterministic, known function
by everything downstream (basis training, linear head fit, GoF test).

WHY A GAUSSIAN?
---------------
q must satisfy three requirements:
  1. Have support wherever the data does (so the ratio r = p/q is finite everywhere)
  2. Be cheap to sample from (we draw millions of reference events throughout the pipeline)
  3. Be different enough from p that r(x) = p(x)/q(x) has real structure for
     the classifiers to learn

A Gaussian fitted to the data satisfies all three. Crucially, q must NOT be
equal to p: if q = p, then r(x) = 1 everywhere, the classifiers have nothing
to learn, and the entire basis collapses. The Gaussian is deliberately simpler
than p — it captures the rough location and spread of the data but misses
non-Gaussian features (multi-modality, skewness, tail structure). These
deviations are exactly what the MLP classifiers in basis.py learn to capture
via the ratio r(x).
"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal


def fit_gaussian_reference(X):
    """
    Fit q(x) = N(mu, Sigma) to the training data via sample moments.

    This is the maximum-likelihood estimator for a Gaussian family.
    The result (mu, Sigma) fully specifies q and is used throughout the
    pipeline to: (a) generate Y=0 reference events for classifier training,
    (b) draw SIR pools for coverage and GoF tests.

    WHY THE FULL TRAINING SET?
    q is fit on ALL of X_train (before the 50/50 split into half_A and half_B).
    This is intentional: we want q to be as representative as possible.
    Using the full set does not cause data leakage because q is treated as a
    fixed, deterministic function — it is not a fitted model that enters the
    likelihood ratio or the sandwich covariance.

    Parameters
    ----------
    X : (N, d) array-like — training MC events.
        N = number of events, d = number of observables (e.g. 4 for MonoJet).

    Returns
    -------
    mu    : (d,) torch.float64 — sample mean, i.e. the centroid of q
    Sigma : (d, d) torch.float64 — sample covariance (ddof=1), i.e. the
            spread and correlations of q. Off-diagonal entries capture linear
            correlations between observables.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.double()  # float64 throughout for numerical precision

    # Sample mean: mu = (1/N) sum_i x_i  — the centroid of the Gaussian
    mu = X.mean(dim=0)

    # Centre the data: Xc[i] = x_i - mu
    Xc = X - mu

    N = X.shape[0]
    # Sample covariance with Bessel's correction (divide by N-1, not N)
    # giving an unbiased estimate of the population covariance.
    # Shape: (d, d). This defines the ellipsoidal shape of q.
    Sigma = (Xc.T @ Xc) / (N - 1)
    return mu, Sigma


def sample_reference(mu, Sigma, N, seed=None):
    """
    Draw N i.i.d. samples from the fitted reference q(x) = N(mu, Sigma).

    This is called at many points in the pipeline, always with a distinct
    seed so all reference draws are statistically independent:
      - basis.py     : Y=0 training events for each MLP classifier
      - wifi_train.py: Y=0 events for the linear head fit and bootstrap Cov(ŵ)
      - run_gof.py   : Y=0 events for the GoF test and toy calibration
      - coverage.py  : SIR pool for evaluating <x_0> under p̂

    WHY CHOLESKY?
    If z ~ N(0, I_d), then mu + z @ L^T ~ N(mu, Sigma), where L is the lower
    Cholesky factor of Sigma (Sigma = L L^T). This avoids constructing the
    full MultivariateNormal object at every call and is numerically stable
    even when Sigma is close to singular.

    Parameters
    ----------
    mu    : (d,) tensor  — reference mean (from fit_gaussian_reference)
    Sigma : (d, d) tensor — reference covariance (from fit_gaussian_reference)
    N     : int — number of samples to draw
    seed  : int or None — RNG seed. Each call site uses a distinct offset
            so that basis training draws, linear head draws, GoF draws, and
            SIR pool draws are all mutually independent.

    Returns
    -------
    X_ref : (N, d) torch.float64 tensor — i.i.d. samples from N(mu, Sigma)
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    # Cholesky decomposition: Sigma = L L^T (L is lower triangular)
    L = torch.linalg.cholesky(Sigma.double())

    d = mu.shape[0]
    # Standard normal noise: z[i] ~ N(0, I_d)
    z = torch.randn(N, d, generator=gen, dtype=torch.float64)

    # Affine transform to N(mu, Sigma): x = mu + z L^T
    # (z is a row vector, so we right-multiply by L^T)
    return mu.double().unsqueeze(0) + z @ L.T


def log_prob(x, mu, Sigma):
    """
    Evaluate log q(x) = log N(x; mu, Sigma) for diagnostics only.

    Not used in the main fitting pipeline. The classifiers in basis.py implicitly
    use the ratio p(x)/q(x) via their BCE training objective without ever
    needing to evaluate log q(x) explicitly.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    dist = MultivariateNormal(mu.double(), covariance_matrix=Sigma.double())
    return dist.log_prob(x.double())
