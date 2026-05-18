"""
Gaussian reference distribution q(x) for the wifi-ensemble pipeline.

Fits a full-covariance multivariate Gaussian to the training data via
method-of-moments (sample mean and sample covariance) and exposes a sampler.
The reference is fixed once at the start of the pipeline; everything
downstream (basis training, linear head fit, GoF) treats q as a deterministic
function.
"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal


def fit_gaussian_reference(X):
    """
    Fit q(x) = N(mu, Sigma) to data X via sample moments.

    Parameters
    ----------
    X : (N, d) array — training data.

    Returns
    -------
    mu    : (d,) torch.float64 tensor
    Sigma : (d, d) torch.float64 tensor (sample covariance, ddof=1)
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    X = X.double()
    mu = X.mean(dim=0)
    Xc = X - mu
    N = X.shape[0]
    Sigma = (Xc.T @ Xc) / (N - 1)
    return mu, Sigma


def sample_reference(mu, Sigma, N, seed=None):
    """
    Draw N i.i.d. samples from N(mu, Sigma).

    Parameters
    ----------
    mu    : (d,) tensor
    Sigma : (d, d) tensor
    N     : int
    seed  : int or None

    Returns
    -------
    (N, d) torch.float64 tensor
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    L = torch.linalg.cholesky(Sigma.double())
    d = mu.shape[0]
    z = torch.randn(N, d, generator=gen, dtype=torch.float64)
    return mu.double().unsqueeze(0) + z @ L.T


def log_prob(x, mu, Sigma):
    """log q(x) under N(mu, Sigma) for diagnostics."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    dist = MultivariateNormal(mu.double(), covariance_matrix=Sigma.double())
    return dist.log_prob(x.double())
