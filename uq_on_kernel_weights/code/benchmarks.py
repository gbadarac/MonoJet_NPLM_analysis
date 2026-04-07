"""
Benchmark data generators.

Each benchmark is registered with a name and a generator function.
The generator signature is:
    generate(N, seed) -> ndarray (N, d)

To add a new benchmark:
    1. Write a generate_xxx function below
    2. Add it to the BENCHMARKS dict at the bottom

The function get_benchmark(name) returns the generator.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────────────────────────────

def generate_2d_gaussian(N, seed=42):
    """
    2D Gaussian:  N(mu, diag(sigma^2))
    mu = (-0.5, 0.6), sigma = (0.25, 0.4)
    """
    mu = np.array([-0.5, 0.6])
    sigma = np.array([0.25, 0.4])
    np.random.seed(seed)
    data = np.random.multivariate_normal(mu, np.diag(sigma ** 2), size=N)
    np.random.shuffle(data)
    return data.astype(np.float64)


def generate_2d_gmm_skew(N, seed=42):
    """
    2D distribution:
      x0: bimodal Gaussian mixture  (50/50, mu=-0.70/sig=0.12 and mu=-0.30/sig=0.12)
      x1: skew-normal               (loc=1.0, scale=0.75, alpha=8.0)
    From the original ENSEMBLEutils.generate_2GMMskew.
    """
    np.random.seed(seed)

    # Feature 0: bimodal Gaussian mixture
    wG = 0.50
    mu_a, sig_a = -0.70, 0.12
    mu_b, sig_b = -0.30, 0.12
    n_a = np.random.binomial(N, wG)
    x0 = np.concatenate([
        np.random.normal(mu_a, sig_a, n_a),
        np.random.normal(mu_b, sig_b, N - n_a),
    ])

    # Feature 1: skew-normal
    loc, scale, alpha = 1.0, 0.75, 8.0
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    z0 = np.random.randn(N)
    z1 = np.random.randn(N)
    x1 = loc + scale * (delta * np.abs(z0) + np.sqrt(1.0 - delta ** 2) * z1)

    data = np.column_stack([x0, x1])
    np.random.shuffle(data)
    return data.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────
# Analytic marginal PDFs (for plotting ratios to truth)
# ──────────────────────────────────────────────────────────────────────

def marginals_2d_gaussian(x_grids):
    """
    Return analytic marginal PDFs for the 2D Gaussian benchmark.

    Parameters
    ----------
    x_grids : list of 1-D arrays, one per dimension

    Returns
    -------
    marginals : list of 1-D arrays (same shapes as x_grids)
    """
    from scipy.stats import norm
    mu = np.array([-0.5, 0.6])
    sigma = np.array([0.25, 0.4])
    return [norm.pdf(x_grids[d], mu[d], sigma[d]) for d in range(2)]


def marginals_2d_gmm_skew(x_grids):
    """
    Return analytic marginal PDFs for the 2D GMM+skew benchmark.

    x0: 50/50 mixture of N(-0.70, 0.12^2) and N(-0.30, 0.12^2)
    x1: skew-normal(loc=1.0, scale=0.75, alpha=8.0)
    """
    from scipy.stats import norm, skewnorm

    # x0 marginal: bimodal Gaussian mixture
    mu_a, sig_a = -0.70, 0.12
    mu_b, sig_b = -0.30, 0.12
    marg0 = 0.5 * norm.pdf(x_grids[0], mu_a, sig_a) + 0.5 * norm.pdf(x_grids[0], mu_b, sig_b)

    # x1 marginal: skew-normal
    marg1 = skewnorm.pdf(x_grids[1], a=8.0, loc=1.0, scale=0.75)

    return [marg0, marg1]


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "2d_gaussian":   generate_2d_gaussian,
    "2d_gmm_skew":   generate_2d_gmm_skew,
}

MARGINALS = {
    "2d_gaussian":   marginals_2d_gaussian,
    "2d_gmm_skew":   marginals_2d_gmm_skew,
}


def get_benchmark(name):
    """
    Look up a benchmark generator by name.

    Parameters
    ----------
    name : str — one of the keys in BENCHMARKS

    Returns
    -------
    generate : callable(N, seed) -> ndarray (N, d)
    """
    if name not in BENCHMARKS:
        available = ", ".join(sorted(BENCHMARKS.keys()))
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARKS[name]


def get_marginals(name):
    """
    Look up the analytic marginal PDF function for a benchmark.

    Parameters
    ----------
    name : str

    Returns
    -------
    marginals_fn : callable(x_grids: list[ndarray]) -> list[ndarray], or None
    """
    return MARGINALS.get(name, None)


def list_benchmarks():
    """Return list of registered benchmark names."""
    return sorted(BENCHMARKS.keys())
