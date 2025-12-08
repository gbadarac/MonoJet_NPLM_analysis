from scipy.stats import multivariate_normal
from sklearn.datasets import make_moons
import numpy as np

def pdf_from_ensemble(x, centroids, coefficients, widths, weights):
    '''
    centroids (n_components, n_kernels, d)
    widths (n_components, n_kernels, d)
    coefficients (n_components, n_kernels)
    weights (n_components)
    '''
    n_components = centroids.shape[0]
    n_kernels = centroids.shape[1]
    out = np.zeros((x.shape[0],))
    for i in range(n_components):
        for j in range(n_kernels):
            out+=weights[i]*coefficients[i,j]*multivariate_normal.pdf(x, mean=centroids[i,j], cov=widths[i, j]**2)
    return out

def pdf_components_from_ensemble(x, centroids, coefficients, widths):
    '''
    centroids (n_components, n_kernels, d)
    widths (n_components, n_kernels, d)
    coefficients (n_components, n_kernels)
    weights (n_components)
    '''
    n_components = centroids.shape[0]
    n_kernels = centroids.shape[1]
    out = [np.zeros((x.shape[0],)) for _ in range(n_components)]
    for i in range(n_components):
        for j in range(n_kernels):
            out[i]+=coefficients[i,j]*multivariate_normal.pdf(x, mean=centroids[i,j], cov=widths[i, j]**2)
    return out



def hit_or_miss_sample(seed, N, centroids, coefficients, widths, weights, bounds, pdf_from_ensemble, max_trials=1_000_000):
    """
    Sample N points using hit-or-miss (rejection sampling) from a possibly signed Gaussian ensemble.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    centroids, coefficients, widths, weights : arrays
        As in pdf_from_ensemble.
    bounds : list of tuples
        [(xmin, xmax), (ymin, ymax), ...] defining the sampling region.
    pdf_from_ensemble : callable
        Function pdf_from_ensemble(x, centroids, coefficients, widths, weights)
        returning density values for array x of shape (n_samples, d).
    max_trials : int
        Safety limit on number of proposal draws.

    Returns
    -------
    samples : ndarray, shape (N, d)
        Accepted sample points.
    signs : ndarray, shape (N,)
        Sign of pdf value at each accepted point (useful if pdf can be negative).
    """
    np.random.seed(seed)
    d = centroids.shape[-1]
    bounds = np.array(bounds)
    low, high = bounds[:, 0], bounds[:, 1]

    # Step 1. Estimate |pdf| max by probing random points
    probe_pts = np.random.uniform(low, high, size=(1000, d))
    vals = pdf_from_ensemble(probe_pts, centroids, coefficients, widths, weights)
    f_max = np.max(np.abs(vals)) * 2  # safety margin

    samples = []
    signs = []

    n_accept = 0
    n_trials = 0

    # Step 2. Hit-or-miss loop
    while n_accept < N and n_trials < max_trials:
        # Uniform random candidate point
        x = np.random.uniform(low, high)
        fx = pdf_from_ensemble(x[None, :], centroids, coefficients, widths, weights)[0]
        # Acceptance criterion
        if (np.random.rand() < fx / f_max) and (fx>0):
            samples.append(x)
            signs.append(np.sign(fx))
            n_accept += 1
        n_trials += 1

    if n_accept < N:
        print(f"Warning: only accepted {n_accept} / {N} points after {n_trials} trials")

    return np.array(samples), np.array(signs)


def generate_2GMMskew(N_train_tot=100_000, seed=1):
    np.random.seed(seed)

    # ---- Feature 1: Bi-modal Gaussian mixture ----
    wG = 0.50
    mu_a, sig_a = -0.70, 0.12
    mu_b, sig_b = -0.30, 0.12
    n1_a = np.random.binomial(N_train_tot, wG)
    x1 = np.concatenate([
        np.random.normal(mu_a, sig_a, n1_a),
        np.random.normal(mu_b, sig_b, N_train_tot - n1_a),
    ])
    E_x1 = wG * mu_a + (1.0 - wG) * mu_b  # analytic mean

    # ---- Feature 2: Skew-Normal ----
    def sample_skewnorm(n, loc, scale, alpha):
        alpha = float(alpha)
        delta = alpha / np.sqrt(1.0 + alpha**2)
        z0 = np.random.randn(n); z1 = np.random.randn(n)
        xstd = delta * np.abs(z0) + np.sqrt(1.0 - delta**2) * z1
        return loc + scale * xstd

    loc2, scale2, alpha2 = 1.0, 0.75, 8.0
    x2 = sample_skewnorm(N_train_tot, loc2, scale2, alpha2)

    delta = alpha2 / np.sqrt(1.0 + alpha2**2)
    E_x2 = loc2 + scale2 * delta * np.sqrt(2.0 / np.pi)

    # Stack
    data_train_tot = np.column_stack([x1, x2]).astype('float32')

    return data_train_tot#, (E_x1, E_x2)

def generate_two_moons(N_train_tot=100_000, seed=1, noise=0.1):
    np.random.seed(seed)
    data_train_tot, y_train_tot = make_moons(
        n_samples=N_train_tot,
        noise=noise,
        random_state=seed
    )
    return data_train_tot.astype('float32')#, y_train_tot.astype('int64')

