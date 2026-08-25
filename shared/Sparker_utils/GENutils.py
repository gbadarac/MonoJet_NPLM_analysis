import numpy as np
import torch
from scipy.spatial.distance import pdist
try:                                  # jax is OPTIONAL: only compute_bandwidths (the MMDfuse
    import jax.numpy as jnp           # bandwidth grid) uses it, and the LRT/NF pipelines never
    from jax import random            # call it. Guarding the import lets jax-free envs (e.g.
except ModuleNotFoundError:           # nplm_env, the NF-LRT env) still import GENutils for
    jnp = None                        # candidate_sigma / evaluate_gaussian_components. Envs
    random = None                     # that DO have jax (kernels_env) are unaffected.
                                                                                                                                                     
def standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        dataset_new[:, j] = (dataset[:, j]- mean)*1./ std
    return dataset_new

def inv_standardize(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        dataset_new[:, j] = dataset[:, j] * std + mean
    return dataset_new

def standardize_physics(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        if np.min(vec) < 0:
            vec = vec- mean
            vec = vec *1./ std
        elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1.       
            vec = vec *1./ mean
        dataset_new[:, j] = vec
    return dataset_new

def inv_standardize_physics(dataset, mean_all, std_all):
    dataset_new = np.copy(dataset)
    for j in range(dataset.shape[1]):
        mean, std = mean_all[j], std_all[j]
        vec  = dataset[:, j]
        if np.min(vec) < 0:
            dataset_new[:, j] = dataset[:, j] * std + mean
        elif np.max(vec) > 1.0:# Assume data is exponential -- just set mean to 1        
            dataset_new[:, j] = dataset[:, j] * mean
    return dataset_new

def candidate_sigma(data, perc=90, n_sub=2000):
    """NPLM bandwidth heuristic (= FLKutils_model.candidate_sigma): the perc-th
    percentile of pairwise distances on the first n_sub points. The subsample keeps
    pdist cheap on large data (O(n_sub^2), not O(N^2)). Informational anchor only —
    the LRT scripts pass a FIXED --kernel_sigma; a per-toy data-driven sigma would
    differ between calibration and test and bias the Z estimate."""
    sub = np.asarray(data[:n_sub], dtype=np.float64)
    return float(np.around(np.percentile(pdist(sub), perc), 1))

  
def compute_bandwidths(data, number_bandwidths):
    # Collection of bandwidths from MMDfuse
    distances = pdist(data)
    median = jnp.median(distances)
    distances = distances + (distances == 0) * median
    dd = jnp.sort(distances)
    lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
    lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
    bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
    return bandwidths

def evaluate_gaussian_components(x, centroids, widths):
    """
    Evaluate the probability density of each Gaussian component at input points.

    Parameters
    ----------
    x : array-like, shape (N, d)
        Input samples to evaluate.
    centroids : array-like, shape (M, d)
        Centers of each Gaussian component in d dimensions.
    widths : array-like, shape (M,)
        Standard deviations for each Gaussian (isotropic per component).

    Returns
    -------
    densities : ndarray, shape (N, M)
        Probability density of each component at each input point.
    """
    x = np.asarray(x)
    centroids = np.asarray(centroids)
    widths = np.asarray(widths)

    N, d = x.shape
    M = centroids.shape[0]

    densities = np.zeros((N, M), dtype=np.float64)

    for m in range(M):
        sigma = widths[m]
        diff = x - centroids[m]                      # (N, d)
        squared_dist = np.sum((diff / sigma) ** 2, axis=1)  # (N,)

        normalization = (2.0 * np.pi) ** (-d / 2.0) * (sigma ** (-d))
        densities[:, m] = normalization * np.exp(-0.5 * squared_dist)

    return densities
