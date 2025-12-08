import numpy as np
import torch
from scipy.spatial.distance import pdist
import jax.numpy as jnp
from jax import random
                                                                                                                                                     
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

def candidate_sigma(data, perc=90):
    # this function estimates the width of the gaussian kernel. 
    # use on a (small) sample of reference data (standardize first if necessary)  
    pairw = pdist(data)
    return np.around(np.percentile(pairw,perc),2)

  
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
