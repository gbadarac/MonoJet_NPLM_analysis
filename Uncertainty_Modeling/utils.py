import numpy as np
import torch
import torch.nn.functional as F
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

def make_flow(num_layers, hidden_features, num_bins, num_blocks, num_features=2, num_context=None, perm=True):
    base_dist = StandardNormal(shape=(num_features,))
    transforms = []
    if num_context == 0:
        num_context = None
    for i in range(num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform( #randomly initialized by PyTorch number generator
            features=num_features,
            context_features=num_context,
            hidden_features=hidden_features,
            num_bins=num_bins,
            num_blocks=num_blocks,
            tail_bound=10.0,
            tails='linear',
            dropout_probability=0.2,
            use_batch_norm=False
        ))
        if i < num_layers - 1 and perm:
            transforms.append(ReversePermutation(features=num_features))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

def average_state_dicts(state_dicts):
    """Uniformly average a list of PyTorch state dicts"""
    n = len(state_dicts) #stores number of dictionaries, i.e. how many bootstrapped datasets you trained 
    avg_dict = {} #initialized an empty dictionary where we will store the averaged parameters 
    for key in state_dicts[0]: #loops through all parameter names in the first model (every model has the same parameter names so we can just take the keys from state_dict[0] to iterate over all params)
        avg_dict[key] = sum(sd[key] for sd in state_dicts) / n #for each key we sum across models and then average 
    return avg_dict

def probs(weights, model_probs):
    """
    Args:
        weights: torch tensor of shape (M,) with requires_grad=True
        model_probs: torch tensor of shape (N, M), where each column is f_i(x) for all x
    Returns:
        p(x) ≈ ∑ w_i * f_i(x) for each x (shape N,)
    """
    return (model_probs * weights).sum(dim=1)  # shape: (N,)

def nll(weights, model_probs):
    """
    Args:
        weights: torch tensor with requires_grad=True
        model_probs: torch.tensor of shape (N, M) with f_i(x)
    Returns:
        Scalar negative log-likelihood
    """
    p_x = probs(weights, model_probs) + 1e-8  # prevent log(0)
    nll = -torch.log(p_x).mean() #averaging the loss over all datapoints 
    return nll # scalar

def nll_numpy(weights_np, model_probs, device="cpu"):
    """
    NumPy-compatible wrapper around the PyTorch nll(), preserving gradient info for Hessian estimation.
    """
    weights_tensor = torch.tensor(weights_np, dtype=torch.float32, requires_grad=True, device=device)
    return nll(weights_tensor, model_probs).item()


def ensemble_pred(weights, model_probs):
    """
    Compute ensemble prediction f(x) = sum_i w_i f_i(x)
    """
    model_vals=(model_probs * torch.tensor(weights, device=model_probs.device)).sum(dim=1)
    return model_vals.cpu().numpy()

def ensemble_unc(weights_unc, model_probs):
    """
    Compute uncertainty on ensemble prediction: sigma^2_f(x) = sum_i (sigma_w_i * f_i(x))^2
    """
    sigma_sq = (torch.tensor(weights_unc, device=model_probs.device) * model_probs).pow(2).sum(dim=1)
    model_unc = torch.sqrt(sigma_sq)
    return model_unc.cpu().numpy()
