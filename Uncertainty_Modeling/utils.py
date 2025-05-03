import numpy as np
import torch
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

# ------------------
# NF Definition
# ------------------
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