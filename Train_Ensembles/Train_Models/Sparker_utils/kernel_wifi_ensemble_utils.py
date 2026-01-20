#!/usr/bin/env python

import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# -------------------------------------------------------------------
# Make SPARKutils importable
# -------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(1, str(THIS_DIR))

from SPARKutils import *  # provides Hierarchical

# -------------------------------------------------------------------
# Ensemble definition
# -------------------------------------------------------------------
# kernel_wifi_ensemble_utils.py

class Ensemble(nn.Module):
    def __init__(
        self,
        weights_init,
        centroids_init,
        coefficients_init,
        widths_init,
        resolution_const,
        resolution_scale,
        lambda_norm=0.0,
        coeffs_clip=0.0,
        train_centroids=False,
        train_coeffs=False,
        train_widths=False,
        train_weights=True,
        model_type="Soft-SparKer2",
        weights_activation=None,
    ):
        """
        weights_init      : (n_ensemble,)
        centroids_init    : list length n_ensemble, each [n_layers][m_l, d]
        coefficients_init : list length n_ensemble, each [n_layers][m_l]
        widths_init       : list length n_ensemble, each [n_layers][m_l, d]
        """
        super().__init__()
        self.lambda_norm = lambda_norm  # kept for possible external use
        self.n_ensemble = len(centroids_init)
        self.tol = 1e-7
        self.softmax = torch.softmax
        self.weights_activation = weights_activation

        # Build the ensemble as a ModuleList
        self.ensemble = nn.ModuleList(
            [
                Hierarchical(
                    input_shape=(None, 1),
                    centroids_list=centroids_init[i],
                    widths_list=widths_init[i],
                    coeffs_list=coefficients_init[i],
                    resolution_const=resolution_const,
                    resolution_scale=resolution_scale,
                    coeffs_clip=coeffs_clip,
                    train_widths=train_widths,
                    train_coeffs=train_coeffs,
                    train_centroids=train_centroids,
                    positive_coeffs=False,
                    probability_coeffs=False,  # softmax of coefficients
                    model=model_type,
                )
                for i in range(self.n_ensemble)
            ]
        )

        # Trainable ensemble weights
        self.weights = nn.Parameter(weights_init.float(), requires_grad=train_weights)
        
    @torch.no_grad()
    def member_probs(self, x):
        """
        Returns (N, M) tensor with f_j(x_n) for each ensemble member.
        """
        return torch.stack(
            [m.call(x)[-1, :, 0] / m.get_norm()[-1] for m in self.ensemble],
            dim=1
        )

    def forward(self, x):
        """
        Evaluate all ensemble members and combine with current weights.
        Returns p(x) = sum_i w_i f_i(x).
        """
        outs = torch.stack(
            [m.call(x)[-1, :, 0] / m.get_norm()[-1] for m in self.ensemble], dim=1
        )  # (batch, n_ensemble)

        if self.weights_activation == "softmax":
            w = self.softmax(self.weights.view(1, -1), dim=1)
        else:
            w = self.weights.view(1, -1)

        weighted = outs * w  # broadcasting
        return weighted.sum(dim=1)  # (batch,)
    
    def norm_regularization(self):
        return self.lambda_norm * (self.weights.sum() - 1.0) 

    def loss(self, x):
        p = self.forward(x)
        return -torch.log(p+1e-12).sum() + self.norm_regularization()

    def count_trainable_parameters(self, verbose=False):
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if verbose:
                    print(f"{name}: {param.numel()} trainable parameters")
                total += param.numel()
        if verbose:
            print(f"Total trainable parameters: {total}")
        return total

# -------------------------------------------------------------------
# Internal helpers to load config + members
# -------------------------------------------------------------------
def _load_kernel_config(folder_path: Path):
    with open(folder_path / "config.json", "r") as f:
        config_json = json.load(f)

    n_kernels = config_json["number_centroids"]
    n_layers = len(n_kernels)
    split_indices = np.cumsum(n_kernels)[:-1]

    resolution_scale = np.array(config_json["resolution_scale"]).reshape((-1,))
    resolution_const = np.array(config_json["resolution_const"]).reshape((-1,))
    resolution_const_t = torch.from_numpy(resolution_const).double()
    resolution_scale_t = torch.from_numpy(resolution_scale).double()

    coeffs_clip = config_json.get("coeffs_clip", 0.0)
    model_type = config_json["model"]

    return (
        config_json,
        n_layers,
        split_indices,
        resolution_const_t,
        resolution_scale_t,
        coeffs_clip,
        model_type,
    )


def _load_kernel_members(folder_path: Path, n_wifi_components, split_indices, n_layers):
    centroids_init, coefficients_init, widths_init = [], [], []

    for i in range(n_wifi_components):
        # use zero-padded seed directories: seed000, seed001, ...
        seed_dir = folder_path / f"seed{i:03d}"

        widths_hist = np.load(seed_dir / "widths_history.npy")

        # find last non-zero step
        count = -1
        for j in range(widths_hist.shape[0]):
            if widths_hist[j][0].sum() != 0:
                count += 1
            else:
                break

        centroids_hist = np.load(seed_dir / "centroids_history.npy")
        coeffs_hist = np.load(seed_dir / "coeffs_history.npy")

        centroids_i = centroids_hist[count]
        coeffs_i = coeffs_hist[count]
        widths_i = widths_hist[count]

        centroids_i = np.split(centroids_i, split_indices, axis=0)
        coeffs_i = np.split(coeffs_i, split_indices, axis=0)
        widths_i = np.split(widths_i, split_indices, axis=0)

        centroids_init.append(
            [torch.from_numpy(centroids_i[j]).double() for j in range(n_layers)]
        )
        coefficients_init.append(
            [torch.from_numpy(coeffs_i[j]).double() for j in range(n_layers)]
        )
        widths_init.append(
            [torch.from_numpy(widths_i[j]).double() for j in range(n_layers)]
        )

    return centroids_init, coefficients_init, widths_init


# -------------------------------------------------------------------
# Public builder
# -------------------------------------------------------------------
def build_wifi_ensemble(
    folder_path,
    n_wifi_components,
    lambda_norm=0.0,
    train_centroids=False,
    train_coeffs=False,
    train_widths=False,
    train_weights=True,
    weights_activation=None,
):
    """
    Build an Ensemble object from a SparKer 'folder_path'.

    Returns
    -------
    ensemble : Ensemble (nn.Module)
    config_json : dict
    """
    folder_path = Path(folder_path).resolve()

    (
        config_json,
        n_layers,
        split_indices,
        resolution_const_t,
        resolution_scale_t,
        coeffs_clip,
        model_type,
    ) = _load_kernel_config(folder_path)

    centroids_init, coefficients_init, widths_init = _load_kernel_members(
        folder_path, n_wifi_components, split_indices, n_layers
    )

    weights_init = torch.ones((n_wifi_components,), dtype=torch.float32) / float(
        n_wifi_components
    )

    ensemble = Ensemble(
        weights_init=weights_init,
        centroids_init=centroids_init,
        coefficients_init=coefficients_init,
        widths_init=widths_init,
        resolution_const=resolution_const_t,
        resolution_scale=resolution_scale_t,
        lambda_norm=lambda_norm,
        coeffs_clip=coeffs_clip,
        train_centroids=train_centroids,
        train_coeffs=train_coeffs,
        train_widths=train_widths,
        train_weights=train_weights,
        model_type=model_type,
        weights_activation=weights_activation,
    )

    return ensemble, config_json
