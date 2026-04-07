"""
HIKER: Hierarchical Ensemble of Gaussian Kernels for density estimation.

Multi-layer architecture. Each layer has M_l kernels at width sigma_l.

Two gauge-fixing modes (controlled by use_norm_kernel):
  True:  A global normalization kernel with w_norm = 1 - sum(w).
         f(x) = sum_l sum_i w_i K_i(x) + w_norm * K_norm(x)
  False: Post-hoc normalization. f(x) = g(x)/Z, Z = sum(w).
         Training uses unnormalized g(x).

Training modes: "sequential" (layer-by-layer) or "joint" (all at once).
Both support early stopping and mini-batch training.

Ensemble mode (N_ENSEMBLE > 1):
  - Trains N models on bootstrap resamples
  - build_ensemble_model(): collects all kernels, rescales weights by 1/N
  - build_ensemble_covariance(): block-diagonal covariance
  - ensemble_density(): evaluates ensemble with per-seed norm kernels
  - Per-seed norm kernels stored separately when USE_NORM_KERNEL=True

Also provides: sampling (direct mixture or hit-or-miss), model reconstruction.
"""

import math
import torch
import torch.nn as nn
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Low-level kernel layer
# ──────────────────────────────────────────────────────────────────────

class KernelLayer(nn.Module):
    """Layer of M isotropic Gaussian kernels sharing a single width."""

    def __init__(self, centroids, sigma, train_centroids=False, train_width=False):
        super().__init__()
        self.M, self.d = centroids.shape
        self.centroids = nn.Parameter(centroids.double(), requires_grad=train_centroids)
        self.width = nn.Parameter(
            torch.tensor([sigma], dtype=torch.float64), requires_grad=train_width
        )

    @property
    def sigma(self):
        return self.width.item()

    def set_sigma(self, sigma):
        self.width.data.fill_(sigma)

    def gauss_const(self):
        return 1.0 / ((2 * math.pi) ** (self.d / 2) * self.width ** self.d)

    def kernels(self, x):
        """Evaluate all M kernels at points x.  Returns (N, M)."""
        x = x.double()
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return self.gauss_const() * torch.exp(-0.5 * dist_sq / self.width ** 2)


# ──────────────────────────────────────────────────────────────────────
# Single HIKER layer (kernels + coefficients, no norm kernel)
# ──────────────────────────────────────────────────────────────────────

class HIKERLayer(nn.Module):
    """
    One resolution layer of HIKER: M_l kernels with trainable coefficients.
    No normalization kernel — that lives in the parent HIKER model.
    """

    def __init__(self, centroids, sigma, coeffs_init=None,
                 train_centroids=True, coeffs_clip=None):
        super().__init__()
        M, d = centroids.shape
        self.kernel_layer = KernelLayer(centroids, sigma, train_centroids=train_centroids)
        if coeffs_init is None:
            coeffs_init = torch.zeros(M, dtype=torch.float64)
        self.coeffs = nn.Parameter(coeffs_init.double(), requires_grad=True)
        self.coeffs_clip = coeffs_clip

    @property
    def M(self):
        return self.kernel_layer.M

    @property
    def d(self):
        return self.kernel_layer.d

    def set_sigma(self, sigma):
        self.kernel_layer.set_sigma(sigma)

    def clip_coeffs(self):
        if self.coeffs_clip is not None:
            self.coeffs.data.clamp_(-self.coeffs_clip, self.coeffs_clip)

    def density(self, x):
        """Layer contribution: sum_i w_i K_i(x). Returns (N,)."""
        K = self.kernel_layer.kernels(x)
        return K @ self.coeffs


# ──────────────────────────────────────────────────────────────────────
# Multi-layer HIKER with single global normalization kernel
# ──────────────────────────────────────────────────────────────────────

class HIKER(nn.Module):
    """
    Hierarchical (multi-layer) HIKER density estimator.

    The full density is:
        f(x) = sum_l sum_i w_i^(l) K_i^(l)(x)  +  w_norm * K_norm(x)

    where w_norm = 1 - sum of all free weights.
    K_norm is a single global normalization kernel with:
        - centre = mean of all layer centroids
        - width  = mean of all layer widths

    Parameters
    ----------
    layers_config : list of dicts, one per layer. Each dict has:
        - "centroids": Tensor (M_l, d)
        - "sigma": float
        - "coeffs_init": Tensor (M_l,) or None
        - "train_centroids": bool (default True)
        - "coeffs_clip": float or None
    """

    def __init__(self, layers_config, norm_centre=None, train_norm_centre=True,
                 use_norm_kernel=True):
        """
        Parameters
        ----------
        layers_config : list of dicts (one per layer)
        norm_centre : Tensor (1, d) or None
            Initial centre for the normalization kernel.
            If None, uses the mean of all layer centroids.
        train_norm_centre : bool
            If False, the normalization kernel centre is frozen at its initial value.
        use_norm_kernel : bool
            If True (default), gauge is fixed via a normalization kernel.
            If False, no norm kernel; density is normalized post-hoc by Z = sum(w).
            The Hessian must then be handled via projection.
        """
        super().__init__()
        self.use_norm_kernel = use_norm_kernel
        self.layers = nn.ModuleList()
        for cfg in layers_config:
            layer = HIKERLayer(
                centroids=cfg["centroids"],
                sigma=cfg["sigma"],
                coeffs_init=cfg.get("coeffs_init"),
                train_centroids=cfg.get("train_centroids", True),
                coeffs_clip=cfg.get("coeffs_clip"),
            )
            self.layers.append(layer)

        # Global normalization kernel (only if enabled)
        if use_norm_kernel:
            if norm_centre is None:
                all_centroids = torch.cat([cfg["centroids"] for cfg in layers_config], dim=0)
                norm_centre = all_centroids.mean(dim=0, keepdim=True)
            mean_sigma = np.mean([cfg["sigma"] for cfg in layers_config])
            self.norm_kernel = KernelLayer(norm_centre, mean_sigma,
                                           train_centroids=train_norm_centre)
        else:
            self.norm_kernel = None

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def d(self):
        return self.layers[0].d

    def w_norm(self):
        """Global normalization weight: 1 - sum of all free weights.
        Only meaningful when use_norm_kernel=True."""
        total = sum(layer.coeffs.sum() for layer in self.layers)
        return 1.0 - total

    def Z(self):
        """Sum of all free weights (for post-hoc normalization)."""
        return sum(layer.coeffs.sum() for layer in self.layers)

    def _raw_density(self, x, j=None):
        """Unnormalized sum of layer contributions up to layer j (or all)."""
        f = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
        n = self.n_layers if j is None else j + 1
        for l in range(n):
            f = f + self.layers[l].density(x)
        return f

    def density(self, x):
        """Full density. Returns (N,)."""
        g = self._raw_density(x)
        if self.use_norm_kernel:
            K_n = self.norm_kernel.kernels(x)
            return g + self.w_norm() * K_n.squeeze(1)
        else:
            return g / (self.Z() + 1e-30)

    def density_cumsum_j(self, x, j):
        """Cumulative density up to layer j (inclusive)."""
        g = self._raw_density(x, j)
        if self.use_norm_kernel:
            K_n = self.norm_kernel.kernels(x)
            return g + self.w_norm() * K_n.squeeze(1)
        else:
            # Normalize by sum of weights in layers 0..j
            Z_j = sum(self.layers[l].coeffs.sum() for l in range(j + 1))
            return g / (Z_j + 1e-30)

    def nll(self, x):
        """NLL for training. Uses unnormalized density when no norm kernel
        (normalization is applied post-hoc, not during training)."""
        if self.use_norm_kernel:
            f = self.density(x)
        else:
            f = self._raw_density(x)
        return -torch.log(f + 1e-10).sum()

    def nll_cumsum_j(self, x, j):
        """NLL for sequential training up to layer j."""
        if self.use_norm_kernel:
            f = self.density_cumsum_j(x, j)
        else:
            f = self._raw_density(x, j)
        return -torch.log(f + 1e-10).mean()

    def clip_coeffs(self):
        for layer in self.layers:
            layer.clip_coeffs()

    def set_sigma_j(self, sigma, j):
        self.layers[j].set_sigma(sigma)

    # ── Accessors ─────────────────────────────────────────────────

    def get_all_coeffs(self):
        """All coefficients. With norm kernel: (M_total+1,) including w_norm.
        Without: (M_total,)."""
        parts = [layer.coeffs for layer in self.layers]
        if self.use_norm_kernel:
            parts.append(self.w_norm().unsqueeze(0))
        return torch.cat(parts)

    def get_free_coeffs(self):
        """Concatenated free (trainable) coefficients: (M_total,)."""
        return torch.cat([layer.coeffs for layer in self.layers])

    def get_all_centroids(self):
        """All centroids. With norm kernel: (M_total+1, d). Without: (M_total, d)."""
        parts = [layer.kernel_layer.centroids for layer in self.layers]
        if self.use_norm_kernel:
            parts.append(self.norm_kernel.centroids)
        return torch.cat(parts, dim=0)

    def get_kernel_matrices(self, x):
        """
        Precompute kernel matrices for all layers + norm kernel (if present).

        Returns
        -------
        K_list : list of Tensor (N, M_l) — main kernels per layer
        K_norm : Tensor (N, 1) or None — normalization kernel (None if disabled)
        """
        K_list = [layer.kernel_layer.kernels(x) for layer in self.layers]
        K_norm = self.norm_kernel.kernels(x) if self.use_norm_kernel else None
        return K_list, K_norm

    def get_all_widths(self):
        """All widths. With norm kernel: (M_total+1,). Without: (M_total,)."""
        widths = []
        for layer in self.layers:
            sigma = layer.kernel_layer.width.item()
            widths.append(torch.full((layer.M,), sigma, dtype=torch.float64))
        if self.use_norm_kernel:
            widths.append(torch.full((1,), self.norm_kernel.width.item(), dtype=torch.float64))
        return torch.cat(widths)


# ──────────────────────────────────────────────────────────────────────
# Width annealing
# ──────────────────────────────────────────────────────────────────────

def annealing_linear(t, ini, fin, t_fin):
    if t < t_fin:
        return ini + (fin - ini) * (t + 1) / t_fin
    return fin


# ──────────────────────────────────────────────────────────────────────
# Sequential training
# ──────────────────────────────────────────────────────────────────────

def train_hiker(model, x_train, layer_configs, lr=1e-4, batch_size=None,
                mode="sequential", early_stopping_patience=None, verbose=True):
    """
    Train a multi-layer HIKER model.

    Parameters
    ----------
    model : HIKER
    x_train : Tensor (N, d)
    layer_configs : list of dicts per layer with:
        epochs, patience, width_init, width_fin, t_ini, decay_epochs
    lr : float
    batch_size : int or None
    mode : str
        "sequential" — train layer 0, then 0+1, etc. Only parameters for
            layers 0..n and the norm kernel are optimized at stage n.
        "joint" — train all layers simultaneously. Uses the epochs/patience
            from the first layer config.
    verbose : bool

    Returns
    -------
    loss_history : list of (layer_idx, epoch, loss_value)
    coeffs_history : list of (layer_idx, epoch, coeffs_array)
        coeffs_array is a numpy copy of all free coefficients at that step.
    """
    loss_history = []
    coeffs_history = []
    centroids_history = []
    N = x_train.shape[0]
    use_batches = batch_size is not None and batch_size < N

    if mode == "joint":
        return _train_joint(model, x_train, layer_configs, lr, batch_size, verbose,
                            early_stopping_patience=early_stopping_patience)

    # ── Sequential training ───────────────────────────────────────
    use_es = early_stopping_patience is not None
    global_epoch = 0

    for n in range(model.n_layers):
        cfg = layer_configs[n]
        epochs = cfg["epochs"]
        patience = cfg.get("patience", 1000)
        width_ini = cfg.get("width_init", model.layers[n].kernel_layer.sigma)
        width_fin = cfg.get("width_fin", width_ini)
        t_ini = cfg.get("t_ini", 0)
        decay_epochs = cfg.get("decay_epochs", 0.5)
        t_fin = int(decay_epochs * epochs)

        # Only optimize layers 0..n and the norm kernel (not future layers)
        params = list(model.norm_kernel.parameters()) if model.norm_kernel is not None else []
        for l in range(n + 1):
            params.extend(model.layers[l].parameters())
        n_params = sum(p.numel() for p in params)

        # Reset early stopping per layer stage
        if use_es:
            best_loss = float("inf")
            es_counter = 0
            best_state = None

        if verbose:
            msg = (f"\n  Layer {n}/{model.n_layers - 1}: "
                   f"M={model.layers[n].M}, "
                   f"width {width_ini:.4f} -> {width_fin:.4f}, "
                   f"{epochs} epochs, {n_params} params")
            if use_batches:
                msg += f", batch_size={batch_size}"
            if use_es:
                msg += f", early_stopping={early_stopping_patience}"
            print(msg)

        optimizer = torch.optim.Adam(params, lr=lr)

        for i in range(epochs):
            global_epoch += 1
            if i > t_ini:
                new_sigma = annealing_linear(i - t_ini, width_ini, width_fin, t_fin)
                model.set_sigma_j(new_sigma, n)

            if use_batches:
                idx = torch.randint(N, (batch_size,), device=x_train.device)
                x_batch = x_train[idx]
            else:
                x_batch = x_train

            optimizer.zero_grad()
            loss = model.nll_cumsum_j(x_batch, j=n)
            loss.backward()
            optimizer.step()
            model.clip_coeffs()

            if (i + 1) % patience == 0:
                with torch.no_grad():
                    val = model.nll_cumsum_j(x_train, j=n).item()
                    coeffs_snap = model.get_free_coeffs().detach().cpu().numpy().copy()
                    centroids_snap = torch.cat(
                        [layer.kernel_layer.centroids for layer in model.layers]
                    ).detach().cpu().numpy().copy()
                loss_history.append((n, global_epoch, val))
                coeffs_history.append((n, global_epoch, coeffs_snap))
                centroids_history.append((n, global_epoch, centroids_snap))

                if use_es:
                    improved = val < best_loss
                    if improved:
                        best_loss = val
                        es_counter = 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        es_counter += 1

                    if verbose:
                        marker = "*" if improved else ""
                        print(f"    epoch {i+1:>6d} (global {global_epoch:>6d})  "
                              f"NLL/N = {val:.6f}  "
                              f"es={es_counter}/{early_stopping_patience} {marker}")

                    if es_counter >= early_stopping_patience:
                        if verbose:
                            print(f"    Early stopping at epoch {i+1}. "
                                  f"Restoring best (NLL/N={best_loss:.6f}).")
                        model.load_state_dict(best_state)
                        break
                else:
                    if verbose:
                        print(f"    epoch {i+1:>6d} (global {global_epoch:>6d})  NLL/N = {val:.6f}")

    return loss_history, coeffs_history, centroids_history


def _train_joint(model, x_train, layer_configs, lr, batch_size, verbose,
                  early_stopping_patience=None):
    """
    Train all layers simultaneously with optional early stopping.

    Early stopping monitors the full-dataset NLL. When it hasn't improved
    for `early_stopping_patience` consecutive evaluation steps, training
    stops and the best model is restored.

    Parameters
    ----------
    early_stopping_patience : int or None
        Number of evaluation steps (each `patience` epochs) without
        improvement before stopping. None = no early stopping.
    """
    cfg = layer_configs[0]
    epochs = cfg["epochs"]
    patience = cfg.get("patience", 1000)
    N = x_train.shape[0]
    use_batches = batch_size is not None and batch_size < N
    loss_history = []
    coeffs_history = []
    centroids_history = []

    use_es = early_stopping_patience is not None
    if use_es:
        best_loss = float("inf")
        es_counter = 0
        best_state = None

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        msg = (f"\n  Joint training: all {model.n_layers} layers, "
               f"{n_params} params, {epochs} epochs")
        if use_batches:
            msg += f", batch_size={batch_size}"
        if use_es:
            msg += f", early_stopping={early_stopping_patience} evals"
        print(msg)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        if use_batches:
            idx = torch.randint(N, (batch_size,), device=x_train.device)
            x_batch = x_train[idx]
        else:
            x_batch = x_train

        optimizer.zero_grad()
        loss = model.nll_cumsum_j(x_batch, j=model.n_layers - 1)
        loss.backward()
        optimizer.step()
        model.clip_coeffs()

        if (i + 1) % patience == 0:
            with torch.no_grad():
                val = model.nll_cumsum_j(x_train, j=model.n_layers - 1).item()
                coeffs_snap = model.get_free_coeffs().detach().cpu().numpy().copy()
                centroids_snap = torch.cat(
                    [layer.kernel_layer.centroids for layer in model.layers]
                ).detach().cpu().numpy().copy()

            loss_history.append((-1, i + 1, val))
            coeffs_history.append((-1, i + 1, coeffs_snap))
            centroids_history.append((-1, i + 1, centroids_snap))

            if use_es:
                improved = val < best_loss
                if improved:
                    best_loss = val
                    es_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    es_counter += 1

                if verbose:
                    marker = "*" if improved else ""
                    print(f"    epoch {i+1:>6d}  NLL/N = {val:.6f}  "
                          f"es={es_counter}/{early_stopping_patience} {marker}")

                if es_counter >= early_stopping_patience:
                    if verbose:
                        print(f"    Early stopping at epoch {i+1}. "
                              f"Restoring best (NLL/N={best_loss:.6f}).")
                    model.load_state_dict(best_state)
                    break
            else:
                if verbose:
                    print(f"    epoch {i+1:>6d}  NLL/N = {val:.6f}")

    return loss_history, coeffs_history, centroids_history


# ──────────────────────────────────────────────────────────────────────
# Convenience constructors
# ──────────────────────────────────────────────────────────────────────

def build_hiker_from_config(x_train, cfg):
    """
    Build a HIKER model from a pipeline config dict.

    Returns model, layer_train_configs.
    """
    N, d = x_train.shape

    M_list = cfg.get("M_KERNELS", 50)
    if isinstance(M_list, int):
        M_list = [M_list]
    n_layers = len(M_list)

    sigma_list = cfg.get("KERNEL_SIGMA", 0.10)
    if isinstance(sigma_list, (int, float)):
        sigma_list = [sigma_list] * n_layers

    epochs_list = cfg.get("EPOCHS", 100000)
    if isinstance(epochs_list, int):
        epochs_list = [epochs_list] * n_layers

    annealing = "WIDTH_INIT" in cfg
    if annealing:
        width_init_list = cfg["WIDTH_INIT"]
        if isinstance(width_init_list, (int, float)):
            width_init_list = [width_init_list] * n_layers
        t_ini = cfg.get("T_INI", 0)
        decay_epochs = cfg.get("DECAY_EPOCHS", 0.5)

    train_centroids = cfg.get("TRAIN_CENTROIDS", True)
    coeffs_clip = cfg.get("COEFFS_CLIP", 100.0)
    patience = cfg.get("PATIENCE", 5000)

    layers_config = []
    layer_train_configs = []

    for l in range(n_layers):
        M_l = M_list[l]
        sigma_l = sigma_list[l]
        init_sigma = width_init_list[l] if annealing else sigma_l

        idx = torch.randperm(N)[:M_l]
        centroids_l = x_train[idx].clone()

        layers_config.append({
            "centroids": centroids_l,
            "sigma": init_sigma,
            "coeffs_init": torch.zeros(M_l, dtype=torch.float64),
            "train_centroids": train_centroids,
            "coeffs_clip": coeffs_clip,
        })

        train_cfg = {
            "epochs": epochs_list[l],
            "patience": patience,
            "width_init": init_sigma,
            "width_fin": sigma_l,
        }
        if annealing:
            train_cfg["t_ini"] = t_ini
            train_cfg["decay_epochs"] = decay_epochs

        layer_train_configs.append(train_cfg)

    use_norm_kernel = cfg.get("USE_NORM_KERNEL", True)
    train_norm_centre = cfg.get("TRAIN_NORM_CENTRE", False)
    norm_centre = x_train.mean(dim=0, keepdim=True).double()
    model = HIKER(layers_config, norm_centre=norm_centre,
                  train_norm_centre=train_norm_centre,
                  use_norm_kernel=use_norm_kernel)
    return model, layer_train_configs


def reconstruct_hiker(train_config, d=2):
    """
    Reconstruct HIKER architecture from train_config.json with dummy centroids,
    so that model.load_state_dict() can be called.
    """
    layer_sizes = train_config.get("layer_sizes")
    if layer_sizes is None:
        M = train_config["M_KERNELS"]
        layer_sizes = M if isinstance(M, list) else [M]

    # Sigma list: for ensemble models, layer_sizes may be much longer than
    # KERNEL_SIGMA. Use a dummy value — load_state_dict overwrites it.
    sigma_list = train_config.get("KERNEL_SIGMA", 0.10)
    if isinstance(sigma_list, (int, float)):
        sigma_list = [sigma_list] * len(layer_sizes)
    # Extend if needed (ensemble has more layers than original config)
    while len(sigma_list) < len(layer_sizes):
        sigma_list.append(sigma_list[-1] if sigma_list else 0.1)

    train_centroids = train_config.get("TRAIN_CENTROIDS", True)
    coeffs_clip = train_config.get("COEFFS_CLIP", 100.0)

    layers_config = []
    for l, M_l in enumerate(layer_sizes):
        layers_config.append({
            "centroids": torch.zeros(M_l, d, dtype=torch.float64),
            "sigma": sigma_list[l] if l < len(sigma_list) else sigma_list[-1],
            "coeffs_init": torch.zeros(M_l, dtype=torch.float64),
            "train_centroids": train_centroids,
            "coeffs_clip": coeffs_clip,
        })

    use_norm_kernel = train_config.get("USE_NORM_KERNEL", True)
    train_norm_centre = train_config.get("TRAIN_NORM_CENTRE", False)
    return HIKER(layers_config, train_norm_centre=train_norm_centre,
                 use_norm_kernel=use_norm_kernel)


# ──────────────────────────────────────────────────────────────────────
# Ensemble: build averaged model from N bootstrap-trained models
# ──────────────────────────────────────────────────────────────────────

def build_ensemble_model(models, device="cpu"):
    """
    Build an ensemble HIKER from N trained models.

    The ensemble density is: f_ens(x) = (1/N) sum_s f_s(x)

    Two modes depending on whether seed models use norm kernels:

    USE_NORM_KERNEL=True per seed:
        Each seed keeps its norm kernel. The ensemble stores norm kernels
        separately with a mapping of which free weights belong to which seed.
        Each seed's density contribution:
            f_s(x) = sum_i w_i^(s) K_i^(s)(x) + (1-sum(w_s)) K_norm^(s)(x)
        Weights are divided by N_ens. The norm weight for each seed is
        dynamically computed as (1/N) * (1 - sum(w_s)).

    USE_NORM_KERNEL=False per seed:
        Each seed's weights are rescaled by 1/(N*Z_s). No norm kernels.
        Density is post-hoc normalized.

    Returns
    -------
    ensemble : HIKER
    ensemble_info : dict with:
        "layer_map": list of (seed_idx, layer_idx)
        "seed_boundaries": list of (start, end) indices into free weights per seed
        "norm_kernels": list of KernelLayer (one per seed) or None
        "N_ens": int
    """
    N_ens = len(models)
    use_norm = models[0].use_norm_kernel
    layers_config = []
    layer_map = []
    seed_boundaries = []  # (start_idx, end_idx) into free weights per seed

    offset = 0
    for s, model in enumerate(models):
        with torch.no_grad():
            start = offset
            if use_norm:
                for l, layer in enumerate(model.layers):
                    centroids = layer.kernel_layer.centroids.detach().clone()
                    sigma = layer.kernel_layer.width.item()
                    coeffs = layer.coeffs.detach().clone() / N_ens
                    layers_config.append({
                        "centroids": centroids,
                        "sigma": sigma,
                        "coeffs_init": coeffs,
                        "train_centroids": False,
                        "coeffs_clip": None,
                    })
                    layer_map.append((s, l))
                    offset += layer.M
            else:
                Z_s = model.Z().item()
                for l, layer in enumerate(model.layers):
                    centroids = layer.kernel_layer.centroids.detach().clone()
                    sigma = layer.kernel_layer.width.item()
                    coeffs = layer.coeffs.detach().clone() / (N_ens * Z_s)
                    layers_config.append({
                        "centroids": centroids,
                        "sigma": sigma,
                        "coeffs_init": coeffs,
                        "train_centroids": False,
                        "coeffs_clip": None,
                    })
                    layer_map.append((s, l))
                    offset += layer.M
            seed_boundaries.append((start, offset))

    # Build the HIKER without norm kernel — norm kernels are stored separately
    ensemble = HIKER(layers_config, use_norm_kernel=False).to(device)

    # Collect norm kernels if applicable
    norm_kernels = None
    if use_norm:
        norm_kernels = []
        for model in models:
            nk = KernelLayer(
                model.norm_kernel.centroids.detach().clone(),
                model.norm_kernel.width.item(),
                train_centroids=False,
            ).to(device)
            norm_kernels.append(nk)

    # Freeze all parameters
    for p in ensemble.parameters():
        p.requires_grad_(False)

    ensemble_info = {
        "layer_map": layer_map,
        "seed_boundaries": seed_boundaries,
        "norm_kernels": norm_kernels,
        "N_ens": N_ens,
        "use_norm_per_seed": use_norm,
    }

    return ensemble, ensemble_info


def ensemble_density(ensemble, x, ensemble_info):
    """
    Evaluate the ensemble density at points x.

    For norm-kernel seeds:
        f_ens(x) = sum_s [ sum_i w_i^(s) K_i^(s)(x)
                          + (1/N - sum(w_s)) K_norm^(s)(x) ]
        where w_i^(s) are already divided by N, so the norm weight per seed
        is (1/N - sum(w_s_free)) = (1/N)(1 - sum(w_s_original)).

    For no-norm-kernel seeds:
        f_ens(x) = g(x)/Z  where g = sum of all layers, Z = sum of all weights
    """
    w = ensemble.get_free_coeffs()
    g = ensemble._raw_density(x)

    if ensemble_info["use_norm_per_seed"] and ensemble_info["norm_kernels"] is not None:
        N_ens = ensemble_info["N_ens"]
        for s, (start, end) in enumerate(ensemble_info["seed_boundaries"]):
            w_s = w[start:end]
            # w_s are already divided by N_ens, so sum(w_s) = sum(w_s_orig)/N
            # The norm weight for this seed: (1/N) - sum(w_s) = (1/N)(1 - sum(w_s_orig))
            w_norm_s = 1.0 / N_ens - w_s.sum()
            K_norm_s = ensemble_info["norm_kernels"][s].kernels(x)  # (N_pts, 1)
            g = g + w_norm_s * K_norm_s.squeeze(1)
        return g  # already normalized (each seed contributes 1/N to total)
    else:
        Z = w.sum()
        return g / (Z + 1e-30)


def build_ensemble_covariance(models, covariances, device="cpu"):
    """
    Build the block-diagonal covariance for the ensemble.

    Each seed's covariance is scaled by 1/N^2 (norm kernel) or 1/(N*Z_s)^2
    (no norm kernel). The covariance is over the FREE weights only — norm
    kernel weights are derived and don't appear in the covariance.
    """
    N_ens = len(models)
    blocks = []

    for s, (model, cov_s) in enumerate(zip(models, covariances)):
        if model.use_norm_kernel:
            scale = 1.0 / (N_ens ** 2)
            blocks.append(scale * cov_s)
        else:
            Z_s = model.Z().item()
            scale = 1.0 / (N_ens * Z_s) ** 2
            blocks.append(scale * cov_s)

    total_size = sum(b.shape[0] for b in blocks)
    cov_ens = torch.zeros(total_size, total_size, dtype=torch.float64, device=device)
    offset = 0
    for block in blocks:
        M = block.shape[0]
        cov_ens[offset:offset + M, offset:offset + M] = block.to(device)
        offset += M

    return cov_ens


# ──────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────

def sample_from_hiker(model, N, seed=None):
    """
    Direct mixture sampling. All weights (including w_norm) must be >= 0.
    """
    if seed is not None:
        np.random.seed(seed)

    weights_all = []
    centroids_all = []
    sigmas_all = []

    with torch.no_grad():
        for layer in model.layers:
            w = layer.coeffs.cpu().numpy()
            c = layer.kernel_layer.centroids.cpu().numpy()
            s = layer.kernel_layer.width.item()
            weights_all.append(w)
            centroids_all.append(c)
            sigmas_all.extend([s] * len(w))

        # Normalization kernel (if present)
        if model.use_norm_kernel:
            w_n = model.w_norm().item()
            c_n = model.norm_kernel.centroids.cpu().numpy()
            s_n = model.norm_kernel.width.item()
            weights_all.append(np.array([w_n]))
            centroids_all.append(c_n)
            sigmas_all.append(s_n)

    weights = np.concatenate(weights_all)
    centroids = np.concatenate(centroids_all, axis=0)
    sigmas = np.array(sigmas_all)

    if np.any(weights < 0):
        raise ValueError(
            f"Cannot sample directly: {(weights < 0).sum()} negative weight(s). "
            "Use hit-or-miss sampling instead."
        )

    weights = weights / weights.sum()
    K = len(weights)
    d = centroids.shape[1]

    components = np.random.choice(K, size=N, p=weights)
    samples = np.zeros((N, d))
    for i in range(N):
        k = components[i]
        samples[i] = centroids[k] + sigmas[k] * np.random.randn(d)

    return samples


def sample_from_hiker_hit_or_miss(model, N, bounds, seed=None, max_trials=10_000_000):
    """Rejection sampling. Works with negative weights."""
    if seed is not None:
        np.random.seed(seed)

    bounds = np.asarray(bounds)
    lo, hi = bounds[:, 0], bounds[:, 1]
    d = len(bounds)

    probe = np.random.uniform(lo, hi, size=(5000, d))
    with torch.no_grad():
        f_probe = model.density(torch.from_numpy(probe).double().to(
            next(model.parameters()).device)).cpu().numpy()
    f_max = np.max(f_probe) * 1.5

    samples = []
    n_accept = 0
    n_trials = 0
    batch = min(100_000, max_trials)
    device = next(model.parameters()).device

    while n_accept < N and n_trials < max_trials:
        n_this = min(batch, max_trials - n_trials)
        x_prop = np.random.uniform(lo, hi, size=(n_this, d))
        u = np.random.rand(n_this)
        with torch.no_grad():
            f_val = model.density(torch.from_numpy(x_prop).double().to(device)).cpu().numpy()
        accept = (u < f_val / f_max) & (f_val > 0)
        samples.append(x_prop[accept])
        n_accept += accept.sum()
        n_trials += n_this

    samples = np.concatenate(samples, axis=0)[:N]
    if len(samples) < N:
        print(f"  WARNING: only accepted {len(samples)}/{N} after {n_trials} trials.")
    return samples


def sample_from_hiker_auto(model, N, bounds=None, seed=None):
    """Auto-choose direct vs hit-or-miss based on weight signs."""
    with torch.no_grad():
        weights = model.get_all_coeffs().cpu().numpy()
    n_neg = (weights < 0).sum()

    if n_neg == 0:
        print(f"  All weights positive — using direct mixture sampling")
        return sample_from_hiker(model, N, seed=seed)
    else:
        print(f"  {n_neg} negative weight(s) — using hit-or-miss sampling")
        if bounds is None:
            with torch.no_grad():
                centroids = model.get_all_centroids().cpu().numpy()
            sigmas = model.get_all_widths().cpu().numpy()
            max_sigma = sigmas.max()
            lo = centroids.min(axis=0) - 5 * max_sigma
            hi = centroids.max(axis=0) + 5 * max_sigma
            bounds = list(zip(lo, hi))
            print(f"  Auto bounds: {bounds}")
        return sample_from_hiker_hit_or_miss(model, N, bounds, seed=seed)


# ──────────────────────────────────────────────────────────────────────
# Numpy utilities
# ──────────────────────────────────────────────────────────────────────

def evaluate_gaussian_components(x, centroids, widths):
    """Evaluate each Gaussian component's PDF at points x (numpy)."""
    x = np.asarray(x, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64).ravel()
    N, d = x.shape
    M = centroids.shape[0]
    densities = np.zeros((N, M))
    for m in range(M):
        diff = x - centroids[m]
        sq_dist = np.sum((diff / widths[m]) ** 2, axis=1)
        norm = (2 * np.pi) ** (-d / 2) * widths[m] ** (-d)
        densities[:, m] = norm * np.exp(-0.5 * sq_dist)
    return densities
