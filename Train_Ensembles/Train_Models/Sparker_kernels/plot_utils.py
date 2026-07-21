# plot_utils.py  (in Train_Models/Sparker_kernels)

import os
import numpy as np
import torch

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


# ---------- basic helpers ----------

FONT = font_manager.FontProperties(family="serif", size=18)

def _setup_ax(fig):
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.15, 0.1, 0.78, 0.8])
    return ax

# ---------- history plots ----------

def plot_loss(epochs_history, loss_history, monitor_idx, output_folder):
    """Plot NLL + regularisers vs monitoring step."""
    fig = plt.figure(figsize=(9, 6))
    ax = _setup_ax(fig)
    ax.plot(epochs_history[2:monitor_idx], loss_history[2:monitor_idx], label="loss")
    ax.legend(prop=FONT, loc="best")
    ax.set_ylabel("Loss", fontsize=18, fontname="serif")
    ax.set_xlabel("Epochs", fontsize=18, fontname="serif")
    ax.tick_params(labelsize=16)
    ax.grid(True)
    fig.savefig(os.path.join(output_folder, "loss.pdf"))
    plt.close(fig)


def plot_centroids_history(epochs_history, centroids_history,
                           monitor_idx, d, total_M, output_folder):
    """Track each centroid position over training in each dimension."""
    for k in range(d):
        fig = plt.figure(figsize=(9, 6))
        ax = _setup_ax(fig)
        for m in range(total_M):
            ax.plot(epochs_history[:monitor_idx],
                    centroids_history[:monitor_idx, m, k:k+1],
                    label=f"{m}")
        ax.set_ylabel("Centroid loc", fontsize=18, fontname="serif")
        ax.set_xlabel("Epochs", fontsize=18, fontname="serif")
        ax.tick_params(labelsize=16)
        ax.grid(True)
        fig.savefig(os.path.join(output_folder, f"centroids_dim{k}.pdf"))
        plt.close(fig)


def plot_coeffs_history(epochs_history, coeffs_history,
                        monitor_idx, total_M, output_folder):
    """Track each kernel coefficient over training."""
    fig = plt.figure(figsize=(9, 6))
    ax = _setup_ax(fig)
    for m in range(total_M):
        ax.plot(epochs_history[:monitor_idx], coeffs_history[:monitor_idx, m],
                label=f"{m}")
    ax.set_ylabel("Coeffs", fontsize=18, fontname="serif")
    ax.set_xlabel("Epochs", fontsize=18, fontname="serif")
    ax.tick_params(labelsize=16)
    ax.grid(True)
    fig.savefig(os.path.join(output_folder, "coeffs.pdf"))
    plt.close(fig)


# ---------- model sampling + marginals ----------

def _final_layer_pdf(model, x, batch_size=200000):
    """
    Evaluate the final layer pdf on x, returning a 1D CPU numpy array of length N.
    Uses model.call(x) and normalises by model.get_norm() like your other plotting code.
    """
    model.eval()

    ref = model.get_centroids()
    device = ref.device
    dtype = ref.dtype

    x = x.to(device=device, dtype=dtype)

    out_list = []
    with torch.no_grad():
        norm = model.get_norm()  # [n_layers] tensor on device
        n_last = int(norm.shape[0]) - 1
        norm_last = norm[n_last]

        N = x.shape[0]
        for start in range(0, N, batch_size):
            xb = x[start:start + batch_size]
            out_all = model.call(xb)         # [n_layers, Nb, 1]
            pdf = (out_all[n_last, :, 0] / norm_last).clamp_min(0.0)
            out_list.append(pdf.detach().cpu())

    return torch.cat(out_list, dim=0).numpy()


def sample_from_kernel_model_rejection(
    model,
    num_samples,
    bounds,
    batch_proposals=50000,
    pmax_probe=200000,
    safety=1.2,
):
    """
    Rejection sample from the final layer pdf in a bounding box.

    bounds: list/array like [[x0_min, x0_max], [x1_min, x1_max], ...]
    returns: (num_samples, d) torch tensor on CPU
    """
    ref = model.get_centroids()
    device = ref.device
    dtype = ref.dtype

    bounds = np.asarray(bounds, dtype=np.float64)
    d = bounds.shape[0]
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    # Estimate pmax from random probes
    probe = lo + (hi - lo) * np.random.rand(pmax_probe, d)
    probe_t = torch.from_numpy(probe).to(device=device, dtype=dtype)
    p_probe = _final_layer_pdf(model, probe_t)
    pmax = float(np.max(p_probe)) * float(safety)
    if not np.isfinite(pmax) or pmax <= 0.0:
        raise RuntimeError("Could not estimate a valid pmax for rejection sampling.")

    accepted = []
    n_acc = 0

    while n_acc < num_samples:
        props = lo + (hi - lo) * np.random.rand(batch_proposals, d)
        props_t = torch.from_numpy(props).to(device=device, dtype=dtype)

        p = _final_layer_pdf(model, props_t)
        u = np.random.rand(batch_proposals)

        keep = u < (p / pmax)
        if np.any(keep):
            acc = props[keep]
            accepted.append(acc)
            n_acc += acc.shape[0]

    samples = np.concatenate(accepted, axis=0)[:num_samples]
    return torch.from_numpy(samples).float().cpu()


def sample_from_kernel_model_exact(model, num_samples):
    """
    EXACT ancestral sampling — ⚠ POSITIVE-COEFFICIENT MODELS ONLY.

    With all c_i >= 0 (positive_coeffs=True) the model is a proper mixture
    p(x) = sum_i (c_i/sum c) N(x; mu_i, diag(w_i^2)): draw component
    i ~ c_i/sum(c), then x ~ that Gaussian. No rejection, no pmax estimate.

    Why this exists: the rejection sampler estimates pmax from UNIFORM box
    probes, which cannot find a sharp 4D peak (~1e-8 of the box volume) ->
    pmax underestimated (measured 13x on the 4D QCD embedding) -> acceptance
    saturates and the sampled histogram CLIPS the peaks the model actually fits.

    ⚠ If you ever go back to SIGNED coefficients (positive_coeffs=False),
    this sampler is invalid — plot_kernel_marginals detects that case and
    falls back to the rejection sampler automatically (with a warning).
    """
    with torch.no_grad():
        c  = model.get_coeffs().detach().cpu().double().reshape(-1).numpy()  # [M]
        mu = model.get_centroids().detach().cpu().double().numpy()           # [M, d]
        w  = model.get_widths().detach().cpu().double().numpy()              # [M, d]
    if c.min() < 0:
        raise ValueError("exact mixture sampling needs all coeffs >= 0 "
                         "(signed model: use sample_from_kernel_model_rejection)")
    probs = c / c.sum()
    comp = np.random.choice(len(probs), size=num_samples, p=probs)
    samples = mu[comp] + w[comp] * np.random.randn(num_samples, mu.shape[1])
    return torch.from_numpy(samples).float().cpu()


def plot_kernel_marginals(
    model,
    x_data,
    feature_names,
    output_folder,
    num_samples=20000,
    bounds=None,
    bins=80,
    filename="marginals_kernel.png",
):
    """
    Plot 1D marginals, data vs samples from kernel model.
    Saves a single figure with one subplot per dimension.

    x_data: torch tensor or numpy array, shape (N, d)
    """
    if isinstance(x_data, np.ndarray):
        x_data_t = torch.from_numpy(x_data).float()
    else:
        x_data_t = x_data.detach().cpu().float()

    d = x_data_t.shape[1]
    if feature_names is None or len(feature_names) != d:
        feature_names = [f"Feature {i+1}" for i in range(d)]

    # Default bounds from data, with a small margin
    if bounds is None:
        x_np = x_data_t.numpy()
        mins = x_np.min(axis=0)
        maxs = x_np.max(axis=0)
        span = maxs - mins
        mins = mins - 0.05 * span
        maxs = maxs + 0.05 * span
        bounds = [[float(mins[i]), float(maxs[i])] for i in range(d)]

    # Sample from the kernel model.
    # Positive coeffs (positive_coeffs=True) -> the model is a proper mixture
    # -> EXACT ancestral sampling. Signed coeffs -> fall back to rejection
    # sampling, whose probe-based pmax CLIPS sharp peaks (measured 13x
    # underestimate on the 4D QCD embedding) — the plotted peaks are then a
    # LOWER BOUND, cross-check with analytic marginals before trusting them.
    coeffs_min = float(model.get_coeffs().detach().min())
    if coeffs_min >= 0:
        print("plot_kernel_marginals: positive coeffs -> exact mixture sampling")
        samples_t = sample_from_kernel_model_exact(model, num_samples=num_samples)
    else:
        print("plot_kernel_marginals: WARNING — signed coeffs (min c = "
              f"{coeffs_min:.3g}) -> rejection sampling; sharp peaks may be "
              "CLIPPED by the pmax estimate (do not over-interpret low peaks)")
        samples_t = sample_from_kernel_model_rejection(
            model,
            num_samples=num_samples,
            bounds=bounds,
        )
    samples_np = samples_t.numpy()
    data_np = x_data_t.numpy()

    # Plot
    fig, axes = plt.subplots(1, d, figsize=(5 * d, 4))
    if d == 1:
        axes = [axes]

    for k in range(d):
        ax = axes[k]
        ax.hist(
            data_np[:, k],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="data",
        )
        ax.hist(
            samples_np[:, k],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="kernel model",
        )
        ax.set_xlabel(feature_names[k])
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend(
            loc="upper right",      # or "best"
            fontsize=11,            # smaller text
            frameon=True,
            framealpha=0.85,        # slightly transparent box
            borderpad=0.3,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.5,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, filename))
    plt.close(fig)

