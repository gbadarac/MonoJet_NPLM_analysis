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


# ---------- model visualisations ----------

def plot_model_marginals_and_heatmap(model, n, output_folder):
    """
    For a given layer index n, plot:
      - x0 / x1 marginals per layer
      - 2D heatmap + centroids for layer n
    """
    colors = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4',
              '#1d91c0', '#225ea8', '#253494', '#081d58'] + ['#081d58'] * 20

    # -------------------------------------------------------
    # Get device & dtype from a model tensor (centroids)
    # -------------------------------------------------------
    ref = model.get_centroids()              # [M, d], tensor on correct device
    device = ref.device
    dtype = ref.dtype

    # Grid (on the *model* device + dtype)
    x0 = torch.arange(-1.5, 0.5, 0.01, device=device, dtype=dtype)
    x1 = torch.arange(-0.5, 4.5, 0.005, device=device, dtype=dtype)
    X0, X1 = torch.meshgrid(x0, x1, indexing="xy")
    grid = torch.stack([X0.flatten(), X1.flatten()], dim=1)  # [N, 2]

    # Also keep CPU copies for plotting axes
    x0_cpu = x0.detach().cpu().numpy()
    x1_cpu = x1.detach().cpu().numpy()
    grid_cpu = grid.detach().cpu().numpy()

    # -------------------------------------------------------
    # Evaluate model on grid (on device), then move to CPU
    # -------------------------------------------------------
    with torch.no_grad():
        out_all = model.call(grid)        # [n_layers, N, 1] on device
        norm = model.get_norm()           # [n_layers] on device
        norm = norm.to(out_all.device)

        # Y for layer n
        Y = (out_all[n, :, 0] / norm[n]).detach().cpu().numpy()

        # All layers normalized: shape [n_layers, N, 1]
        model_on_grid = (out_all / norm.view(-1, 1, 1)).detach().cpu().numpy()

    # -------------------------------------------------------
    # Marginal plots
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ni in range(n + 1):
        axes[0].hist(grid_cpu[:, 0],
                     weights=model_on_grid[ni, :, 0],
                     lw=2,
                     bins=x0_cpu[::10],
                     color=colors[ni],
                     histtype="step",
                     label=f"layer {ni}")
    axes[0].set_xlabel("x_0")
    axes[0].set_ylabel("Model output")

    for ni in range(n + 1):
        axes[1].hist(grid_cpu[:, 1],
                     weights=model_on_grid[ni, :, 0],
                     lw=2,
                     bins=x1_cpu[::10],
                     color=colors[ni],
                     histtype="step",
                     label=f"layer {ni}")
    axes[1].set_xlabel("x_1")

    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title=None,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(os.path.join(output_folder, "marginals.png"))
    plt.close(fig)

    # -------------------------------------------------------
    # 2D heatmap + centroids for layer n
    # -------------------------------------------------------
    fig = plt.figure(figsize=(4, 3))
    plt.scatter(grid_cpu[:, 0], grid_cpu[:, 1], c=Y, edgecolors="none", s=1)

    centr = model.get_centroids().detach().cpu().numpy()
    ampl = model.get_coeffs().detach().cpu().numpy()[:, 0]
    centr = centr[ampl > 0]
    if len(centr):
        plt.scatter(centr[:, 0], centr[:, 1], color="black")

    plt.colorbar()
    plt.xlim(-1.5, 0.5)
    plt.ylim(-0.5, 4.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, f"2Dheatmap_{n}.png"))
    plt.close(fig)

def plot_gt_heatmap(feature, output_folder):
    """2D histogram of the (bootstrapped) training data."""
    fig = plt.figure(figsize=(4, 3))

    # make sure we are on CPU for numpy / matplotlib
    feature_cpu = feature.detach().cpu().numpy()

    x0 = torch.arange(-1.5, 0.5, 0.1).double()
    x1 = torch.arange(-0.5, 4.5, 0.05).double()
    x0_cpu = x0.numpy()
    x1_cpu = x1.numpy()

    plt.hist2d(
        feature_cpu[:, 0],
        feature_cpu[:, 1],
        bins=[x0_cpu, x1_cpu],
        density=True,
    )
    plt.colorbar()
    plt.xlim(-1.5, 0.5)
    plt.ylim(-0.5, 4.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, "2Dheatmap_GT.png"))
    plt.close(fig)

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

    # Sample from the kernel model
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



