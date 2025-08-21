import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch, traceback
from torchmin import minimize
from torch import nn
from torch.autograd import Variable
import argparse
from utils_LRT import TAU
import os
from utils_flows import make_flow
import gc
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--toys', type=int, required=True, help="Number of toys")
parser.add_argument('-c', "--calibration", type=str, default="True",
                    help="Enable calibration mode (True/False)")
parser.add_argument('--w_path', type=str, help="Path to fitted weights .npy")
parser.add_argument('--w_cov_path', type=str, help="Path to weights covariance .npy")
parser.add_argument('--hit_or_miss_data', type=str,
                    help="Path to hit-or-miss MC samples (needed if calibration=True)")
parser.add_argument('--out_dir', type=str, required=True, help="Base output directory")
parser.add_argument('--ensemble_dir', type=str, required=True, help="Directory containing the ensemble models")
args = parser.parse_args()

# Convert string to bool
calibration = args.calibration.lower() == "true"

# -------- Load optional reference weights (float32) --------
w_init = np.load(args.w_path).astype(np.float32).reshape(-1) if args.w_path else None
w_cov  = np.load(args.w_cov_path).astype(np.float32)         if args.w_cov_path else None
if (w_init is None) ^ (w_cov is None):
    # If only one is given, ignore both to avoid half-configured state
    w_init, w_cov = None, None

if w_init is not None and w_cov is not None:
    w_init = torch.from_numpy(w_init).float()
    w_cov  = torch.from_numpy(w_cov).float()

# -------- Output dir --------
out_dir = os.path.join(args.out_dir, "calibration" if calibration else "comparison")
os.makedirs(out_dir, exist_ok=True)
print(f"Output directory set to: {out_dir}", flush=True)

N_events = 10000

def generate_target_data(n_points, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean_feat1, std_feat1 = -0.5, 0.25
    mean_feat2, std_feat2 = 0.6, 0.4
    feat1 = np.random.normal(mean_feat1, std_feat1, n_points)
    feat2 = np.random.normal(mean_feat2, std_feat2, n_points)
    return np.column_stack((feat1, feat2)).astype(np.float32)

# -------- Ground truth data --------
if calibration:
    if args.hit_or_miss_data:
        data = np.load(args.hit_or_miss_data)[:N_events].astype(np.float32)
    else:
        raise ValueError("Calibration mode requires --hit_or_miss_data")
else:
    data = generate_target_data(N_events, seed=args.toys)

if w_init is not None and w_cov is not None:
    print(f"Loaded w shape: {w_init.shape}, w_cov shape: {w_cov.shape}")
print(f"Calibration mode: {calibration}")
print(f"Data shape: {data.shape}")

# Float32 tensor for both flows and TAU
x_data = torch.from_numpy(data).float()

# -------- Load ensemble --------
f_i_file = os.path.join(args.ensemble_dir, "f_i.pth")
cfg_file = os.path.join(args.ensemble_dir, "architecture_config.json")

# Load on CPU directly
f_i_statedicts = torch.load(f_i_file, map_location="cpu")
with open(cfg_file) as f:
    config = json.load(f)

# ------------------ Evaluate f_i(x) -> model_probs (float32) ------------------
model_probs_list = []

for state_dict in f_i_statedicts:
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_bins=config["num_bins"],
        num_blocks=config["num_blocks"],
        num_features=config["num_features"]
    )

    flow.load_state_dict(state_dict)
    flow = flow.to("cpu").float()  # keep model on CPU, float32

    flow.eval()
    batch_size = 5000
    flow_probs = []

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i:i+batch_size]  # float32 on CPU
            logp_batch = flow.log_prob(x_batch)  # float32
            flow_probs.append(torch.exp(logp_batch))

    flow_probs_tensor = torch.cat(flow_probs, dim=0).detach()  # float32
    model_probs_list.append(flow_probs_tensor)

    # Cleanup
    del flow_probs
    del flow
    torch.cuda.empty_cache()
    gc.collect()

model_probs = torch.stack(model_probs_list, dim=1).contiguous().float()
model_probs.requires_grad_(False)
del model_probs_list
gc.collect()

del f_i_statedicts
gc.collect()

# ------------------ Helpers ------------------
def probs(weights, model_probs):
    # weights: (M,), model_probs: (N, M)
    return (model_probs * weights).sum(dim=1)  # (N,)

def aux(weights, weights_0, weights_cov):
    d = torch.distributions.MultivariateNormal(weights_0, covariance_matrix=weights_cov)
    return d.log_prob(weights)

def nll_aux(weights, weights_0, weights_cov):
    p = probs(weights, model_probs)
    if not torch.all(p > 0):
        return weights.sum() * float("inf")
    loss = -torch.log(p + 1e-8).sum() - aux(weights, weights_0, weights_cov).sum()
    return loss

# ------------------ TAU models (float32 everywhere) ------------------
epochs = 100000

# Denominator model (no extra kernels)
model_den = TAU((None, 2),
    ensemble_probs=model_probs,   # float32
    weights_init=w_init,          # float32 or None
    weights_cov=w_cov,            # float32 or None
    weights_mean=w_init,          # float32 or None
    gaussian_center=[], gaussian_coeffs=[], gaussian_sigma=None,
    lambda_regularizer=1e6, train_net=False)

# Numerator model (with kernels)
n_kernels = 100
centers = x_data[:n_kernels].clone()                 # float32
coeffs  = torch.ones((n_kernels,), dtype=torch.float32) / n_kernels

model_num = TAU((None, 2),
    ensemble_probs=model_probs,   # float32
    weights_init=w_init,
    weights_cov=w_cov,
    weights_mean=w_init,
    gaussian_center=centers, gaussian_coeffs=coeffs, gaussian_sigma=0.1,
    lambda_regularizer=1e6, train_net=True)

# ------------------ Train DENOMINATOR ------------------
optimizer = torch.optim.Adam(model_den.parameters(), lr=0.0001)
loss_hist_den, epoch_hist_den = [], []

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = model_den.loss(x_data)     # float32 tensor
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f"[DEN] Epoch {epoch+1}/{epochs}  Loss: {loss.item():.6f}")
        loss_hist_den.append(loss.item())
        epoch_hist_den.append(epoch + 1)

den_losses = np.array(loss_hist_den, dtype=np.float32)
den_epochs = np.array(epoch_hist_den, dtype=np.int32)
np.savez(os.path.join(out_dir, "den_loss_curve.npz"),
         epochs=den_epochs, loss=den_losses)

# Save loss curve
fig, ax = plt.subplots()
ax.plot(epoch_hist_den, loss_hist_den)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
fig.savefig(os.path.join(out_dir, "denominator_loss.png"), dpi=180, bbox_inches="tight")
plt.close(fig)

# Save log-likelihood values
denominator = model_den.loglik(x_data).detach().cpu().numpy()
np.save(os.path.join(out_dir, "denominator.npy"), denominator)

np.save(os.path.join(out_dir, "den_ensemble_weights.npy"),
        model_den.weights.detach().cpu().numpy())


# ------------------ Train NUMERATOR ------------------
optimizer = torch.optim.Adam(model_num.parameters(), lr=0.0001)
loss_hist_num, epoch_hist_num = [], []

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = model_num.loss(x_data)     # float32 tensor
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f"[NUM] Epoch {epoch+1}/{epochs}  Loss: {loss.item():.6f}")
        loss_hist_num.append(loss.item())
        epoch_hist_num.append(epoch + 1)

num_losses = np.array(loss_hist_num, dtype=np.float32)
num_epochs = np.array(epoch_hist_num, dtype=np.int32)
np.savez(os.path.join(out_dir, "num_loss_curve.npz"),
         epochs=num_epochs, loss=num_losses)

# Save loss curve
fig, ax = plt.subplots()
ax.plot(epoch_hist_num, loss_hist_num)
ax.set_yscale("log")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
fig.savefig(os.path.join(out_dir, "numerator_loss.png"), dpi=180, bbox_inches="tight")
plt.close(fig)

# Save outputs and fitted params
numerator = model_num.loglik(x_data).detach().cpu().numpy()
np.save(os.path.join(out_dir, "numerator.npy"), numerator)

np.save(os.path.join(out_dir, "num_kernel_coeffs.npy"),
        model_num.get_coeffs().detach().cpu().numpy())

np.save(os.path.join(out_dir, "num_ensemble_weights.npy"),
        model_num.weights.detach().cpu().numpy())

test = numerator - denominator
print('test: ', test)
np.save(os.path.join(out_dir, "test.npy"), test)

# =========================
# Conditional line-slice plots for each axis
# =========================
from scipy.stats import gaussian_kde, multivariate_normal

def _evaluate_mixture_pdf_on_grid(state_dicts, config, weights_tensor, z_np, batch=8192):
    """Evaluate mixture-of-flows pdf at points z_np (N,2), with softmax-normalized weights."""
    w = torch.softmax(weights_tensor.detach().cpu().float(), dim=0)  # (m,)
    z = torch.from_numpy(z_np).float()
    total = torch.zeros(z.shape[0], dtype=torch.float32)

    for sd, wj in zip(state_dicts, w):
        if wj.item() == 0.0:
            continue
        f = make_flow(
            num_layers=config["num_layers"],
            hidden_features=config["hidden_features"],
            num_bins=config["num_bins"],
            num_blocks=config["num_blocks"],
            num_features=config["num_features"],
        )
        f.load_state_dict(sd)
        f = f.to("cpu").float().eval()
        vals = []
        with torch.no_grad():
            for i in range(0, z.shape[0], batch):
                lp = f.log_prob(z[i:i+batch])
                vals.append(torch.exp(lp))
        pj = torch.cat(vals, dim=0)  # (N,)
        total += float(wj) * pj
        del f
        gc.collect()
    return total.numpy().astype(np.float32)

def _evaluate_krl_on_grid(layer, z_np):
    """Evaluate GaussianKernelLayer pdf at points z_np (N,2)."""
    if (layer is None) or (layer.centers.numel() == 0):
        return np.zeros((z_np.shape[0],), dtype=np.float32)
    with torch.no_grad():
        zt = torch.from_numpy(z_np).float()
        return layer(zt).detach().cpu().numpy().astype(np.float32)

def _evaluate_gt_on_grid(z_np):
    """Ground-truth joint pdf along the line points z_np."""
    if not calibration:
        mean = np.array([-0.5, 0.6], dtype=np.float32)
        cov  = np.diag(np.array([0.25**2, 0.4**2], dtype=np.float32))
        return multivariate_normal(mean=mean, cov=cov).pdf(z_np).astype(np.float32)
    else:
        kde = gaussian_kde(data.T)          # 2D KDE on your data
        return kde(z_np.T).astype(np.float32)

def _build_line(axis, npts=2000):
    """Return x-grid (npts,), z-grid (npts,2) with the other axis fixed at the median."""
    other = 1 - axis
    fixed = np.median(data[:, other]).astype(np.float32)

    # choose range from robust percentiles + small padding
    lo, hi = np.quantile(data[:, axis], 0.005), np.quantile(data[:, axis], 0.995)
    pad = 0.05 * (hi - lo)
    xmin, xmax = float(lo - pad), float(hi + pad)

    x = np.linspace(xmin, xmax, npts, dtype=np.float32)
    if axis == 0:
        z = np.stack([x, np.full_like(x, fixed)], axis=1)
    else:
        z = np.stack([np.full_like(x, fixed), x], axis=1)
    return x, z, xmin, xmax, fixed

def plot_line(axis):
    # grid along the chosen axis
    x, z, xmin, xmax, fixed = _build_line(axis, npts=2000)

    # ground truth (analytic or KDE)
    y_gt = _evaluate_gt_on_grid(z)

    # ensemble with uploaded weights (fallback to denominator weights if none)
    sd = torch.load(f_i_file, map_location="cpu")
    w_for_plot = w_init if (w_init is not None) else model_den.weights.detach()
    y_ens = _evaluate_mixture_pdf_on_grid(sd, config, w_for_plot, z, batch=8192)

    # ensemble + extra DOF (numerator)
    y_ens_num = _evaluate_mixture_pdf_on_grid(sd, config, model_num.weights.detach(), z, batch=8192)
    y_krl     = _evaluate_krl_on_grid(model_num.network, z)
    coeffs    = torch.softmax(model_num.get_coeffs().detach().cpu(), dim=0).numpy().astype(np.float32)  # (2,)
    y_num     = coeffs[0] * y_ens_num + coeffs[1] * y_krl

    # draw
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(x, y_gt,  label="Ground truth",              linewidth=1.8)
    ax.plot(x, y_ens, label="Ensemble",       linewidth=1.8)
    ax.plot(x, y_num, label="Ensemble + extra DOF",      linewidth=1.8)

    ax.set_xlim(xmin, xmax)
    ax.margins(x=0.02, y=0.04)
    ax.set_xlabel(f"Axis {axis}", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, handlelength=2.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out_dir, f"line_slice_{axis}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    del sd
    gc.collect()

# Make both slices: vary axis 0 (fix axis 1), and vary axis 1 (fix axis 0)
plot_line(axis=0)
plot_line(axis=1)
