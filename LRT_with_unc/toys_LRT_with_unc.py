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
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

seed = int(args.toys)

import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

# -------- Output dir: <base>/<mode>/toy_<seed> --------
mode = "calibration" if calibration else "comparison"
toy_name = f"toy_{args.toys}"  # if you rename to --seed, change to args.seed
out_dir = os.path.join(args.out_dir, mode, toy_name)
os.makedirs(out_dir, exist_ok=True)
print(f"Output directory set to: {out_dir}", flush=True)

N_events = 100000

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
        full = np.load(args.hit_or_miss_data).astype(np.float32)

        # Per-toy deterministic RNG → different subset for each toy id
        rng = np.random.default_rng(seed=seed)

        # Sample without replacement when possible; fall back to with replacement
        if full.shape[0] >= N_events:
            idx = rng.choice(full.shape[0], size=N_events, replace=False)
        else:
            idx = rng.choice(full.shape[0], size=N_events, replace=True)

        data = full[idx].copy()   # copy to avoid any unexpected views
    else:
        raise ValueError("Calibration mode requires --hit_or_miss_data")
else:
    data = generate_target_data(N_events, seed=args.toys)


if w_init is not None and w_cov is not None:
    print(f"Loaded w shape: {w_init.shape}, w_cov shape: {w_cov.shape}")
print(f"Calibration mode: {calibration}")
print(f"Data shape: {data.shape}")

# move data to device once
x_data = torch.from_numpy(data).float().to(device, non_blocking=True)
# -------- Load ensemble --------
f_i_file = os.path.join(args.ensemble_dir, "f_i.pth")
cfg_file = os.path.join(args.ensemble_dir, "architecture_config.json")

# Load on CPU directly
f_i_statedicts = torch.load(f_i_file, map_location="cpu")
with open(cfg_file) as f:
    config = json.load(f)

# ------------------ Evaluate f_i(x) -> model_probs (float32) ------------------
model_probs_list = []
batch_size = 5000

for state_dict in f_i_statedicts:
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_bins=config["num_bins"],
        num_blocks=config["num_blocks"],
        num_features=config["num_features"]
    )
    flow.load_state_dict(state_dict)
    flow = flow.to(device).float().eval()  # run on GPU if available

    vals = []
    with torch.no_grad():
        for i in range(0, x_data.shape[0], batch_size):
            x_batch = x_data[i:i+batch_size].to(device, non_blocking=True)
            lp = flow.log_prob(x_batch)
            vals.append(torch.exp(lp).detach().to("cpu"))  # collect on CPU

    flow_probs_tensor = torch.cat(vals, dim=0)   # CPU tensor
    model_probs_list.append(flow_probs_tensor)

    # Cleanup
    del vals, flow
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

# Stack on CPU, then a single transfer to device
model_probs = torch.stack(model_probs_list, dim=1).contiguous().float()
del model_probs_list; gc.collect()
model_probs = model_probs.to(device, non_blocking=True)

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
epochs = 50000
patience = 1000
tol = 1e-5
max_time_min = 55
t0 = time.time()

# Make sure w_init/w_cov are on device if present
w_init_dev = w_init.to(device) if isinstance(w_init, torch.Tensor) else w_init
w_cov_dev  = w_cov.to(device)  if isinstance(w_cov,  torch.Tensor)  else w_cov

# Denominator: no extra kernels
model_den = TAU(
    (None, 2),
    ensemble_probs=model_probs,
    weights_init=w_init_dev,
    weights_cov=w_cov_dev,
    weights_mean=w_init_dev,
    gaussian_center=[],
    gaussian_coeffs=[],
    gaussian_sigma=None,
    lambda_regularizer=1e6,
    train_net=False).to(device)

# Numerator: pass ensemble_probs=model_probs (NOT None)
n_kernels = 20
centers = x_data[:n_kernels].clone()  # already on device
coeffs  = torch.ones((n_kernels,), dtype=torch.float32, device=device) / n_kernels

model_num = TAU(
    (None, 2),
    ensemble_probs=model_probs,
    weights_init=w_init_dev,
    weights_cov=w_cov_dev,
    weights_mean=w_init_dev,
    gaussian_center=centers,
    gaussian_coeffs=coeffs,
    gaussian_sigma=0.05,
    lambda_regularizer=1e6,
    lambda_net=0,
    train_net=True).to(device)

# ------------------ Train Function ------------------
def train_loop(model, name, lr=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist, epoch_hist = [], []
    best = float("inf")
    bad = 0

    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x_data)   # runs on device
        loss.backward()
        opt.step()

        # log + early stop checks every 100 iters
        if epoch % 100 == 0:
            cur = float(loss.detach().item())
            print(f"[{name}] epoch {epoch} loss {cur:.6f}", flush=True)
            loss_hist.append(cur)
            epoch_hist.append(epoch)

            if cur + tol < best:
                best = cur
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"[{name}] early stopping at epoch {epoch} best {best:.6f}", flush=True)
                    break

        # time budget: stop cleanly before SLURM wall time
        if (time.time() - t0) / 60.0 > max_time_min:
            print(f"[{name}] stopping due to time budget at epoch {epoch}", flush=True)
            break

    return np.array(epoch_hist, np.int32), np.array(loss_hist, np.float32)

model_den.ensemble_probs = model_probs
model_num.ensemble_probs = model_probs

den_epochs, den_losses = train_loop(model_den, "DEN")

# Save loss curve
fig, ax = plt.subplots()
ax.plot(den_epochs, den_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
fig.savefig(os.path.join(out_dir, "denominator_loss.png"), dpi=180, bbox_inches="tight")
plt.close(fig)

# Save log-likelihood values
denominator = model_den.loglik(x_data).detach().cpu().numpy()

num_epochs, num_losses = train_loop(model_num, "NUM")

# Save loss curve (linear y-scale)
fig, ax = plt.subplots()
ax.plot(num_epochs, num_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
fig.savefig(os.path.join(out_dir, "numerator_loss.png"), dpi=180, bbox_inches="tight")
plt.close(fig)

# Save outputs and fitted params
numerator = model_num.loglik(x_data).detach().cpu().numpy()

test = numerator - denominator
print('test: ', test)

out_json = {
    "num": numerator.tolist(),
    "den": denominator.tolist(),
    "test": test.tolist(),
}
with open(os.path.join(out_dir, "lrt_outputs.json"), "w") as f:
    json.dump(out_json, f, indent=2)

def data_loglik(model, x):
        if model.train_net:
            ensemble, net_out = model.call(x)
            p = torch.clamp(ensemble[:, 0] + net_out, min=1e-12)   # positivity for eval
        else:
            p = torch.clamp(model.call(x).squeeze(-1), min=1e-12)
        return torch.log(p).sum()

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
        dev = layer.centers.device  # use the layer's device
        zt = torch.from_numpy(z_np).float().to(dev, non_blocking=True)
        out = layer(zt)
        return out.detach().cpu().numpy().astype(np.float32)

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
    '''
    y_ens_num = _evaluate_mixture_pdf_on_grid(sd, config, model_num.weights.detach(), z, batch=8192)
    y_krl     = _evaluate_krl_on_grid(model_num.network, z)
    coeffs    = torch.softmax(model_num.get_coeffs().detach().cpu(), dim=0).numpy().astype(np.float32)  # (2,)
    y_num     = coeffs[0] * y_ens_num + coeffs[1] * y_krl
    '''

    y_ens_num = _evaluate_mixture_pdf_on_grid(sd, config, model_num.weights.detach(), z, batch=8192)
    y_krl     = _evaluate_krl_on_grid(model_num.network, z)
    # somma diretta: niente c0,c1
    y_num     = y_ens_num + y_krl

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

# =========================
# Binned conditional line-slice overlays (same labels)
# =========================

from math import sqrt, pi

'''
def sample_mixture_of_flows(state_dicts, config, weights_tensor, n_samples, device):
    """Sample from mixture of flows with softmax-normalized weights."""
    n_samples = int(n_samples)  # <- ensure plain int
    
    w = torch.softmax(weights_tensor.detach().cpu().float(), dim=0).numpy()
    

    counts = np.random.multinomial(n_samples, w)
    out = []
    for sd, n_i in zip(state_dicts, counts):
        n_i = int(n_i)          # <- ensure plain int
        if n_i <= 0:
            continue
        f = make_flow(
            num_layers=config["num_layers"],
            hidden_features=config["hidden_features"],
            num_bins=config["num_bins"],
            num_blocks=config["num_blocks"],
            num_features=config["num_features"],
        )
        f.load_state_dict(sd)
        f = f.to(device).float().eval()
        with torch.no_grad():
            try:
                s = f.sample(n_i)          # (n_i, d)
            except TypeError:
                s = f.sample((n_i,))       # (n_i, d)
        out.append(s.detach().cpu().numpy())
        del f
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return np.concatenate(out, axis=0).astype(np.float32) if out else np.empty((0, config["num_features"]), np.float32)
'''
def sample_mixture_of_flows(state_dicts, config, weights_tensor, n_samples, device):
    """Sample from a signed-weight mixture of flows.
    Usa quote di campionamento ∝ |w_i|, ma ritorna anche pesi per-campione = sign(w_i).
    Ritorna: samples: (N, d) float32, sample_weights: (N,) float32 (±1 o 0 se nessun campione)
    """
    n_samples = int(n_samples)
    w_raw = weights_tensor.detach().cpu().float().numpy()  # pesi grezzi, anche negativi
    abs_sum = np.sum(np.abs(w_raw))
    if n_samples <= 0 or len(state_dicts) == 0:
        return np.empty((0, config["num_features"]), np.float32), np.empty((0,), np.float32)

    # quote di campionamento (non-negative, somma 1); se tutti zero, usa uniforme
    if abs_sum > 0:
        probs = np.abs(w_raw) / abs_sum
    else:
        probs = np.ones_like(w_raw) / len(w_raw)

    counts = np.random.multinomial(n_samples, probs)
    out_samples, out_weights = [], []

    for sd, n_i, w_i in zip(state_dicts, counts, w_raw):
        n_i = int(n_i)
        if n_i <= 0:
            continue
        f = make_flow(
            num_layers=config["num_layers"],
            hidden_features=config["hidden_features"],
            num_bins=config["num_bins"],
            num_blocks=config["num_blocks"],
            num_features=config["num_features"],
        )
        f.load_state_dict(sd)
        f = f.to(device).float().eval()
        with torch.no_grad():
            try:
                s = f.sample(n_i)       # (n_i, d)
            except TypeError:
                s = f.sample((n_i,))    # (n_i, d)
        s = s.detach().cpu().numpy().astype(np.float32)
        out_samples.append(s)
        out_weights.append(np.full((n_i,), np.sign(w_i), dtype=np.float32))
        del f
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if not out_samples:
        return np.empty((0, config["num_features"]), np.float32), np.empty((0,), np.float32)

    return np.concatenate(out_samples, axis=0), np.concatenate(out_weights, axis=0)

'''
def sample_kernel_mixture(layer, n_samples):
    """Sample from GaussianKernelLayer: N(center_i, sigma^2 I) with softmax coeffs."""
    n_samples = int(n_samples)  # <- ensure plain int
    if (layer is None) or (layer.centers.numel() == 0) or n_samples == 0:
        return np.empty((0, 2), dtype=np.float32)
    with torch.no_grad():
        centers = layer.centers.detach().cpu().numpy()                         # (K, d)
        coeffs  = torch.softmax(layer.coefficients.detach().cpu(), 0).numpy()  # (K,)
        sigma   = float(layer.sigma.detach().cpu())
    K, d = centers.shape
    counts = np.random.multinomial(n_samples, coeffs)
    pieces = []
    for c_i, n_i in zip(centers, counts):
        n_i = int(n_i)   # <- ensure plain int
        if n_i <= 0:
            continue
        pieces.append(np.random.normal(loc=c_i, scale=sigma, size=(n_i, d)).astype(np.float32))
    return np.concatenate(pieces, axis=0).astype(np.float32) if pieces else np.empty((0, d), np.float32)
'''
def sample_kernel_mixture(layer, n_samples):
    """Sample from GaussianKernelLayer con pesi firmati.
    Quote ∝ |a_j| per assegnare il numero di campioni; per-campione peso = sign(a_j).
    Ritorna: samples: (N, d) float32, sample_weights: (N,) float32
    """
    n_samples = int(n_samples)
    if (layer is None) or (layer.centers.numel() == 0) or n_samples == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    with torch.no_grad():
        centers = layer.centers.detach().cpu().numpy()            # (K, d)
        a_raw   = layer.coefficients.detach().cpu().numpy()       # (K,)
        sigma   = float(layer.sigma.detach().cpu())
    K, d = centers.shape

    abs_sum = np.sum(np.abs(a_raw))
    if abs_sum > 0:
        probs = np.abs(a_raw) / abs_sum
    else:
        probs = np.ones_like(a_raw) / len(a_raw)

    counts = np.random.multinomial(n_samples, probs)
    pieces, wpieces = [], []
    for c_i, n_i, a_i in zip(centers, counts, a_raw):
        n_i = int(n_i)
        if n_i <= 0:
            continue
        samples_i = np.random.normal(loc=c_i, scale=sigma, size=(n_i, d)).astype(np.float32)
        pieces.append(samples_i)
        wpieces.append(np.full((n_i,), np.sign(a_i), dtype=np.float32))

    if not pieces:
        return np.empty((0, d), np.float32), np.empty((0,), np.float32)
    return np.concatenate(pieces, axis=0), np.concatenate(wpieces, axis=0)



def _analytic_conditional_curve(axis, fixed, bins):
    """Analytic p(x_axis | x_other=fixed) for the Gaussian ground truth (non-calibration)."""
    centers = 0.5*(bins[1:]+bins[:-1])
    mean = np.array([-0.5, 0.6], dtype=np.float32)
    sigs = np.array([0.25, 0.4], dtype=np.float32)
    mu = mean[axis]
    sig = sigs[axis]
    pdf = (1.0/(sqrt(2*pi)*sig))*np.exp(-0.5*((centers-mu)/sig)**2)
    return centers, pdf

def _kde_conditional_curve(samples_1d, bins):
    from scipy.stats import gaussian_kde
    centers = 0.5*(bins[1:]+bins[:-1])
    kde = gaussian_kde(samples_1d)
    return centers, kde(centers)

def plot_line_binned(axis, band=0.03, nbins=60, n_samp=150_000):
    """
    axis: which axis to vary (0 or 1)
    band: half-width of the strip around the fixed coordinate (in data units)
    """
    other = 1 - axis
    # same x-range as your line-slice
    x, z, xmin, xmax, fixed = _build_line(axis, npts=2000)
    bins = np.linspace(xmin, xmax, nbins+1)

    # --- strip selection on real data (this is our "Ground truth" histogram)
    mask_data = np.abs(data[:, other] - fixed) <= band
    data_strip = data[mask_data, axis]

    # --- sample models
    sd_for_sampling = torch.load(f_i_file, map_location="cpu")

    '''
    # Ensemble-only (denominator)
    ens_samples = sample_mixture_of_flows(sd_for_sampling, config, model_den.weights, n_samp, device)
    mask_ens = np.abs(ens_samples[:, other] - fixed) <= band
    ens_strip = ens_samples[mask_ens, axis]
    '''

    ens_samples, ens_w = sample_mixture_of_flows(sd_for_sampling, config, model_den.weights, n_samp, device)
    mask_ens = np.abs(ens_samples[:, other] - fixed) <= band
    ens_strip = ens_samples[mask_ens, axis]
    ens_w_strip = ens_w[mask_ens]

    '''
    # Numerator: mixture of flows + kernels with top-level coeffs
    with torch.no_grad():
        top = torch.softmax(model_num.get_coeffs().detach().cpu(), dim=0).numpy()  # [w_ens, w_krl]
    n_ens = int(np.random.binomial(int(n_samp), float(top[0])))
    n_krl = int(n_samp) - n_ens
    num_ens = sample_mixture_of_flows(sd_for_sampling, config, model_num.weights, n_ens, device) if n_ens>0 else np.empty((0,2),np.float32)
    num_krl = sample_kernel_mixture(model_num.network, n_krl)                                     if n_krl>0 else np.empty((0,2),np.float32)
    num_samples = np.concatenate([num_ens, num_krl], axis=0) if (len(num_ens)+len(num_krl)) else np.empty((0,2),np.float32)
    mask_num = np.abs(num_samples[:, other] - fixed) <= band
    num_strip = num_samples[mask_num, axis]
    '''
    with torch.no_grad():
        sum_w = model_num.weights.detach().cpu().numpy().sum()
        sum_a = model_num.network.get_coefficients().detach().cpu().numpy().sum() if model_num.train_net else 0.0
        # quote al top level ∝ masse assolute
        top_abs = np.abs([sum_w, sum_a])
        tot_abs = top_abs.sum() if top_abs.sum() > 0 else 1.0
        p_ens = float(top_abs[0] / tot_abs)

    n_ens = int(np.random.binomial(int(n_samp), p_ens))
    n_krl = int(n_samp) - n_ens

    num_ens_samp, num_ens_w = (sample_mixture_of_flows(sd_for_sampling, config, model_num.weights, n_ens, device)
                            if n_ens > 0 else (np.empty((0,2),np.float32), np.empty((0,),np.float32)))
    num_krl_samp, num_krl_w = (sample_kernel_mixture(model_num.network, n_krl)
                            if n_krl > 0 else (np.empty((0,2),np.float32), np.empty((0,),np.float32)))

    if len(num_ens_samp) or len(num_krl_samp):
        num_samples = np.concatenate([num_ens_samp, num_krl_samp], axis=0)
        num_weights = np.concatenate([num_ens_w,    num_krl_w],    axis=0)
    else:
        num_samples = np.empty((0,2), np.float32)
        num_weights = np.empty((0,),  np.float32)

    mask_num = np.abs(num_samples[:, other] - fixed) <= band
    num_strip = num_samples[mask_num, axis]
    num_w_strip = num_weights[mask_num]


    # --- plot
    fig, ax = plt.subplots(figsize=(6, 4.2))

    ''''
    # histograms (density=True makes area = 1 over the plotted range)
    ax.hist(data_strip, bins=bins, density=True, histtype="step", linewidth=1.8, label="Ground truth")
    if len(ens_strip):
        ax.hist(ens_strip,  bins=bins, density=True, histtype="step", linewidth=1.8, label="Ensemble")
    if len(num_strip):
        ax.hist(num_strip,  bins=bins, density=True, histtype="step", linewidth=1.8, label="Ensemble + extra DOF")
    '''
    ax.hist(data_strip, bins=bins, density=True, histtype="step", linewidth=1.8, label="Ground truth")

    if len(ens_strip):
        ax.hist(ens_strip, bins=bins, density=True, histtype="step", linewidth=1.8,
                weights=ens_w_strip, label="Ensemble")

    if len(num_strip):
        ax.hist(num_strip, bins=bins, density=True, histtype="step", linewidth=1.8,
                weights=num_w_strip, label="Ensemble + extra DOF")

    # optional smooth reference curve on top (same label as your line plot)
    if not calibration:
        cx, ref = _analytic_conditional_curve(axis, fixed, bins)
        ax.plot(cx, ref, linewidth=1.2, alpha=0.8)
    else:
        if len(data_strip) > 50:
            cx, ref = _kde_conditional_curve(data_strip, bins)
            ax.plot(cx, ref, linewidth=1.0, alpha=0.6)

    # --- styling to match plot_line
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0.02, y=0.04)
    ax.set_xlabel(f"Axis {axis}", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, handlelength=2.0)

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out_dir, f"line_slice_binned_{axis}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    del sd_for_sampling
    gc.collect()


# make both binned slices to match your existing line plots
plot_line_binned(axis=0, band=0.03, nbins=60, n_samp=150000)
plot_line_binned(axis=1, band=0.03, nbins=60, n_samp=150000)

