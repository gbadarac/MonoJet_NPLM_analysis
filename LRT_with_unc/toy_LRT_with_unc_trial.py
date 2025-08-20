#!/usr/bin/env python
# coding: utf-8

import os, time, json, argparse, gc, traceback
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from utils_LRT import TAU
from utils_flows import make_flow

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--toys', type=int, required=True, help="Number of toys")
parser.add_argument('-c', '--calibration', type=str, default="True", help="Enable calibration mode (True/False)")
parser.add_argument('--w_path', type=str, help="Path to fitted weights .npy")
parser.add_argument('--w_cov_path', type=str, help="Path to weights covariance .npy")
parser.add_argument('--hit_or_miss_data', type=str, help="Path to hit-or-miss MC samples (required if calibration=True)")
parser.add_argument('--out_dir', type=str, required=True, help="Base output directory")
parser.add_argument('--ensemble_dir', type=str, required=True, help="Directory containing f_i.pth and architecture_config.json")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert string to bool
calibration = args.calibration.lower() == "true"

# ------------------
# Load optional reference weights and covariance
# ------------------
w_init = np.load(args.w_path).astype(np.float32).reshape(-1) if args.w_path else None
w_cov  = np.load(args.w_cov_path).astype(np.float32)         if args.w_cov_path else None
if (w_init is None) ^ (w_cov is None):
    # Half-configured state is dangerous, drop both
    w_init, w_cov = None, None

if w_init is not None:
    w_init = torch.from_numpy(w_init).float()
if w_cov is not None:
    w_cov = torch.from_numpy(w_cov).float()

# ------------------
# Output dir
# ------------------
out_dir = os.path.join(args.out_dir, "calibration" if calibration else "comparison")
os.makedirs(out_dir, exist_ok=True)
print(f"Output directory set to: {out_dir}", flush=True)
print(f"Calibration mode: {calibration}", flush=True)
if w_init is not None and w_cov is not None:
    print(f"Loaded w shape: {tuple(w_init.shape)}, w_cov shape: {tuple(w_cov.shape)}", flush=True)

# ------------------
# Data helpers
# ------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_target_data(n_points, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean_feat1, std_feat1 = -0.5, 0.25
    mean_feat2, std_feat2 = 0.6, 0.4
    feat1 = np.random.normal(mean_feat1, std_feat1, n_points)
    feat2 = np.random.normal(mean_feat2, std_feat2, n_points)
    return np.column_stack((feat1, feat2)).astype(np.float32)

def build_toy_data(i: int, base_pool: np.ndarray, N_mean: int, calibration_flag: bool) -> np.ndarray:
    """
    Returns a float32 array of shape (N_toy, 2).
    Poisson fluctuates the size around N_mean.
    """
    rng = np.random.default_rng(seed=i * 1000003 + 17)
    N_toy = int(rng.poisson(lam=N_mean, size=1)[0])

    if calibration_flag:
        if len(base_pool) >= N_toy:
            idx = rng.choice(len(base_pool), size=N_toy, replace=False)
        else:
            idx = rng.choice(len(base_pool), size=N_toy, replace=True)
        return base_pool[idx].astype(np.float32)
    else:
        return generate_target_data(N_toy, seed=i * 7919 + 23).astype(np.float32)

# ------------------
# Load pools once
# ------------------
N_EVENTS_MEAN = 10000

if calibration:
    if not args.hit_or_miss_data:
        raise ValueError("Calibration mode requires --hit_or_miss_data")
    data_pool = np.load(args.hit_or_miss_data).astype(np.float32)
else:
    data_pool = None

# ------------------
# Ensemble files
# ------------------
f_i_file = os.path.join(args.ensemble_dir, "f_i.pth")
cfg_file = os.path.join(args.ensemble_dir, "architecture_config.json")
if not os.path.isfile(f_i_file):
    raise FileNotFoundError(f"Missing ensemble file: {f_i_file}")
if not os.path.isfile(cfg_file):
    raise FileNotFoundError(f"Missing config file: {cfg_file}")

with open(cfg_file) as f:
    config = json.load(f)

# ------------------
# Per-toy runner
# ------------------
epochs = 100000  # same as your script

def run_one_toy(x_np: np.ndarray, toy_dir: str, toy_seed: int) -> float:
    """
    Runs your full LRT training for one toy and returns T = sum(loglik_num - loglik_den).
    Writes per-toy artifacts under toy_dir.
    """
    os.makedirs(toy_dir, exist_ok=True)

    # Prepare tensor
    x_data = torch.from_numpy(x_np).float()

    # Recompute model_probs for this toy on CPU
    state_dicts_toy = torch.load(f_i_file, map_location="cpu")
    model_probs_list = []
    with torch.no_grad():
        bs = 5000
        for sd in state_dicts_toy:
            flow = make_flow(
                num_layers=config["num_layers"],
                hidden_features=config["hidden_features"],
                num_bins=config["num_bins"],
                num_blocks=config["num_blocks"],
                num_features=config["num_features"]
            )
            flow.load_state_dict(sd)
            flow = flow.to("cpu").float().eval()

            flow_probs = []
            for j in range(0, len(x_data), bs):
                xb = x_data[j:j+bs]
                logp = flow.log_prob(xb)
                flow_probs.append(torch.exp(logp))
            flow_probs_tensor = torch.cat(flow_probs, dim=0).detach()
            model_probs_list.append(flow_probs_tensor)

            del flow, flow_probs, flow_probs_tensor
            gc.collect()

    model_probs_toy = torch.stack(model_probs_list, dim=1).contiguous().float()
    model_probs_toy.requires_grad_(False)
    del model_probs_list, state_dicts_toy
    gc.collect()

    # Build TAU models
    model_den = TAU((None, 2),
        ensemble_probs=model_probs_toy,
        weights_init=w_init,
        weights_cov=w_cov,
        weights_mean=w_init,
        gaussian_center=[], gaussian_coeffs=[], gaussian_sigma=None,
        lambda_regularizer=1e6, train_net=False)

    n_kernels = min(20, len(x_data))
    centers = x_data[:n_kernels].clone()
    coeffs  = torch.ones((n_kernels,), dtype=torch.float32) / max(1, n_kernels)

    model_num = TAU((None, 2),
        ensemble_probs=model_probs_toy,
        weights_init=w_init,
        weights_cov=w_cov,
        weights_mean=w_init,
        gaussian_center=centers, gaussian_coeffs=coeffs, gaussian_sigma=0.3,
        lambda_regularizer=1e6, train_net=True)

    # Train DEN
    opt = torch.optim.Adam(model_den.parameters(), lr=0.001)
    loss_hist_den, epoch_hist_den = [], []
    for e in range(epochs):
        opt.zero_grad()
        loss = model_den.loss(x_data)
        loss.backward()
        opt.step()
        if (e + 1) % 1000 == 0:
            loss_hist_den.append(loss.item())
            epoch_hist_den.append(e + 1)
    fig, ax = plt.subplots()
    ax.plot(epoch_hist_den, loss_hist_den)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
    fig.savefig(os.path.join(toy_dir, "denominator_loss.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    denominator = model_den.loglik(x_data).detach().cpu().numpy()
    np.save(os.path.join(toy_dir, "denominator_loglik.npy"), denominator)

    # Train NUM
    opt = torch.optim.Adam(model_num.parameters(), lr=0.001)
    loss_hist_num, epoch_hist_num = [], []
    for e in range(epochs):
        opt.zero_grad()
        loss = model_num.loss(x_data)
        loss.backward()
        opt.step()
        if (e + 1) % 1000 == 0:
            loss_hist_num.append(loss.item())
            epoch_hist_num.append(e + 1)
    fig, ax = plt.subplots()
    ax.plot(epoch_hist_num, loss_hist_num)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
    fig.savefig(os.path.join(toy_dir, "numerator_loss.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    numerator = model_num.loglik(x_data).detach().cpu().numpy()
    np.save(os.path.join(toy_dir, "numerator_loglik.npy"), numerator)

    np.save(os.path.join(toy_dir, "final_kernel_coeffs.npy"),
            model_num.get_coeffs().detach().cpu().numpy())
    np.save(os.path.join(toy_dir, "final_ensemble_weights.npy"),
            model_num.weights.detach().cpu().numpy())

    # Test statistic
    T = float((numerator - denominator).sum())
    with open(os.path.join(toy_dir, "toy_summary.txt"), "w") as f:
        f.write(f"seed: {toy_seed}\nN: {len(x_np)}\nT: {T}\n")

    # Clean
    del model_den, model_num, model_probs_toy, x_data
    gc.collect()
    return T

# ------------------
# Toy loop
# ------------------
Ntoys = int(args.toys)
base_seed = int(time.time()) & 0xFFFFFFFF
seeds = base_seed + np.arange(Ntoys, dtype=np.int64)

T_values = []
for i in range(Ntoys):
    toy_seed = int(seeds[i])
    set_all_seeds(toy_seed)

    if calibration:
        x_np = build_toy_data(i, data_pool, N_EVENTS_MEAN, calibration_flag=True)
    else:
        x_np = build_toy_data(i, None, N_EVENTS_MEAN, calibration_flag=False)

    toy_dir = os.path.join(out_dir, f"toy_{i:05d}")
    print(f"[Toy {i+1}/{Ntoys}] seed={toy_seed}, N={len(x_np)}, dir={toy_dir}", flush=True)

    try:
        T_i = run_one_toy(x_np, toy_dir, toy_seed)
        T_values.append(T_i)
    except Exception as e:
        print(f"[Toy {i}] error: {e}", flush=True)
        traceback.print_exc()
        T_values.append(np.nan)

# ------------------
# Aggregate results
# ------------------
T_values = np.array(T_values, dtype=np.float64)
np.save(os.path.join(out_dir, "t_values.npy"), T_values)
np.save(os.path.join(out_dir, "toy_seeds.npy"), seeds)

h5_path = os.path.join(out_dir, "tvalues.h5")
with h5py.File(h5_path, "w") as f:
    f.create_dataset("seeds", data=seeds, compression="gzip")
    f.create_dataset("T",     data=T_values, compression="gzip")

print(f"Saved {len(T_values)} toys. T mean={np.nanmean(T_values):.6f}, std={np.nanstd(T_values, ddof=1):.6f}", flush=True)

