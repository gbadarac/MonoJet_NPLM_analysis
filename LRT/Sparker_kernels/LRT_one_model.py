import glob, math, time, os, json, argparse, datetime, sys
from pathlib import Path
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Make Sparker_utils importable
# -------------------------------------------------------------------
THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SPARKER_UTILS = REPO_ROOT / "Train_Ensembles" / "Train_Models" / "Sparker_utils"
sys.path.insert(0, str(SPARKER_UTILS))

import LRTGOFutils_v2 as lrt
import GENutils as gen

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True,
                    help="Dir with config.json and seed*/ histories (Train_Ensembles output).")
parser.add_argument('--model_seed', type=int, required=True,
                    help="Index of the single model to use (e.g. 0 -> seed000).")
parser.add_argument('--out_base', type=str, required=True,
                    help="Base output dir (e.g. .../LRT/Sparker_kernels/results).")
parser.add_argument('--seed_format', type=str, default="seed%03d",
                    help="Seed folder format: seed%03d for seed000, seed%01d for seed0.")
parser.add_argument('--target_data', type=str, default=None,
                    help="Target/ground truth data: single .npy file. Required for CALIBRATION=0.")
parser.add_argument('-n', '--ntest', type=int, required=True,
                    help="Number of points in the test data.")
parser.add_argument('-c', '--calibration', type=int, required=True,
                    help="1 = calibration toy (sample from single model), 0 = test on target data.")
parser.add_argument('-s', '--seed', type=int, default=None,
                    help="Toy seed.")
parser.add_argument('--toy_id', type=int, default=None,
                    help="Toy index used for folder/file naming (0-based). Falls back to seed.")
parser.add_argument('--save_arrays', action='store_true',
                    help="If set, also save per-event numerator/denominator/test arrays.")
args = parser.parse_args()

# -------------------------------------------------------------------
# Seed / label
# -------------------------------------------------------------------
seed = args.seed
if seed is None:
    seed = (datetime.datetime.now().microsecond
            + datetime.datetime.now().second
            + datetime.datetime.now().minute)
print('Random seed:', seed)
np.random.seed(seed)

label = args.toy_id if args.toy_id is not None else seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)

# -------------------------------------------------------------------
# Hyperparameters  (match ensemble LRT.py)
# -------------------------------------------------------------------
Ntest                = args.ntest
n_kernels_numerator  = 100
epochs_tau           = 100000
patience             = 1000
kernel_width_numerator = 0.3
lambda_L2_numerator  = 10000
lr_tau               = 1e-6
clip_tau             = 0.0005
train_centers_tau    = False

# -------------------------------------------------------------------
# Output folder
# -------------------------------------------------------------------
mode_tag = "calibration" if args.calibration else "test"

seed_fmt = args.seed_format
run_tag = "SparKer1_%s_Ntest%i_M%i_W%s_L%g" % (
    seed_fmt % args.model_seed,
    Ntest,
    n_kernels_numerator,
    str(kernel_width_numerator),
    lambda_L2_numerator,
)
if clip_tau is not None:
    run_tag += "_clip%s" % str(clip_tau)
# DEN profiles the GMM mixture coefficients freely (analogous to free_wifi_weights in LRT.py)
run_tag += "_free_gmm_weights"

out_dir = os.path.join(args.out_base, run_tag, mode_tag, "seed%i" % label)
os.makedirs(out_dir, exist_ok=True)
print("Writing outputs to:", out_dir, flush=True)

# -------------------------------------------------------------------
# Load single model
# -------------------------------------------------------------------
with open(os.path.join(args.model_dir, "config.json"), "r") as f:
    config_json = json.load(f)

seed_dir = os.path.join(args.model_dir, seed_fmt % args.model_seed)
tmp = np.load(os.path.join(seed_dir, "widths_history.npy"))
count = -1
for j in range(tmp.shape[0]):
    if tmp[j][0].sum():
        count += 1
    else:
        print(count)
        break

centroids    = np.load(os.path.join(seed_dir, "centroids_history.npy"))[count]       # (K, d)
coefficients = np.load(os.path.join(seed_dir, "coeffs_history.npy"))[count]          # (K,)
widths       = np.load(os.path.join(seed_dir, "widths_history.npy"))[count, :, 0]    # (K,)

# Normalize coefficients to a proper probability vector
coefficients = coefficients / coefficients.sum()

print(f"Loaded model {seed_fmt % args.model_seed}: "
      f"centroids {centroids.shape}, coefficients {coefficients.shape}, widths {widths.shape}",
      flush=True)

# -------------------------------------------------------------------
# Helper: sample N points directly from the single GMM
# -------------------------------------------------------------------
def sample_from_gmm(centroids, coefficients, widths, n_samples, rng):
    """
    Sample from a Gaussian mixture model with isotropic components.
    centroids : (K, d)
    coefficients : (K,) normalized probability weights
    widths : (K,) per-component isotropic std
    """
    K, d = centroids.shape
    k_indices = rng.choice(K, size=n_samples, p=coefficients)
    noise = rng.standard_normal((n_samples, d))
    samples = centroids[k_indices] + widths[k_indices, np.newaxis] * noise
    return samples.astype(np.float32)

# -------------------------------------------------------------------
# Load / generate test data
# -------------------------------------------------------------------
if args.calibration:
    rng = np.random.default_rng(seed=seed)
    data_all = sample_from_gmm(centroids, coefficients, widths, Ntest, rng)
    bootstrap_sample = data_all          # already exactly Ntest points
    print(f"Generated {bootstrap_sample.shape[0]} calibration samples from GMM.", flush=True)
else:
    if args.target_data is None:
        raise ValueError("calibration=0 but --target_data not provided.")
    data_all = np.load(args.target_data)
    print(f"Loaded {data_all.shape[0]} target data points.", flush=True)
    idx = np.random.choice(len(data_all), Ntest, replace=False)
    bootstrap_sample = data_all[idx]

# -------------------------------------------------------------------
# Evaluate individual GMM components at test points.
# Each of the K Gaussian components is treated as an "ensemble member"
# with a trainable mixture coefficient — directly analogous to how
# LRT.py profiles WiFi weights for the ensemble.
#
# Layout (mirrors the ensemble convention in LRT.py):
#   model_probs      (N, K-1): components 0..K-2, with free weights w[0..K-2]
#   model_norm_probs (N, 1)  : component K-1,     weight = 1 - sum(w)
#   weights_init     (K-1,)  : trained GMM coefficients (initial values)
#
# At initialisation f(x) = sum_k coeff_k * comp_k(x) = original GMM PDF. ✓
# DEN profiles w freely (no prior) to best fit the data — same logic as
# --free_wifi_weights in LRT.py.
# -------------------------------------------------------------------
N = bootstrap_sample.shape[0]

EPS = 1e-300
comps = gen.evaluate_gaussian_components(bootstrap_sample, centroids, widths)  # (N, K)
comps = np.maximum(comps, EPS)

model_probs      = torch.from_numpy(comps[:, :-1].astype(np.float64))   # (N, K-1)
model_norm_probs = torch.from_numpy(comps[:, -1:].astype(np.float64))   # (N, 1)
weights_init_gmm = torch.from_numpy(coefficients[:-1].astype(np.float64))  # (K-1,)

print("model_probs       shape:", model_probs.shape,
      " min:", float(model_probs.min()),
      " max:", float(model_probs.max()), flush=True)
print("model_norm_probs  shape:", model_norm_probs.shape,
      " min:", float(model_norm_probs.min()),
      " max:", float(model_norm_probs.max()), flush=True)
print("weights_init_gmm  shape:", weights_init_gmm.shape,
      " sum:", float(weights_init_gmm.sum()),
      " w_norm_init:", float(1.0 - weights_init_gmm.sum()), flush=True)

model_probs      = torch.clamp(model_probs, min=EPS)
model_norm_probs = torch.clamp(model_norm_probs, min=EPS)

x_data = torch.from_numpy(bootstrap_sample).double().to(device)

# -------------------------------------------------------------------
# TAU — Denominator
# Profiles the K-1 GMM mixture coefficients freely (no prior),
# analogous to --free_wifi_weights in LRT.py.
# train_net=False: no extra test kernels in the denominator.
# -------------------------------------------------------------------
model_den = lrt.TAU(
    (None, centroids.shape[1]),
    ensemble_probs=model_probs.to(device),
    ensemble_norm_probs=model_norm_probs.to(device),
    weights_init=weights_init_gmm.clone(),
    weights_cov=None,    # free profiling — no Gaussian prior on GMM coefficients
    weights_mean=None,
    gaussian_center=[],
    gaussian_coeffs=[],
    gaussian_sigma=None,
    train_net=False,
    train_weights=True,  # profile GMM mixture coefficients
).to(device)
model_den = model_den.double()

# Sanity check at initialisation (weights = trained coefficients → f = original GMM PDF)
with torch.no_grad():
    den_p0 = model_den.call(x_data)[:, 0]
    den_p0 = torch.clamp(den_p0, min=model_den.eps)
    den0 = torch.log(den_p0).sum()
    print("den0 finite:", torch.isfinite(den0).item(), "den0:", den0.item(), flush=True)
    if not torch.isfinite(den0):
        raise RuntimeError("DEN loglik is not finite at init.")

# Train DEN: profile GMM mixture coefficients on the data
den_epochs, den_losses, _ = lrt.train_loop(
    x_data, model_den, "DEN",
    epochs=epochs_tau,
    lr=lr_tau,
    patience=patience,
)

fig, ax = plt.subplots()
ax.plot(den_epochs, den_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
fig.savefig(os.path.join(out_dir, "seed%i_denominator_loss.png" % label), dpi=180, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------
# TAU — Numerator (single model + extra kernels)
# -------------------------------------------------------------------
centers = x_data[:n_kernels_numerator].clone()
coeffs  = torch.ones(n_kernels_numerator, dtype=torch.float64, device=device) / n_kernels_numerator

model_num = lrt.TAU(
    (None, centroids.shape[1]),
    ensemble_probs=model_probs.to(device),
    ensemble_norm_probs=model_norm_probs.to(device),
    weights_init=weights_init_gmm.clone(),  # same initial GMM coefficients as DEN
    weights_cov=None,    # free profiling — no prior
    weights_mean=None,
    gaussian_center=centers.to(device),
    gaussian_coeffs=coeffs.to(device),
    gaussian_sigma=kernel_width_numerator,
    lambda_net=lambda_L2_numerator,
    train_net=True,
    train_centers=train_centers_tau,
    clip_net_coeffs=clip_tau,
    train_weights=True,  # profile GMM mixture coefficients (same as DEN)
).to(device)
model_num = model_num.double()

num_epochs, num_losses, _ = lrt.train_loop(
    x_data, model_num, "NUM",
    epochs=epochs_tau,
    lr=lr_tau,
    patience=patience,
)

fig, ax = plt.subplots()
ax.plot(num_epochs, num_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
fig.savefig(os.path.join(out_dir, "seed%i_numerator_loss.png" % label), dpi=180, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------
# Compute T = loglik_num - loglik_den   (pure log-LR, no aux terms)
# -------------------------------------------------------------------
with torch.no_grad():
    den_p = model_den.call(x_data)[:, 0]
    den_p = torch.clamp(den_p, min=model_den.eps)
    den_log_data = torch.log(den_p)                         # (N,)

    ens_p, net_out = model_num.call(x_data)
    num_p = ens_p[:, 0] + net_out
    num_p = torch.clamp(num_p, min=model_num.eps)
    num_log_data = torch.log(num_p)                         # (N,)

    T_tensor = num_log_data.sum() - den_log_data.sum()
    test     = num_log_data - den_log_data

    T = float(T_tensor.detach().cpu().item())
    numerator   = num_log_data.detach().cpu().numpy()
    denominator = den_log_data.detach().cpu().numpy()
    test_np     = test.detach().cpu().numpy()

print(f"T = {T:.6f}", flush=True)
print(f"mean per-event log LR = {float(test.mean()):.6f}", flush=True)

# -------------------------------------------------------------------
# Save outputs
# -------------------------------------------------------------------
with open(os.path.join(out_dir, f"seed{label}_T.txt"), "w") as f:
    f.write(f"{T}\n")

np.save(os.path.join(out_dir, f"seed{label}_T.npy"), np.array(T, dtype=np.float64))

if args.save_arrays:
    np.save(os.path.join(out_dir, f"seed{label}_test.npy"),       test_np)
    np.save(os.path.join(out_dir, f"seed{label}_numerator.npy"),  numerator)
    np.save(os.path.join(out_dir, f"seed{label}_denominator.npy"), denominator)

np.save(os.path.join(out_dir, f"seed{label}_coeffs.npy"),
        model_num.network.get_coefficients().detach().cpu().numpy())

# Save profiled GMM mixture weights (analogous to den_weights / num_weights in LRT.py)
np.save(os.path.join(out_dir, f"seed{label}_den_gmm_weights.npy"),
        model_den.weights.detach().cpu().numpy())
np.save(os.path.join(out_dir, f"seed{label}_num_gmm_weights.npy"),
        model_num.weights.detach().cpu().numpy())
np.save(os.path.join(out_dir, f"seed{label}_init_gmm_weights.npy"),
        weights_init_gmm.cpu().numpy())

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
kernel_coeffs_final = model_num.network.get_coefficients().detach().cpu().numpy()
raw_coeffs_final    = model_num.network.coefficients.detach().cpu().numpy()
n_at_clip = int(np.sum(np.abs(raw_coeffs_final) >= clip_tau * 0.999))

w_den_final  = model_den.weights.detach().cpu().numpy()
w_num_final  = model_num.weights.detach().cpu().numpy()
w_init_arr   = weights_init_gmm.cpu().numpy()
delta_den    = w_den_final - w_init_arr
delta_num    = w_num_final - w_init_arr

np.set_printoptions(precision=6, suppress=True, linewidth=120)
print(f"--- GMM mixture weights ({len(w_init_arr)} free components) ---")
print(f"  max |Δw_den| = {np.abs(delta_den).max():.6f}  (mean |Δw| = {np.abs(delta_den).mean():.6f})")
print(f"  max |Δw_num| = {np.abs(delta_num).max():.6f}  (mean |Δw| = {np.abs(delta_num).mean():.6f})")
print("--- Kernel coefficients (mean-centred, %i components) ---" % n_kernels_numerator)
print(f"  final      : {kernel_coeffs_final}")
print(f"  saturated at clip={clip_tau}: {n_at_clip}/{n_kernels_numerator}", flush=True)
