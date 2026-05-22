import glob, gc, os, json, argparse, sys, ctypes
from pathlib import Path
import torch
import numpy as np


def _np2t(arr, dtype=torch.float64):
    """numpy → tensor compatible with numpy 2.x + torch 1.x (ctypes fallback)."""
    arr_c = np.ascontiguousarray(arr, dtype=np.float64 if dtype == torch.float64 else np.float32)
    try:
        return torch.from_numpy(arr_c).to(dtype)
    except TypeError:
        t = torch.empty(arr_c.shape, dtype=dtype)
        ctypes.memmove(t.data_ptr(),
                       arr_c.ctypes.data_as(ctypes.c_void_p),
                       arr_c.nbytes)
        return t
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Repo paths
# -------------------------------------------------------------------
THIS_DIR      = Path(__file__).resolve().parent          # .../LRT
REPO_ROOT     = THIS_DIR.parent                           # .../MonoJet_NPLM_analysis
SPARKER_UTILS = REPO_ROOT / "shared" / "Sparker_utils"
NF_UTILS      = REPO_ROOT / "Train_Ensembles" / "Train_Models"

sys.path.insert(0, str(SPARKER_UTILS))
sys.path.insert(0, str(NF_UTILS))

import LRTGOFutils_v2 as lrt
import GENutils as gen

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Model type
parser.add_argument('--model_type', type=str, required=True, choices=['kernels', 'nf'],
                    help="'kernels' = Sparker kernel ensemble; 'nf' = normalizing flow ensemble.")

# Kernels-specific
parser.add_argument('--ensemble_dir', type=str, default=None,
                    help="[kernels] Dir with config.json and seed*/ histories.")
parser.add_argument('--seed_format', type=str, default='seed%03d',
                    help="[kernels] Seed folder format, e.g. seed%03d.")

# NF-specific
parser.add_argument('--fi_path', type=str, default=None,
                    help="[nf] Path to f_i.pth (list of NF state dicts).")
parser.add_argument('--arch_config', type=str, default=None,
                    help="[nf] Path to architecture_config.json.")

# Common WiFi / data
parser.add_argument('--w_path', type=str, required=True,
                    help="Path to fitted WiFi weights .npy.")
parser.add_argument('--w_cov_path', type=str, required=True,
                    help="Path to covariance of fitted WiFi weights .npy.")
parser.add_argument('--out_base', type=str, required=True,
                    help="Base output directory.")
parser.add_argument('--calib_data', type=str, default=None,
                    help="Calibration data: folder of *.npy OR a single .npy file.")
parser.add_argument('--target_data', type=str, default=None,
                    help="Target data: single .npy file.")
parser.add_argument('-e', '--nensemble', type=int, required=True,
                    help="Number of ensemble models (incl. norm model).")
parser.add_argument('-n', '--ntest', type=int, required=True,
                    help="Number of test/calibration events.")
parser.add_argument('-c', '--calibration', type=int, required=True,
                    help="1 = calibration (null) toys; 0 = test on target data.")
parser.add_argument('-s', '--seed', type=int, default=None)
parser.add_argument('--toy_id', type=int, default=None)
parser.add_argument('--save_arrays', action='store_true')
parser.add_argument('--fix_wifi_weights', action='store_true',
                    help="Freeze WiFi weights at central value (frozen mode).")
parser.add_argument('--free_wifi_weights', action='store_true',
                    help="Profile WiFi weights freely with no prior (free mode, C→∞).")

args = parser.parse_args()

train_wifi_weights = not args.fix_wifi_weights
use_prior = train_wifi_weights and not args.free_wifi_weights

seed  = args.seed
print('Random seed:', seed)
label = args.toy_id if args.toy_id is not None else seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

# -------------------------------------------------------------------
# Hyperparameters (shared)
# -------------------------------------------------------------------
Ntest                  = args.ntest
lambda_regularizer     = 0
n_kernels_numerator    = 100
epochs_tau             = 100000
epochs_delta           = 100000
patience               = 1000
kernel_width_numerator = 0.3
lambda_L2_numerator    = 10000
lr_delta               = 1e-6
lr_tau                 = 1e-6
clip_tau               = 0.0005
train_centers_tau      = False

# -------------------------------------------------------------------
# Output folder
# -------------------------------------------------------------------
mode_tag = "calibration" if args.calibration else "test"

prefix = "SparKer" if args.model_type == "kernels" else "NF"
run_tag = "%s%i_Ntest%i_M%i_W%s_L%g" % (
    prefix, args.nensemble, Ntest,
    n_kernels_numerator, str(kernel_width_numerator), lambda_L2_numerator,
)
if clip_tau is not None:
    run_tag += "_clip%s" % str(clip_tau)

wifi_tag = '_'.join(os.path.basename(os.path.dirname(
    os.path.abspath(args.w_cov_path))).split('_')[-2:])
run_tag += "_wifi_%s" % wifi_tag

if not train_wifi_weights:
    run_tag += "_frozen_weights"
elif args.free_wifi_weights:
    run_tag += "_free_weights"

out_dir = os.path.join(args.out_base, run_tag, mode_tag, "seed%i" % label)
os.makedirs(out_dir, exist_ok=True)
print("Writing outputs to:", out_dir, flush=True)

# -------------------------------------------------------------------
# Load WiFi weights (common to both model types)
# -------------------------------------------------------------------
weights_centralv = np.load(args.w_path)       # (M,)
weights_cov_raw  = np.load(args.w_cov_path)   # (M-1,M-1) for kernels or (M,M) for NF

# NF WiFi produces a full (M,M) covariance; drop last row/col to get the
# M-1 free-parameter covariance (last weight = 1 - sum(others)).
if weights_cov_raw.shape[0] == len(weights_centralv):
    weights_cov_init = weights_cov_raw[:-1, :-1]
else:
    weights_cov_init = weights_cov_raw   # already (M-1, M-1)

n_wifi_components = args.nensemble   # M total models (last is norm model)

# -------------------------------------------------------------------
# Load ensemble model (model-type specific)
# -------------------------------------------------------------------
if args.model_type == 'kernels':
    with open(os.path.join(args.ensemble_dir, "config.json"), "r") as f:
        config_json = json.load(f)
    n_kernels_list = config_json["number_centroids"]
    model_type_cfg = config_json["model"]
    seed_fmt = args.seed_format

    centroids_init, coefficients_init, widths_init = [], [], []
    centroids_norm, coefficients_norm, widths_norm = [], [], []

    for i in range(n_wifi_components):
        seed_dir = os.path.join(args.ensemble_dir, seed_fmt % i)
        tmp = np.load(os.path.join(seed_dir, "widths_history.npy"))
        count = -1
        for j in range(tmp.shape[0]):
            if tmp[j][0].sum():
                count += 1
            else:
                print(count)
                break
        centroids_all = np.load(os.path.join(seed_dir, "centroids_history.npy"))[count]
        coeffs_all    = np.load(os.path.join(seed_dir, "coeffs_history.npy"))[count]
        widths_all    = np.load(os.path.join(seed_dir, "widths_history.npy"))[count, :, 0]

        if i < n_wifi_components - 1:
            centroids_init.append(centroids_all)
            coefficients_init.append(coeffs_all)
            widths_init.append(widths_all)
        else:
            centroids_norm.append(centroids_all)
            coefficients_norm.append(coeffs_all)
            widths_norm.append(widths_all)

    centroids_init      = np.stack(centroids_init, axis=0)
    coefficients_init   = np.stack(coefficients_init, axis=0)
    coefficients_init   = coefficients_init / np.sum(coefficients_init, axis=1, keepdims=True)
    widths_init         = np.stack(widths_init, axis=0)
    centroids_norm      = np.stack(centroids_norm, axis=0)
    coefficients_norm   = np.stack(coefficients_norm, axis=0)
    coefficients_norm   = coefficients_norm / np.sum(coefficients_norm, axis=1, keepdims=True)
    widths_norm         = np.stack(widths_norm, axis=0)
    print(centroids_init.shape, coefficients_init.shape, widths_init.shape)

elif args.model_type == 'nf':
    from utils_flows import make_flow
    f_i_statedicts = torch.load(args.fi_path, map_location="cpu")
    with open(args.arch_config) as f:
        arch_config = json.load(f)
    print(f"Loaded {len(f_i_statedicts)} NF state dicts, arch: {arch_config}", flush=True)

# -------------------------------------------------------------------
# Ensemble evaluation helper
# -------------------------------------------------------------------
NF_BATCH = 5000   # events per forward pass for NF evaluation

def _eval_ensemble(data_np):
    """
    Evaluate the M-model ensemble on data_np (N, d) numpy array.
    Returns numpy array (N, M): column i is the density of model i.
    Layout: [:, :-1] = mixture models, [:, -1] = norm model.
    """
    if args.model_type == 'kernels':
        probs = []
        for i in range(n_wifi_components - 1):
            comps = gen.evaluate_gaussian_components(
                data_np, centroids_init[i], widths_init[i])
            probs.append((comps * coefficients_init[i]).sum(axis=1))
        norm_comps = gen.evaluate_gaussian_components(
            data_np, centroids_norm[0], widths_norm[0])
        norm_prob = (norm_comps * coefficients_norm[0]).sum(axis=1)
        return np.column_stack(probs + [norm_prob])   # (N, M)

    elif args.model_type == 'nf':
        per_model = []
        for state_dict in f_i_statedicts:
            flow_kwargs = {k: v for k, v in arch_config.items() if k != 'backend'}
            flow = make_flow(**flow_kwargs)
            flow.load_state_dict(state_dict)
            flow = flow.to(device).float().eval()
            vals = []
            with torch.no_grad():
                for i in range(0, len(data_np), NF_BATCH):
                    xb = _np2t(data_np[i:i + NF_BATCH], dtype=torch.float32).to(device)
                    vals.append(torch.exp(flow.log_prob(xb)).detach().cpu().double())
            per_model.append(torch.cat(vals))
            del flow
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        return torch.stack(per_model, dim=1).numpy().copy()  # (N, M)

# -------------------------------------------------------------------
# Load calibration pool or target data
# -------------------------------------------------------------------
if args.calibration:
    if args.calib_data is None:
        raise ValueError("calibration=1 but --calib_data not provided.")
    if os.path.isdir(args.calib_data):
        files = sorted(glob.glob(os.path.join(args.calib_data, "*.npy")))
        if not files:
            raise FileNotFoundError(f"No .npy files in {args.calib_data}")
        data_all = np.concatenate([np.load(f) for f in files], axis=0)
    else:
        data_all = np.load(args.calib_data)
else:
    if args.target_data is None:
        raise ValueError("calibration=0 but --target_data not provided.")
    data_all = np.load(args.target_data)

print('Pool size:', data_all.shape[0], flush=True)
np.random.seed(seed)

# -------------------------------------------------------------------
# Subsample / SIR
# Constrained calibration (CALIBRATION=1, use_prior=True): posterior-
# predictive null — sample w^toy ~ N(ŵ, C) and SIR-resample the pool.
# This is necessary for the null T distribution on toys to match the
# null T distribution on real data (CALIBRATION=1 only; in CALIBRATION=0
# the LRT is run on real target data, no toy generation involved).
# All other modes: standard random subsampling.
# -------------------------------------------------------------------
if args.calibration and use_prior:
    cov_np   = weights_cov_init            # (M-1, M-1)
    eps_chol = 1e-8 * np.trace(cov_np) / cov_np.shape[0]
    L_chol   = np.linalg.cholesky(cov_np + eps_chol * np.eye(cov_np.shape[0]))
    z        = np.random.randn(len(weights_centralv) - 1)
    w_toy_free = weights_centralv[:-1] + L_chol @ z
    w_toy_norm = 1.0 - w_toy_free.sum()
    print(f"SIR: ||w^toy - w_hat|| = "
          f"{np.linalg.norm(np.append(w_toy_free, w_toy_norm) - weights_centralv):.4f}",
          flush=True)

    # Evaluate ensemble on full pool (expensive but done once)
    print("SIR: evaluating ensemble on full pool...", flush=True)
    pool_all = _eval_ensemble(data_all)          # (N_pool, M)
    pool_mix = pool_all[:, :-1]                  # (N_pool, M-1)
    pool_nrm = pool_all[:, -1]                   # (N_pool,)

    p_toy = pool_mix @ w_toy_free + pool_nrm * w_toy_norm
    p_hat = pool_mix @ weights_centralv[:-1] + pool_nrm * weights_centralv[-1]
    p_toy = np.maximum(p_toy, 1e-300)
    p_hat = np.maximum(p_hat, 1e-300)

    log_w  = np.log(p_toy) - np.log(p_hat)
    log_w -= log_w.max()
    sir_w  = np.exp(log_w)
    sir_w /= sir_w.sum()
    ess    = 1.0 / (sir_w ** 2).sum()
    print(f"SIR: ESS = {ess:.0f} / {len(data_all)}", flush=True)

    sir_idx        = np.random.choice(len(data_all), size=Ntest, replace=True, p=sir_w)
    bootstrap_sample = data_all[sir_idx]

    # Reuse pool evaluation — no second forward pass needed
    model_probs_all = _np2t(pool_all[sir_idx])              # (Ntest, M)
    del pool_all, pool_mix, pool_nrm
    gc.collect()

else:
    # Frozen, free, or CALIBRATION=0: standard random subsampling.
    idx              = np.random.choice(len(data_all), Ntest, replace=False)
    bootstrap_sample = data_all[idx]
    model_probs_all  = _np2t(_eval_ensemble(bootstrap_sample))  # (Ntest, M)

# -------------------------------------------------------------------
# Split into mixture and norm components, clamp, sanity check
# -------------------------------------------------------------------
EPS = 1e-300
model_probs      = torch.clamp(model_probs_all[:, :-1], min=EPS)    # (Ntest, M-1)
model_norm_probs = torch.clamp(model_probs_all[:, -1:], min=EPS)    # (Ntest, 1)

def _stats(name, t):
    t = t.detach().cpu()
    print(name, "shape", tuple(t.shape),
          "finite", torch.isfinite(t).all().item(),
          "min", float(t.min()), "max", float(t.max()),
          "neg", int((t < 0).sum()), "zero", int((t == 0).sum()),
          flush=True)

_stats("model_probs",      model_probs)
_stats("model_norm_probs", model_norm_probs)

x_data = _np2t(bootstrap_sample).to(device)
x_dim  = x_data.shape[1]

# -------------------------------------------------------------------
# WiFi weight initialisation for the optimiser
# -------------------------------------------------------------------
if train_wifi_weights:
    noise_scale = 0.1 * np.abs(weights_cov_init.diagonal()).mean()
    w_init = _np2t(weights_centralv + np.random.normal(
        scale=noise_scale, size=weights_centralv.shape))
else:
    w_init = _np2t(weights_centralv)

w0 = w_init[:-1]
print("sum(w_init) =", float(w0.sum()), "w_norm_init =", float(1.0 - w0.sum()))
p_check = (model_probs.cpu() @ w0) + (model_norm_probs.cpu().squeeze(1) * (1.0 - w0.sum()))
print("p_check finite:", torch.isfinite(p_check).all().item(),
      "min", float(p_check.min()), "num<=0", int((p_check <= 0).sum()), flush=True)

w_centralv = _np2t(weights_centralv)
w_cov      = _np2t(weights_cov_init)

# -------------------------------------------------------------------
# TAU models
# -------------------------------------------------------------------
model_den = lrt.TAU(
    (None, x_dim),
    ensemble_probs=model_probs.to(device),
    ensemble_norm_probs=model_norm_probs.to(device),
    weights_init=w_init[:-1].to(device),
    weights_cov=w_cov.to(device)       if use_prior else None,
    weights_mean=w_centralv[:-1].to(device) if use_prior else None,
    gaussian_center=[],
    gaussian_coeffs=[],
    gaussian_sigma=None,
    lambda_regularizer=lambda_regularizer,
    train_net=False,
    train_weights=train_wifi_weights,
).to(device).double()

with torch.no_grad():
    den_p0 = torch.clamp(model_den.call(x_data)[:, 0], min=model_den.eps)
    den0   = torch.log(den_p0).sum()
    print("den0 finite:", torch.isfinite(den0).item(), "den0:", den0.item(), flush=True)
    if not torch.isfinite(den0):
        raise RuntimeError("DEN loglik not finite at init.")

if train_wifi_weights:
    den_epochs, den_losses, _ = lrt.train_loop(
        x_data, model_den, "DEN", epochs=epochs_delta,
        lr=lr_delta, patience=int(patience))
    fig, ax = plt.subplots()
    ax.plot(den_epochs, den_losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
    fig.savefig(os.path.join(out_dir, f"seed{label}_denominator_loss.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

centers = x_data[:n_kernels_numerator].clone()
coeffs  = torch.ones(n_kernels_numerator, dtype=torch.float64, device=device) / n_kernels_numerator

model_num = lrt.TAU(
    (None, x_dim),
    ensemble_probs=model_probs.to(device),
    ensemble_norm_probs=model_norm_probs.to(device),
    weights_init=w_init[:-1].to(device),
    weights_cov=w_cov.to(device)       if use_prior else None,
    weights_mean=w_centralv[:-1].to(device) if use_prior else None,
    gaussian_center=centers.to(device),
    gaussian_coeffs=coeffs.to(device),
    gaussian_sigma=kernel_width_numerator,
    lambda_regularizer=lambda_regularizer,
    lambda_net=lambda_L2_numerator,
    train_net=True,
    train_centers=train_centers_tau,
    clip_net_coeffs=clip_tau,
    train_weights=train_wifi_weights,
).to(device).double()

num_epochs, num_losses, _ = lrt.train_loop(
    x_data, model_num, "NUM", epochs=epochs_tau,
    lr=lr_tau, patience=patience)

fig, ax = plt.subplots()
ax.plot(num_epochs, num_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
fig.savefig(os.path.join(out_dir, f"seed{label}_numerator_loss.png"),
            dpi=180, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------
# Test statistic T
# -------------------------------------------------------------------
with torch.no_grad():
    N = x_data.shape[0]

    den_p      = torch.clamp(model_den.call(x_data)[:, 0], min=model_den.eps)
    den_log    = torch.log(den_p)

    ens_p, net_out = model_num.call(x_data)
    num_p      = torch.clamp(ens_p[:, 0] + net_out, min=model_num.eps)
    num_log    = torch.log(num_p)

    if train_wifi_weights:
        aux_num  = model_num.log_auxiliary_term()
        aux_den  = model_den.log_auxiliary_term()
        T_tensor = (num_log.sum() + aux_num) - (den_log.sum() + aux_den)
        test     = (num_log - den_log) + ((aux_num - aux_den) / N)
    else:
        T_tensor = num_log.sum() - den_log.sum()
        test     = num_log - den_log

    T        = float(T_tensor.detach().cpu().item())
    numerator   = num_log.detach().cpu().numpy().copy()
    denominator = den_log.detach().cpu().numpy().copy()
    test_np  = test.detach().cpu().numpy().copy()

print(f"T = {T:.6f}")
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
        model_num.network.get_coefficients().detach().cpu().numpy().copy())
np.save(os.path.join(out_dir, f"seed{label}_kernel_centers.npy"),
        centers.detach().cpu().numpy().copy())

w_den_final  = model_den.weights.detach().cpu().numpy().copy()
w_num_final  = model_num.weights.detach().cpu().numpy().copy()
w_init_arr   = w_init[:-1].detach().cpu().numpy().copy()
w_prior_mean = w_centralv[:-1].cpu().numpy().copy()

np.save(os.path.join(out_dir, f"seed{label}_den_weights.npy"),  w_den_final)
np.save(os.path.join(out_dir, f"seed{label}_num_weights.npy"),  w_num_final)
np.save(os.path.join(out_dir, f"seed{label}_init_weights.npy"), w_init_arr)

kernel_coeffs_final = model_num.network.get_coefficients().detach().cpu().numpy().copy()
raw_coeffs_final    = model_num.network.coefficients.detach().cpu().numpy().copy()
n_at_clip = sum(1 for c in raw_coeffs_final.flat if abs(float(c)) >= clip_tau * 0.999)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
M_free = len(w_prior_mean)
print(f"--- WiFi weights ({M_free} free components) ---")
print(f"  prior mean : {w_prior_mean}")
print(f"  init       : {w_init_arr}")
print(f"  DEN final  : {w_den_final}")
print(f"  NUM final  : {w_num_final}")
print(f"--- Kernel coefficients (mean-centred, {n_kernels_numerator} components) ---")
print(f"  final      : {kernel_coeffs_final}")
print(f"  saturated at clip={clip_tau}: {n_at_clip}/{n_kernels_numerator}", flush=True)
