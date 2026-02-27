import glob, h5py, math, time, os, json, random, yaml, argparse, datetime, sys
from scipy.stats import norm, expon, chi2, uniform, chisquare
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist
from pathlib import Path
import torch

import numpy as np
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches

from torch.autograd.functional import hessian
from torch.autograd import grad

# -------------------------------------------------------------------
# Make Sparker_utils importable (repo structure)
# -------------------------------------------------------------------
THIS_DIR  = Path(__file__).resolve().parent                 # .../LRT/Sparker_kernels
REPO_ROOT = THIS_DIR.parents[1]                              # .../MonoJet_NPLM_analysis
SPARKER_UTILS = REPO_ROOT / "Train_Ensembles" / "Train_Models" / "Sparker_utils"

sys.path.insert(0, str(SPARKER_UTILS))

# IMPORTANT: use the v2 implementation (has the new TAU)
import LRTGOFutils_v2 as lrt

import ENSEMBLEutils as ens
import GENutils as gen

parser   = argparse.ArgumentParser()
parser.add_argument('--ensemble_dir', type=str, required=True,
                    help="Dir with config.json and seed*/ histories (Train_Ensembles output).")
parser.add_argument('--w_path', type=str, required=True,
                    help="Path to fitted WiFi weights .npy (e.g. final_weights.npy).")
parser.add_argument('--w_cov_path', type=str, required=True,
                    help="Path to covariance of fitted weights .npy.")
parser.add_argument('--out_base', type=str, required=True,
                    help="Base output dir (e.g. .../LRT/Sparker_kernels/results/<tag>).")
parser.add_argument('--seed_format', type=str, default="seed%03d",
                    help="Seed folder format: seed%03d for seed000, seed%01d for seed0.")
parser.add_argument('--calib_data', type=str, default=None,
                    help="Calibration data: folder of *.npy OR a single .npy file.")
parser.add_argument('--target_data', type=str, default=None,
                    help="Target/ground truth data: usually a single .npy file (e.g. 500k_2d_gaussian_heavy_tail_target_set.npy).")
parser.add_argument('-e', '--nensemble', type=int, help="number of ensembled models", required=True)
parser.add_argument('-n', '--ntest', type=int, help="number of points in the test data", required=True)
parser.add_argument('-c', '--calibration', type=int, help="is it a calibration toy", required=True)
parser.add_argument('-s', '--seed', type=int, help="toy seed", required=False, default=None)
parser.add_argument('--save_arrays', action='store_true', help="If set, also save per-event numerator/denominator/test arrays.")
args     = parser.parse_args()

# random seed
seed = args.seed
if seed==None:
    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().min
#np.random.seed(seed)
print('Random seed:'+str(seed))

# train on GPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Ntest = args.ntest
ensemble_dir = args.ensemble_dir
w_path = args.w_path
w_cov_path = args.w_cov_path
out_base = args.out_base
seed_fmt = args.seed_format

lambda_regularizer = 0
n_kernels_numerator = 100

epochs_tau   = 200000
epochs_delta = 200000
patience = 1000

kernel_width_numerator = 0.08
lambda_L2_numerator = 0

lr_delta = 1e-7
lr_tau   = 1e-7

clip_tau = 0.1
train_centers_tau = False

test_id_string = 'Ntest%i_Lnorm%s/M%i_W%s_L%s' % (
    Ntest,
    str(lambda_regularizer),
    n_kernels_numerator,
    str(kernel_width_numerator),
    str(lambda_L2_numerator),
)
if clip_tau is not None:
    test_id_string += '_clip%s' % (str(clip_tau))
if train_centers_tau:
    test_id_string += '_train_centers'

# -------------------------
# Output folder naming
# -------------------------
ensemble_tag = Path(ensemble_dir).name  # last folder name of the ensemble path
mode_tag = "calibration" if args.calibration else "test"

run_tag = (
    f"{ensemble_tag}"
    f"_wifi{args.nensemble}"
    f"_Ntest{Ntest}"
    f"_{mode_tag}_weights_history_claude_v2"
)

# Put your existing detailed hyperparam string one level deeper
out_dir = os.path.join(out_base, run_tag, test_id_string, mode_tag)

os.makedirs(out_dir, exist_ok=True)
print("Writing outputs to:", out_dir, flush=True)

# ------------------------

with open(os.path.join(ensemble_dir, "config.json"), "r") as jsonfile:
    config_json = json.load(jsonfile)
n_kernels = config_json["number_centroids"]
n_layers = len(n_kernels)
model_type = config_json["model"]
split_indices = np.cumsum(n_kernels)[:-1]

n_wifi_components =args.nensemble
centroids_init, coefficients_init, widths_init = [], [], []
centroids_norm, coefficients_norm, widths_norm = [], [], []

for i in range(n_wifi_components):
    seed_dir = os.path.join(ensemble_dir, seed_fmt % i)
    tmp = np.load(os.path.join(seed_dir, "widths_history.npy"))
    count=-1
    for j in range(tmp.shape[0]):
        if tmp[j][0].sum(): count+=1
        else: 
            break
    centroids_all_i = np.load(os.path.join(seed_dir, "centroids_history.npy"))[count]
    coeffs_all_i    = np.load(os.path.join(seed_dir, "coeffs_history.npy"))[count]
    widths_all_i    = np.load(os.path.join(seed_dir, "widths_history.npy"))[count, :, 0]
    
    # last entry is the normalisation component
    centroids_norm_i = centroids_all_i[-1:]
    centroids_i      = centroids_all_i[:-1]

    coeffs_norm_i = coeffs_all_i[-1:]
    coeffs_i      = coeffs_all_i[:-1]

    widths_norm_i = widths_all_i[-1:]
    widths_i      = widths_all_i[:-1]

    centroids_init.append(centroids_i)
    coefficients_init.append(coeffs_i)
    widths_init.append(widths_i)

    centroids_norm.append(centroids_norm_i)
    coefficients_norm.append(coeffs_norm_i)
    widths_norm.append(widths_norm_i)

centroids_init  = np.stack(centroids_init, axis=0)
coefficients_init = np.stack(coefficients_init, axis=0)
coefficients_init = coefficients_init/np.sum(coefficients_init, axis=1, keepdims=True)
widths_init = np.stack(widths_init, axis=0)
print(centroids_init.shape, coefficients_init.shape, widths_init.shape)

centroids_norm  = np.stack(centroids_norm, axis=0)
coefficients_norm = np.stack(coefficients_norm, axis=0)
widths_norm = np.stack(widths_norm, axis=0)
print(centroids_norm.shape, coefficients_norm.shape, widths_norm.shape)

weights_centralv = np.load(w_path)
weights_cov_init = np.load(w_cov_path)

if args.calibration:
    # data = samples from your *ensemble* (for calibration toys)
    if args.calib_data is None:
        raise ValueError("calibration=1 but --calib_data not provided.")

    if os.path.isdir(args.calib_data):
        files = sorted(glob.glob(os.path.join(args.calib_data, "*.npy")))
        if len(files) == 0:
            raise FileNotFoundError(f"No .npy files found in calib_data dir: {args.calib_data}")
        data_all = np.concatenate([np.load(f) for f in files], axis=0)
    else:
        # single file
        data_all = np.load(args.calib_data)

else:
    if args.target_data is None:
        raise ValueError("calibration=0 but --target_data not provided.")
    data_all = np.load(args.target_data)

        
print('number of available data points:', data_all.shape[0])
np.random.seed(seed)
idx = np.random.choice(len(data_all), Ntest, replace=False)
bootstrap_sample = data_all[idx]

#evaluate the ensemble at the test points
# evaluate the ensemble at the test points
# IMPORTANT: gen.evaluate_gaussian_components returns per-component pdfs (Ntest, K),
# but TAU expects per-model pdfs (Ntest, nensemble). So we must collapse over K using coefficients.

model_probs_members = []
model_norm_probs_members = []

for i in range(n_wifi_components):
    # components pdfs: (Ntest, K)
    comps_i = gen.evaluate_gaussian_components(
        bootstrap_sample,
        centroids_init[i],
        widths_init[i],
    )
    # collapse components -> per-model pdf: (Ntest,)
    pdf_i = (comps_i * coefficients_init[i]).sum(axis=1)
    model_probs_members.append(pdf_i)

    comps_norm_i = gen.evaluate_gaussian_components(
        bootstrap_sample,
        centroids_norm[i],
        widths_norm[i],
    )
    # collapse norm components too (usually K=1, but keep it general): (Ntest,)
    pdf_norm_i = (comps_norm_i * coefficients_norm[i]).sum(axis=1)
    model_norm_probs_members.append(pdf_norm_i)

# stack across ensemble members -> (Ntest, nensemble)
model_probs = np.stack(model_probs_members, axis=1)
model_norm_probs = np.mean(np.stack(model_norm_probs_members, axis=1), axis=1, keepdims=True)  # (N,1)

model_probs = torch.from_numpy(model_probs).double()
model_norm_probs = torch.from_numpy(model_norm_probs).double()

# ----------------------------------------------------------------------
# Prevent log(0) / division by 0 inside TAU
EPS = 1e-300  # safe floor for float64
model_probs = torch.clamp(model_probs, min=EPS)
model_norm_probs = torch.clamp(model_norm_probs, min=EPS)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Sanity checks for pdf tensors (catch zeros/underflow/NaNs early)
def stats(name, t):
    t = t.detach().cpu()
    tiny = (t < 1e-300).sum().item()     # float64 danger zone
    zero = (t == 0).sum().item()
    neg  = (t < 0).sum().item()
    print(
        name,
        "shape", tuple(t.shape),
        "finite", torch.isfinite(t).all().item(),
        "min", t.min().item(),
        "max", t.max().item(),
        "neg", neg,
        "zero", zero,
        "tiny<1e-300", tiny,
    )

stats("model_probs", model_probs)
stats("model_norm_probs", model_norm_probs)
# ----------------------------------------------------------------------

x_data = torch.from_numpy(bootstrap_sample).double().to(device)

noise_scale_init = 0.1 * np.abs(weights_cov_init.diagonal()).mean()
w_init = torch.from_numpy(
    weights_centralv + np.random.normal(scale=noise_scale_init, size=weights_centralv.shape)
).double()

# ----------------------------------------------------------------------
# DIAGNOSTIC: check the *actual* density used by TAU at init
w0 = w_init.detach().cpu()

print("sum(w_init) =", float(w0.sum()))
print("w_norm_init =", float(1.0 - w0.sum()))

# p_comb = (model_probs @ w) + (model_norm_probs * w_norm)
# model_norm_probs is (N,1), so squeeze -> (N,)
p_comb = (model_probs.detach().cpu() @ w0) + (
    model_norm_probs.detach().cpu().squeeze(1) * (1.0 - w0.sum())
)

print(
    "p_comb finite:", torch.isfinite(p_comb).all().item(),
    "min", float(p_comb.min()),
    "max", float(p_comb.max()),
    "num<=0", int((p_comb <= 0).sum()),
)
# ----------------------------------------------------------------------

w_centralv = torch.from_numpy(weights_centralv).double()
w_cov = torch.from_numpy(weights_cov_init).double()


# ------------------ TAU models (float32 everywhere) ------------------
# Denominator: no extra kernels
model_den = lrt.TAU(
    (None, 2),
    ensemble_probs=model_probs.to(device),
    ensemble_norm_probs=model_norm_probs.to(device),
    weights_init=w_init.to(device),
    weights_cov=w_cov.to(device),
    weights_mean=w_centralv.to(device),
    gaussian_center=[],
    gaussian_coeffs=[],
    gaussian_sigma=None,
    lambda_regularizer=lambda_regularizer,
    train_net=False,
    train_weights=True
).to(device)

model_den = model_den.double()

# ----------------------------------------------------------------------
# Fail fast: check denominator loglik is finite before training
with torch.no_grad():
    den_p0 = model_den.call(x_data)[:, 0]
    den_p0 = torch.clamp(den_p0, min=model_den.eps)
    den0 = torch.log(den_p0).sum()
    print("den0 finite:", torch.isfinite(den0).item(), "den0:", den0.item())
    if not torch.isfinite(den0):
        raise RuntimeError("DEN loglik is not finite at init. Likely log(0)/division by 0 in TAU.")
# ----------------------------------------------------------------------

save_every = 1000

den_epochs, den_losses = lrt.train_loop(
    x_data, model_den, "DEN",
    epochs=epochs_delta, lr=lr_delta,
    patience=int(patience),
    save_every=save_every,
    out_dir=out_dir,
    seed=seed
)

# Save loss curve
fig, ax = plt.subplots()
ax.plot(den_epochs, den_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
fig.savefig(os.path.join(out_dir, "seed%i_denominator_loss.png"%(seed)), dpi=180, bbox_inches="tight")
plt.close(fig)

#denominator = model_den.loglik(x_data).detach().cpu().numpy()

# Numerator
centers = x_data[:n_kernels_numerator].clone()  # already on device
coeffs  = torch.ones((n_kernels_numerator,), dtype=torch.float64, device=device) / n_kernels_numerator

model_num = lrt.TAU(
    (None, 2),
    ensemble_probs=model_probs.to(device),
    ensemble_norm_probs=model_norm_probs.to(device),
    weights_init=w_init.to(device),
    weights_cov=w_cov.to(device),
    weights_mean=w_centralv.to(device),
    gaussian_center=centers.to(device),
    gaussian_coeffs=coeffs.to(device),
    gaussian_sigma=kernel_width_numerator,
    lambda_regularizer=lambda_regularizer,
    lambda_net=lambda_L2_numerator,
    train_net=True,
    train_centers=train_centers_tau,
    clip_net_coeffs=clip_tau,
    train_weights=True
).to(device)

model_num = model_num.double()

num_epochs, num_losses = lrt.train_loop(
    x_data, model_num, "NUM",
    epochs=epochs_tau,
    lr=lr_tau,            # single LR supported by this train_loop
    patience=patience,
    save_every=save_every,
    out_dir=out_dir,
    seed=seed
)

#-----------------------------------------------------------------------
# Diagnostic plots for weight evolution (sum(w), w_norm, max|z|, global sigma, tail prob)
#-----------------------------------------------------------------------

def plot_sumw(tag):
    e = np.load(os.path.join(out_dir, f"seed{seed}_{tag}_weights_epoch_hist.npy"))
    s = np.load(os.path.join(out_dir, f"seed{seed}_{tag}_weights_sum_hist.npy"))
    wn = np.load(os.path.join(out_dir, f"seed{seed}_{tag}_weights_wnorm_hist.npy"))

    # Plot sum(w)
    fig, ax = plt.subplots()
    ax.plot(e, s)
    ax.axhline(1.0, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("sum(weights)")
    ax.set_title(f"{tag}: sum(weights) vs epoch")
    fig.savefig(os.path.join(out_dir, f"seed{seed}_{tag}_sumw_vs_epoch.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot w_norm
    fig, ax = plt.subplots()
    ax.plot(e, wn)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("w_norm = 1 - sum(weights)")
    ax.set_title(f"{tag}: w_norm vs epoch")
    fig.savefig(os.path.join(out_dir, f"seed{seed}_{tag}_wnorm_vs_epoch.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

plot_sumw("DEN")
plot_sumw("NUM")
#-----------------------------------------------------------------------

# Save loss curve (linear y-scale)
fig, ax = plt.subplots()
ax.plot(num_epochs, num_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
fig.savefig(os.path.join(out_dir, "seed%i_numerator_loss.png"%(seed)), dpi=180, bbox_inches="tight")
plt.close(fig)

#numerator = model_num.loglik(x_data).detach().cpu().numpy()

# ----------------------------------------------------------------------
with torch.no_grad():
    N = x_data.shape[0]

    # ---------- data terms ----------
    den_p = model_den.call(x_data)[:, 0]
    den_p = torch.clamp(den_p, min=model_den.eps)
    den_log_data = torch.log(den_p)                     # (N,)

    ens_p, net_out = model_num.call(x_data)             # ens_p: (N,1), net_out: (N,)
    num_p = ens_p[:, 0] + net_out
    num_p = torch.clamp(num_p, min=model_num.eps)
    num_log_data = torch.log(num_p)                     # (N,)

    # ---------- auxiliary terms (global) ----------
    aux_num = model_num.log_auxiliary_term()            # scalar
    aux_den = model_den.log_auxiliary_term()            # scalar

    # Training maximizes MAP = sum log p(x) + log P(w | prior)
    # Consistent LRT: T = [sum num_log_data + aux_num] - [sum den_log_data + aux_den]
    T_tensor = (num_log_data.sum() + aux_num) - (den_log_data.sum() + aux_den)
    T = float(T_tensor.detach().cpu().item())

    # Build per event array whose sum equals T by spreading aux difference evenly
    test = (num_log_data - den_log_data) + ((aux_num - aux_den) / N)

    # Save arrays in numpy
    numerator   = num_log_data.detach().cpu().numpy()
    denominator = den_log_data.detach().cpu().numpy()
    test_np     = test.detach().cpu().numpy()

print(f"T = {T:.6f}")
print(f"mean per-event log LR = {float(test.mean()):.6f}")
# ----------------------------------------------------------------------

# Always save scalar T
with open(os.path.join(out_dir, f"seed{seed}_T.txt"), "w") as f:
    f.write(f"{T}\n")

np.save(os.path.join(out_dir, f"seed{seed}_T.npy"), np.array(T, dtype=np.float64))

if args.save_arrays:
    np.save(os.path.join(out_dir, f"seed{seed}_test.npy"), test_np)
    np.save(os.path.join(out_dir, f"seed{seed}_numerator.npy"), numerator)
    np.save(os.path.join(out_dir, f"seed{seed}_denominator.npy"), denominator)

np.save(os.path.join(out_dir, f"seed{seed}_coeffs.npy"),
        model_num.network.get_coefficients().detach().cpu().numpy())