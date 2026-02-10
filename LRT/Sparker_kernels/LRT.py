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
epochs_delta = 20000
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
    f"_{mode_tag}"
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
            print(count)
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
model_norm_probs = np.stack(model_norm_probs_members, axis=1)

model_probs = torch.from_numpy(model_probs).double()
model_norm_probs = torch.from_numpy(model_norm_probs).double()

x_data = torch.from_numpy(bootstrap_sample).double().to(device)

noise_scale_init = 0.1 * np.abs(weights_cov_init.diagonal()).mean()
w_init = torch.from_numpy(
    weights_centralv + np.random.normal(scale=noise_scale_init, size=weights_centralv.shape)
).double()

w_centralv = torch.from_numpy(weights_centralv).double()
w_cov = torch.from_numpy(weights_cov_init).double()


# ------------------ TAU models (float32 everywhere) ------------------
# Denominator: no extra kernels
model_den = lrt.TAU(
    (None, 1),
    ensemble_probs=model_probs.to(device),
    ensemble_norm_probs=model_norm_probs.to(device),
    weights_init=w_init.to(device),
    weights_cov=w_cov.to(device),
    weights_mean=w_centralv.to(device),
    gaussian_center=[],
    gaussian_coeffs=[],
    gaussian_sigma=None,
    lambda_regularizer=lambda_regularizer,
    train_net=False
).to(device)

model_den = model_den.double()

den_epochs, den_losses = lrt.train_loop(
    x_data, model_den, "DEN",
    epochs=epochs_delta, lr=lr_delta,
    patience=int(patience)
)

# Save loss curve
fig, ax = plt.subplots()
ax.plot(den_epochs, den_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
fig.savefig(os.path.join(out_dir, "seed%i_denominator_loss.png"%(seed)), dpi=180, bbox_inches="tight")
plt.close(fig)
denominator = model_den.loglik(x_data).detach().cpu().numpy()

# Numerator
centers = x_data[:n_kernels_numerator].clone()  # already on device
coeffs  = torch.ones((n_kernels_numerator,), dtype=torch.float64, device=device) / n_kernels_numerator

model_num = lrt.TAU(
    (None, 1),
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
    clip_net_coeffs=clip_tau
).to(device)

model_num = model_num.double()

num_epochs, num_losses = lrt.train_loop(
    x_data, model_num, "NUM",
    epochs=epochs_tau, lr=lr_tau,
    patience=patience
)

# Save loss curve (linear y-scale)
fig, ax = plt.subplots()
ax.plot(num_epochs, num_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
fig.savefig(os.path.join(out_dir, "seed%i_numerator_loss.png"%(seed)), dpi=180, bbox_inches="tight")
plt.close(fig)
numerator = model_num.loglik(x_data).detach().cpu().numpy()

test = numerator - denominator
print('numerator:', numerator,
      #model_num.normalization_constraint_term(),
      #model_num.network.coefficients, model_num.network.coefficients.sum(),
      #model_num.weights, model_num.weights.sum()
)
print('denominator:', denominator,
      #model_den.normalization_constraint_term(),
      #model_den.weights, model_den.weights.sum()
)
print('test: ', test)
# save test statistic                                                             
t_file = open(out_dir + 'seed%i_t.txt' % (seed), 'w')
t_file.write("%f,%f,%f\n" % (test, numerator, denominator))
t_file.close()

np.save(out_dir + 'seed%i_coeffs.npy' % (seed),
        model_num.network.get_coefficients().detach().cpu().numpy())