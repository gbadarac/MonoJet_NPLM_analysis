#!/usr/bin/env python

import os, sys, glob, h5py, math, time, json, random, yaml, argparse, datetime
from pathlib import Path

from scipy.stats import norm, expon, chi2, uniform, chisquare
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist
import torch
import jax.numpy as jnp
from jax import random as jax_random
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid

from torch.autograd.functional import hessian
from torch.autograd import grad

# -------------------------------------------------------------------
# Make parent directory (Train_Models) importable, then import utilities
# -------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../Train_Models/Sparker_kernels
PROJECT_ROOT = THIS_DIR.parent                      # .../Train_Models
sys.path.insert(0, str(PROJECT_ROOT))

from Sparker_utils.SPARKutils import *
from Sparker_utils.PLOTutils import *
from Sparker_utils.GENutils import *
from Sparker_utils.sampler_mixture import *
from Sparker_utils.losses import *
from Sparker_utils.regularizers import *

from plot_utils import (
    plot_loss,
    plot_centroids_history,
    plot_coeffs_history,
    plot_model_marginals_and_heatmap,
    plot_gt_heatmap,
    plot_kernel_marginals,
)

# -------------------------------------------------------------------
# Args
# -------------------------------------------------------------------
default_seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True,
                    help="Path to .npy file with 2D target data")
parser.add_argument("--outdir", type=str, required=True,
                    help="Base output directory for this trial")
parser.add_argument("--seed", type=int, default=default_seed,
                    help="Random seed / component index (defaults to SLURM_ARRAY_TASK_ID)")
parser.add_argument("--n_models", type=int, default=None,
                    help="Number of models in the SLURM array, used only for naming")
parser.add_argument("--centroids_per_layer", type=str, default=None,
                    help="Comma separated list, e.g. 40,30,20,10,5. Overrides n_layers and kernels_per_layer.")
args = parser.parse_args()

# --------------------------------------------------------------------------------
# Load target dataset (2D: bimodal Gaussian + skew-normal) from disk
# --------------------------------------------------------------------------------

data_file = args.data_path
data_train_tot = np.load(data_file).astype("float32")   # shape (N, 2)
N_train_tot = data_train_tot.shape[0]
print(f"Loaded target data from {data_file} with shape {data_train_tot.shape}")

# Base output directory: like NF "trial_dir"
BASE_OUTPUT_DIRECTORY = args.outdir  # e.g. .../EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail

# --------------------------------------------------------------------------------
# Config and output directory
# --------------------------------------------------------------------------------
def create_config_file(config_table, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as outfile:
            json.dump(config_table, outfile, indent=4)
    return config_path

# Decide per layer centroids
if args.centroids_per_layer is not None:
    number_centroids = [int(x) for x in args.centroids_per_layer.split(",")]
    N_LAYERS = len(number_centroids)
    K_PER_L = number_centroids[0]
else:
    N_LAYERS = args.n_layers
    K_PER_L = args.kernels_per_layer
    number_centroids = [K_PER_L for _ in range(N_LAYERS)]

N_MODELS = args.n_models  # can be None

# Width schedule like Gaia's, only if nlayers = 5 and a list was intended
if N_LAYERS == 5:
    # larger widths 
    width_fin_list = [0.15, 0.10, 0.07, 0.05, 0.035]
    
    # narrower widths 
    #width_fin_list = [0.08, 0.06, 0.045, 0.035, 0.025]
else:
    # Generic schedule that still ends at 0.05
    width_fin_list = np.linspace(0.10, 0.02, N_LAYERS).tolist()[::-1]

config_json = {
    "N": 100000,
    "model": "SparKer",
    "output_directory": None,
    "learning_rate": 0.05,
    "coeffs_reg": "unit1",

    "epochs": [2000 for _ in range(N_LAYERS)],
    "patience": 10,
    "plt_patience": 2000,
    "plot": True,
    "plot_marginals": True,

    "width_init": width_fin_list,
    "width_fin": width_fin_list,

    "t_ini": 0,
    "decay_epochs": 0.5,

    "coeffs_init": [0 for _ in range(N_LAYERS)],
    "coeffs_clip": 10000000,

    "number_centroids": number_centroids,

    "coeffs_reg_lambda": 0,
    "widths_reg_lambda": 0,
    "entropy_reg_lambda": 0,

    "resolution_scale": [0],
    "resolution_const": [0],

    "train_coeffs": True,
    "train_widths": False,
    "train_centroids": True,

    # metadata
    "n_layers": N_LAYERS,
    "kernels_per_layer": K_PER_L,
    "n_models": N_MODELS,
}

# --------------------------
# Build a compact trial name
# --------------------------
d = data_train_tot.shape[1]
total_M = int(np.sum(config_json["number_centroids"]))
n_layers = len(config_json["number_centroids"])
k_per_l = config_json["number_centroids"][0] if n_layers > 0 else 0

n_models_str = f"models{N_MODELS}_" if N_MODELS is not None else ""

trial_name = (
    f"N_{N_train_tot}_dim_{d}_kernels_{config_json['model']}_"
    f"{n_models_str}"
    f"L{n_layers}_K{k_per_l}_M{total_M}_"
    f"Nboot{config_json['N']}_lr{config_json['learning_rate']}_"
    f"clip_{int(config_json['coeffs_clip']):d}_"
    f"no_masking"
)

# Final output directory for this trial:
# .../EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/
#N_100000_dim_2_kernels_Soft-SparKer2_M60_Nboot10000_lr0.01
OUTPUT_DIRECTORY = os.path.join(BASE_OUTPUT_DIRECTORY, trial_name)

config_json["output_directory"] = OUTPUT_DIRECTORY
json_path = create_config_file(config_json, OUTPUT_DIRECTORY)

# --------------------------------------------------------------------------------
# NLL and training loop
# --------------------------------------------------------------------------------

def NLL(pred):
    return -torch.log(pred + 1e-10).mean()

def training_loop(seed, data_train_tot, config_json, json_path):
    # train kernels on different bootstrapped datasets 
    # np.random.seed(seed)
    # print('Random seed:', seed)
    model_seed = int(seed)
    bootstrap_seed = int(seed) + 10000

    print("Seed:", seed)
    print("  model_seed:", model_seed)
    print("  bootstrap_seed:", bootstrap_seed)

    # (optional but good) also seed torch for any torch randomness
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    # CPU or GPU depending on availability
    cuda = torch.cuda.is_available()
    if cuda:
        major, minor = torch.cuda.get_device_capability()
        print(f"Found CUDA device with capability sm_{major}{minor}")
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print("Using device:", DEVICE)

    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)

    plot    = config_json["plot"]
    plot_marginals = config_json.get("plot_marginals", False)
    N_train = config_json["N"]

    # output path
    OUTPUT_PATH = config_json["output_directory"]
    output_folder = os.path.join(OUTPUT_PATH, f"seed{seed:03d}")
    os.makedirs(output_folder, exist_ok=True)

    # generate bootstrapped dataset 
    #indices = np.random.choice(np.arange(len(data_train_tot)), size=N_train, replace=True)
    #feature = torch.from_numpy(data_train_tot[indices]).to(DEVICE)
    # generate bootstrapped dataset
    np.random.seed(bootstrap_seed)
    indices = np.random.choice(np.arange(len(data_train_tot)), size=N_train, replace=True)
    feature = torch.from_numpy(data_train_tot[indices]).to(DEVICE)

    # model definition
    model_type   = config_json["model"]
    patience     = config_json["patience"]
    plt_patience = config_json["plt_patience"]
    total_epochs = np.array(config_json["epochs"])
    coeffs_clip  = config_json["coeffs_clip"]

    train_coeffs    = config_json["train_coeffs"]
    train_widths    = config_json["train_widths"]
    train_centroids = config_json["train_centroids"]

    M = config_json["number_centroids"]  # list of kernels per layer
    d = feature.shape[1]
    print('Problem dimensions:', d)
    len_feature = feature.shape[0]
    lr = config_json["learning_rate"]
    coeffs_regularizer_str = config_json["coeffs_reg"]
    print('Coeffs regularizer:', coeffs_regularizer_str)

    resolution_scale = np.array(config_json["resolution_scale"]).reshape((-1,))
    resolution_const = np.array(config_json["resolution_const"]).reshape((-1,))
    resolution_const = torch.from_numpy(resolution_const).float().to(DEVICE)
    resolution_scale = torch.from_numpy(resolution_scale).float().to(DEVICE)
    print('Resolution constants:', resolution_const)

    # init
    width_ini = config_json["width_init"]
    width_fin = config_json["width_fin"]
    print('Widths:', width_fin)
    t_ini        = config_json["t_ini"]
    decay_epochs = config_json["decay_epochs"]
    n_layers     = len(width_ini)

    #widths_init    = [np.ones((M[i], d)) * width_ini[i] for i in range(n_layers)]
    #coeffs_init    = [(np.random.binomial(p=0.5, n=1, size=(M[i], 1)) * 2 - 1) * config_json["coeffs_init"][i] for i in range(n_layers)]
    #centroids_init = [feature[np.random.randint(len_feature, size=M[i]), :] for i in range(n_layers)]
    
    widths_init = [np.ones((M[i], d)) * width_ini[i] for i in range(n_layers)]

    np.random.seed(model_seed)
    coeffs_init = [
        (np.random.binomial(p=0.5, n=1, size=(M[i], 1)) * 2 - 1) * config_json["coeffs_init"][i]
        for i in range(n_layers)
    ]
    centroids_init = [
        feature[np.random.randint(len_feature, size=M[i]), :]
        for i in range(n_layers)
    ]


    # move everything to DEVICE
    coeffs_init    = [torch.from_numpy(coeffs_init[i]).float().to(DEVICE) for i in range(n_layers)]
    widths_init    = [torch.from_numpy(widths_init[i]).float().to(DEVICE) for i in range(n_layers)]
    centroids_init = [centroids_init[i].to(DEVICE) for i in range(n_layers)]

    lam_coeffs  = config_json["coeffs_reg_lambda"]
    lam_widths  = config_json["widths_reg_lambda"]
    lam_entropy = config_json["entropy_reg_lambda"]

    if coeffs_regularizer_str == "L2":
        coeffs_regularizer = L2Regularizer
    elif coeffs_regularizer_str == "L1":
        coeffs_regularizer = L1Regularizer
    elif coeffs_regularizer_str in ["unit1", "unit2"]:
        # both 'unit1' and 'unit2' use the same implementation you have: UnitSqRegularizer
        coeffs_regularizer = UnitSqRegularizer
    elif coeffs_regularizer_str == "":
        lam_coeffs = 0
        coeffs_regularizer = None
    else:
        print('Wrong coeffs_regularizer argument from config. '
            'Choose between "L1", "L2", "unit1", "unit2" or "".')
        coeffs_regularizer = None

    model = Hierarchical(
        input_shape=(None, 1),
        centroids_list=centroids_init,
        widths_list=widths_init,
        coeffs_list=coeffs_init,
        resolution_const=resolution_const,
        resolution_scale=resolution_scale,
        coeffs_clip=coeffs_clip,
        train_widths=train_widths,
        train_coeffs=train_coeffs,
        train_centroids=train_centroids,
        positive_coeffs=False,
        probability_coeffs=False,
        model=model_type,
    ).to(DEVICE)

    # initial loss
    pred = model.call_j(feature, j=0)
    print('Initial loss:')
    loss_value = 0
    nplm_loss_value = NLL(pred)
    print("nplm ", nplm_loss_value.detach().cpu().numpy())
    if lam_coeffs and coeffs_regularizer is not None:
        print("coeff", coeffs_regularizer(model.get_coeffs()).detach().cpu().numpy())
    if lam_entropy:
        print("entropy", CentroidsEntropyRegularizer(model.get_centroids_entropy()).detach().cpu().numpy())

    loss_value += nplm_loss_value
    if lam_coeffs and coeffs_regularizer is not None:
        loss_value += lam_coeffs * coeffs_regularizer(model.get_coeffs())
    if lam_widths:
        loss_value += lam_widths * widths_regularizer(model.get_widths())
    if lam_entropy:
        loss_value += lam_entropy * CentroidsEntropyRegularizer(model.get_centroids_entropy())

    # history
    max_monitor = int(np.sum(1 + total_epochs / patience))
    widths_history    = np.zeros((max_monitor, np.sum(M), d))
    centroids_history = np.zeros((max_monitor, np.sum(M), d))
    coeffs_history    = np.zeros((max_monitor, np.sum(M)))
    loss_history      = np.zeros(max_monitor)
    epochs_history    = np.zeros(max_monitor)

    widths_history[0, :, :]  = model.get_widths().detach().cpu().numpy()
    centroids_history[0, :, :] = model.get_centroids().detach().cpu().numpy()
    coeffs_history[0, :]     = model.get_coeffs().detach().cpu().numpy().reshape((np.sum(M)))
    loss_history[0]          = loss_value.detach().cpu().numpy()
    epochs_history[0]        = 0

    # training
    t1 = time.time()
    ttmp = time.time()
    monitor_idx = 1

    for n in range(n_layers):
        print("layer:", n, ', time since last layer:', time.time() - ttmp)
        ttmp = time.time()
        parameters = []

        if train_coeffs:
            for m in range(n_layers):
                if m <= n:
                    parameters.append(model.get_coeffs_j(j=m))
        if train_widths:
            for m in range(n_layers):
                if m <= n:
                    parameters.append(model.get_widths_j(j=m))
        if train_centroids:
            for m in range(n_layers):
                if m <= n:
                    parameters.append(model.get_centroids_j(j=m))

        optimizer = torch.optim.Adam(parameters, lr=lr)

        for i in range(int(total_epochs[n])):
            if i > t_ini:
                model.set_width_j(
                    Annealing_Linear(
                        t=i - t_ini,
                        ini=width_ini[n],
                        fin=width_fin[n],
                        t_fin=int(decay_epochs * total_epochs[n])
                    ),
                    j=n
                )

            optimizer.zero_grad()
            nplm_loss_value = NLL(model.call_cumsum_j(feature, j=n))
            loss_value = nplm_loss_value

            if lam_coeffs and coeffs_regularizer is not None:
                loss_value = loss_value + lam_coeffs * coeffs_regularizer(model.get_coeffs_j(j=n))
            if lam_widths:
                loss_value = loss_value + lam_widths * widths_regularizer(model.get_widths_j(j=n))
            if lam_entropy:
                loss_value = loss_value + lam_entropy * CentroidsEntropyRegularizer(model.get_centroids_entropy())

            loss_value.backward()
            optimizer.step()
            model.clip_coeffs()

            if not (i % patience):
                widths_history[monitor_idx, :, :]    = model.get_widths().detach().cpu().numpy()
                centroids_history[monitor_idx, :, :] = model.get_centroids().detach().cpu().numpy()
                coeffs_history[monitor_idx, :]       = model.get_coeffs().detach().cpu().numpy().reshape((np.sum(M)))
                loss_history[monitor_idx]            = loss_value.detach().cpu().numpy()
                epochs_history[monitor_idx]          = monitor_idx
                monitor_idx += 1
                print('epoch: %i, NLL loss: %f, COEFFS: %f' %
                      (int(i + 1), nplm_loss_value, loss_value - nplm_loss_value))

            if not plot:
                continue
            if ((i % plt_patience) or (i == 0)) and (i != (total_epochs[n] - 1)):
                continue

            # --------- plots (delegated to plot_utils) ---------
            total_M = int(np.sum(M))
            
            plot_loss(epochs_history, loss_history, monitor_idx, output_folder)
            plot_centroids_history(epochs_history, centroids_history,
                                   monitor_idx, d, total_M, output_folder)
            plot_coeffs_history(epochs_history, coeffs_history,
                                monitor_idx, total_M, output_folder)
            #plot_model_marginals_and_heatmap(model, n, output_folder)
            

    # GT heatmap
    #plot_gt_heatmap(feature, output_folder)

    t2 = time.time()
    print('End training')
    print('Execution time: ', t2 - t1)

    pred = model.call(feature)[-1, :]
    nplm_loss_final = NLL(pred)

    # save test statistic
    t_value = float((-2 * nplm_loss_final).detach().cpu().item())
    with open(os.path.join(output_folder, 't.txt'), 'w') as t_file:
        t_file.write("%f\n" % t_value)
    print('NPLM test at the end of training: ', "%f" % t_value)

    # save exec time
    with open(os.path.join(output_folder, 'time.txt'), 'w') as t_file:
        t_file.write("%f\n" % (t2 - t1))

    if plot_marginals:
        feature_names = ["Feature 1", "Feature 2"]
        plot_kernel_marginals(
            model=model,
            x_data=feature,                 # bootstrapped data used for training
            feature_names=feature_names,
            output_folder=output_folder,
            num_samples=20000,              # adjust as you like
            filename="marginals_kernel.png"
        )

    # save monitoring metrics
    np.save(os.path.join(output_folder, 'loss_history.npy'),      loss_history)
    np.save(os.path.join(output_folder, 'centroids_history.npy'), centroids_history)
    np.save(os.path.join(output_folder, 'widths_history.npy'),    widths_history)
    np.save(os.path.join(output_folder, 'coeffs_history.npy'),    coeffs_history)

    print('Done')
    return model, feature

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running training for seed = {args.seed}")
    training_loop(args.seed, data_train_tot, config_json, json_path)


