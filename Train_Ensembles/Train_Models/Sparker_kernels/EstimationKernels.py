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

import matplotlib as mpl
mpl.use('Agg')   # important: before importing pyplot on the cluster
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches

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
args = parser.parse_args()

# --------------------------------------------------------------------------------
# Load target dataset (2D: bimodal Gaussian + skew-normal) from disk
# --------------------------------------------------------------------------------

data_file = args.data_path
# Optional: analytic mean file (not strictly needed for training)
mu_file   = os.path.join(os.path.dirname(data_file), "mu_2d_gaussian_heavy_tail_target.npy")

data_train_tot = np.load(data_file).astype('float32')   # shape (N, 2)
N_train_tot = data_train_tot.shape[0]
print("Loaded target data:", data_train_tot.shape)

if os.path.exists(mu_file):
    mu_target = np.load(mu_file)
    print("Loaded analytic mu_target:", mu_target)
else:
    mu_target = None
    print("mu_target file not found, continuing without it.")

# Quick kernel-width diagnostics (median pairwise distance)
dist_matrix = cdist(data_train_tot[:1000], data_train_tot[:1000], metric='euclidean')
print("Median pairwise distance:", np.median(dist_matrix))
print("Quantiles:", np.quantile(dist_matrix, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

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

config_json = {
    "N"   : 10000,  # bootstrap size per WiFi component
    "model": 'Soft-SparKer2',
    "output_directory": None,  # filled below
    "learning_rate": 0.01,
    "coeffs_reg": "unit1",
    "epochs": [2000 for _ in range(15)],
    "patience": 500,
    "plt_patience": 2000,
    "plot": True,
    "width_init": [4 for _ in range(15)],
    "width_fin":[0.1 for _ in range(15)],
    "t_ini": 0,
    "decay_epochs": 0.9,
    "coeffs_init": [0 for _ in range(15)],
    "coeffs_clip": 100,
    "number_centroids": [4 for _ in range(15)],
    "coeffs_reg_lambda":  0,
    "widths_reg_lambda":  0,
    "entropy_reg_lambda": 0,
    "resolution_scale": [0],
    "resolution_const": [0],
    "train_coeffs": True,
    "train_widths": False,
    "train_centroids": True,
}

# --------------------------
# Build a compact trial name
# --------------------------
d = data_train_tot.shape[1]
total_M = int(np.sum(config_json["number_centroids"]))

trial_name = (
    f"N_{N_train_tot}_dim_{d}_kernels_"
    f"{config_json['model']}_M{total_M}_"
    f"Nboot{config_json['N']}_lr{config_json['learning_rate']}"
)

# Final output directory for this trial:
# .../EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/
#     N_100000_dim_2_kernels_Soft-SparKer2_M60_Nboot10000_lr0.01
OUTPUT_DIRECTORY = os.path.join(BASE_OUTPUT_DIRECTORY, trial_name)

config_json["output_directory"] = OUTPUT_DIRECTORY
json_path = create_config_file(config_json, OUTPUT_DIRECTORY)


# --------------------------------------------------------------------------------
# NLL and training loop
# --------------------------------------------------------------------------------

def NLL(pred):
    return -torch.log(pred + 1e-10).mean()

def training_loop(seed, data_train_tot, config_json, json_path):
    # random seed
    np.random.seed(seed)
    print('Random seed:', seed)

    # CPU only for now (set cuda=True if you want to use GPUs later)
    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)

    plot    = config_json["plot"]
    N_train = config_json["N"]

    # output path
    OUTPUT_PATH = config_json["output_directory"]
    output_folder = os.path.join(OUTPUT_PATH, f"seed{seed:03d}")
    os.makedirs(output_folder, exist_ok=True)

    # bootstrap data
    indices = np.random.choice(np.arange(len(data_train_tot)), size=N_train, replace=True)
    feature = torch.from_numpy(data_train_tot[indices])

    # model definition
    model_type   = config_json["model"]
    patience     = config_json["patience"]
    plt_patience = config_json["plt_patience"]
    total_epochs = np.array(config_json["epochs"])
    coeffs_clip  = config_json["coeffs_clip"]

    train_coeffs    = config_json["train_coeffs"]
    train_widths    = config_json["train_widths"]
    train_centroids = config_json["train_centroids"]

    M = config_json["number_centroids"]  # list
    d = feature.shape[1]
    print('Problem dimensions:', d)
    len_feature = feature.shape[0]
    lr = config_json["learning_rate"]
    coeffs_regularizer_str = config_json["coeffs_reg"]
    print('Coeffs regularizer:', coeffs_regularizer_str)

    resolution_scale = np.array(config_json["resolution_scale"]).reshape((-1,))
    resolution_const = np.array(config_json["resolution_const"]).reshape((-1,))
    resolution_const = torch.from_numpy(resolution_const).double()
    resolution_scale = torch.from_numpy(resolution_scale).double()
    print('Resolution constants:', resolution_const)

    # init
    width_ini = config_json["width_init"]
    width_fin = np.sort(config_json["width_fin"])[::-1]
    print('Widths:', width_fin)
    t_ini        = config_json["t_ini"]
    decay_epochs = config_json["decay_epochs"]
    n_layers     = len(width_ini)

    widths_init    = [np.ones((M[i], d)) * width_ini[i] for i in range(n_layers)]
    coeffs_init    = [(np.random.binomial(p=0.5, n=1, size=(M[i], 1)) * 2 - 1) * config_json["coeffs_init"][i]
                      for i in range(n_layers)]
    centroids_init = [feature[np.random.randint(len_feature, size=M[i]), :] for i in range(n_layers)]

    coeffs_init    = [torch.from_numpy(coeffs_init[i]).double() for i in range(n_layers)]
    widths_init    = [torch.from_numpy(widths_init[i]).double() for i in range(n_layers)]
    centroids_init = [centroids_init[i] for i in range(n_layers)]

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
    print("nplm ", nplm_loss_value.detach().numpy())
    if lam_coeffs and coeffs_regularizer is not None:
        print("coeff", coeffs_regularizer(model.get_coeffs()).detach().numpy())
    if lam_entropy:
        print("entropy", CentroidsEntropyRegularizer(model.get_centroids_entropy()).detach().numpy())

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

    widths_history[0, :, :]  = model.get_widths().detach().numpy()
    centroids_history[0, :, :] = model.get_centroids().detach().numpy()
    coeffs_history[0, :]     = model.get_coeffs().detach().numpy().reshape((np.sum(M)))
    loss_history[0]          = loss_value.detach().numpy()
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
                widths_history[monitor_idx, :, :]    = model.get_widths().detach().numpy()
                centroids_history[monitor_idx, :, :] = model.get_centroids().detach().numpy()
                coeffs_history[monitor_idx, :]       = model.get_coeffs().detach().numpy().reshape((np.sum(M)))
                loss_history[monitor_idx]            = loss_value.detach().numpy()
                epochs_history[monitor_idx]          = monitor_idx
                monitor_idx += 1
                print('epoch: %i, NLL loss: %f, COEFFS: %f' %
                      (int(i + 1), nplm_loss_value, loss_value - nplm_loss_value))

            if not plot:
                continue
            if ((i % plt_patience) or (i == 0)) and (i != (total_epochs[n] - 1)):
                continue

            # --------- plots ---------
            font = font_manager.FontProperties(family='serif', size=18)

            # loss
            fig = plt.figure(figsize=(9, 6))
            fig.patch.set_facecolor('white')
            ax = fig.add_axes([0.15, 0.1, 0.78, 0.8])
            plt.plot(epochs_history[2:monitor_idx], loss_history[2:monitor_idx], label='loss')
            plt.legend(prop=font, loc='best')
            plt.ylabel('Loss', fontsize=18, fontname='serif')
            plt.xlabel('Epochs', fontsize=18, fontname='serif')
            plt.xticks(fontsize=16, fontname='serif')
            plt.yticks(fontsize=16, fontname='serif')
            plt.grid()
            plt.savefig(os.path.join(output_folder, 'loss.pdf'))
            plt.close(fig)

            # centroids
            for k in range(d):
                fig = plt.figure(figsize=(9, 6))
                fig.patch.set_facecolor('white')
                ax = fig.add_axes([0.15, 0.1, 0.78, 0.8])
                for m in range(np.sum(M)):
                    plt.plot(epochs_history[:monitor_idx],
                             centroids_history[:monitor_idx, m, k:k+1],
                             label='%i' % m)
                plt.ylabel('Centroid loc', fontsize=18, fontname='serif')
                plt.xlabel('Epochs', fontsize=18, fontname='serif')
                plt.xticks(fontsize=16, fontname='serif')
                plt.yticks(fontsize=16, fontname='serif')
                plt.grid()
                plt.savefig(os.path.join(output_folder, 'centroids_dim%i.pdf' % k))
                plt.close(fig)

            # coeffs
            fig = plt.figure(figsize=(9, 6))
            fig.patch.set_facecolor('white')
            ax = fig.add_axes([0.15, 0.1, 0.78, 0.8])
            for m in range(np.sum(M)):
                plt.plot(epochs_history[:monitor_idx], coeffs_history[:monitor_idx, m], label='%i' % m)
            plt.ylabel('Coeffs', fontsize=18, fontname='serif')
            plt.xlabel('Epochs', fontsize=18, fontname='serif')
            plt.xticks(fontsize=16, fontname='serif')
            plt.yticks(fontsize=16, fontname='serif')
            plt.grid()
            plt.savefig(os.path.join(output_folder, 'coeffs.pdf'))
            plt.close(fig)

            # model marginals + centroids heatmap
            with torch.no_grad():
                colors = ['#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4',
                          '#1d91c0','#225ea8','#253494','#081d58'] + ['#081d58' for _ in range(20)]

                x0 = torch.arange(-1.5, 0.5, 0.01).double()
                x1 = torch.arange(-0.5, 4.5, 0.005).double()
                X0, X1 = torch.meshgrid(x0, x1, indexing='xy')
                grid = torch.stack([X0.flatten(), X1.flatten()], dim=1)

                Y = model.call(grid).detach()[n, :, 0] / model.get_norm()[n]
                model_on_grid = model.call(grid).detach().numpy() / model.get_norm()

                fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

                for ni in range(n + 1):
                    axes[0].hist(grid[:, 0].numpy(),
                                 weights=model_on_grid[ni, :, 0],
                                 lw=2,
                                 bins=x0[::10],
                                 color=colors[ni],
                                 histtype='step',
                                 label='layer %i' % ni)
                axes[0].set_xlabel("x_0")
                axes[0].set_ylabel("Model output")

                for ni in range(n + 1):
                    axes[1].hist(grid[:, 1].numpy(),
                                 weights=model_on_grid[ni, :, 0],
                                 bins=x1[::10],
                                 lw=2,
                                 color=colors[ni],
                                 histtype='step',
                                 label='layer %i' % ni)
                axes[1].set_xlabel("x_1")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, 'marginals.png'))
                plt.close(fig)

                fig = plt.figure(figsize=(4, 3))
                plt.scatter(grid[:, 0], grid[:, 1], c=Y, edgecolors='none', s=1)
                centr = model.get_centroids().detach().numpy()
                ampl  = model.get_coeffs().detach().numpy()[:, 0]
                centr = centr[ampl > 0]
                plt.scatter(centr[:, 0], centr[:, 1], color='black')
                plt.colorbar()
                plt.xlim(-1.5, 0.5)
                plt.ylim(-0.5, 4.5)
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, '2Dheatmap_%i.png' % n))
                plt.close(fig)

    # GT heatmap
    fig = plt.figure(figsize=(4, 3))
    x0 = torch.arange(-1.5, 0.5, 0.1).double()
    x1 = torch.arange(-0.5, 4.5, 0.05).double()
    plt.hist2d(feature[:, 0].numpy(), feature[:, 1].numpy(),
               bins=[x0.numpy(), x1.numpy()], density=True)
    plt.colorbar()
    plt.xlim(-1.5, 0.5)
    plt.ylim(-0.5, 4.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '2Dheatmap_GT.png'))
    plt.close(fig)

    t2 = time.time()
    print('End training')
    print('Execution time: ', t2 - t1)

    pred = model.call(feature)[-1, :]
    nplm_loss_final = NLL(pred)

    # save test statistic
    with open(os.path.join(output_folder, 't.txt'), 'w') as t_file:
        t_file.write("%f\n" % (-2 * nplm_loss_final))
    print('NPLM test at the end of training: ', "%f" % (-2 * nplm_loss_final))

    # save exec time
    with open(os.path.join(output_folder, 'time.txt'), 'w') as t_file:
        t_file.write("%f\n" % (t2 - t1))

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



