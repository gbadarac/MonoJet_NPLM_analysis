import glob, h5py, math, time, os, json, random, yaml, argparse, datetime
from scipy.stats import norm, expon, chi2, uniform, chisquare
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist
from pathlib import Path
import torch
import jax.numpy as jnp
from jax import random as jax_random   # avoid clashing with Python's random
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches

from torch.autograd.functional import hessian
from torch.autograd import grad

# === Sparker utils (now as a proper package) ===
from Sparker_utils.SPARKutils import *
from Sparker_utils.PLOTutils import *
from Sparker_utils.GENutils import *
from Sparker_utils.sampler_mixture import *
from Sparker_utils.losses import *
from Sparker_utils.regularizers import *


# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to training data")
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--seed", type=int, help="Random seed for training")
parser.add_argument("--n_epochs", type=int, default=1001)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--collect_all", action="store_true", help="Collect all trained models into f_i.pth")
parser.add_argument("--num_models", type=int, default=1, help="Used with --collect_all to collect N models")
parser.add_argument("--num_features", type=int, required=True, help="Dimensionality of the data")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------
# Load data 
# ------------------
target_data = np.load(args.data_path) #load data from generate_target_data.py 
target_tensor = torch.from_numpy(target_data)

