#!/usr/bin/env python
# coding: utf-8

import os, json, argparse
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import sys
sys.path.insert(0, "/work/gbadarac/zuko")
import zuko
print(zuko.__file__)
from zuko.utils import total_KL_divergence
from utils_flows import make_flow_zuko
import mplhep as hep

# Use CMS style for plots
hep.style.use("CMS")

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--trial_dir", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--plot", type=str, required=True)
args = parser.parse_args()

# ------------------
# Load models and initial weights 
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets device to GPU if available 
os.makedirs(args.trial_dir, exist_ok=True)

# ------------------
# Load architecture config and data 
# ------------------
# Save architecture 

with open(os.path.join(args.trial_dir, "architecture_config.json")) as f:
    config = json.load(f) 

flow = make_flow_zuko(num_layers=config["num_layers"],hidden_features=config["hidden_fetaures"], num_bins=config["num_bins"], num_blocks=config["num_blocks"], num_features=2, num_context=0, bayesian=config["bayesian"])

flow.load_state_dict(torch.load(os.path.join(args.trial_dir, "model_000", "model.pth"), map_location=device))
flow.eval()

x_data = torch.from_numpy(np.load(args.data_path)).float().to(device)

f_i_models = []
with torch.no_grad():
    for i in range(100):
        f_i_models.append(flow().log_prob(x_data))


D = torch.stack(f_i_models, dim=1)
D_mean = D.mean(dim=1)
D_std = D.exp().std(dim=1)
