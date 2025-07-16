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