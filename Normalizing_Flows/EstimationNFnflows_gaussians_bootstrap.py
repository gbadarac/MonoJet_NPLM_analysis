#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import torch
import os
import argparse
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.flows import Flow

hep.style.use("CMS")

# ------------------
# Argument Parsing
# ------------------
parser = argparse.ArgumentParser(description='Normalizing Flow Bootstrap Training')
parser.add_argument('--n_epochs', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--num_layers', type=int, required=True)
parser.add_argument('--num_blocks', type=int, required=True)
parser.add_argument('--hidden_features', type=int, required=True)
parser.add_argument('--num_bins', type=int, required=True)
parser.add_argument('--bootstrap_seed', type=int, required=True)
parser.add_argument('--run_id', type=int, required=True)
args = parser.parse_args()


# ------------------
# Target Distribution
# ------------------
n_bkg = 800000
mean_feat1, std_feat1 = -0.5, 0.25
mean_feat2, std_feat2 = 0.6, 0.4

bkg_feat1 = np.random.normal(mean_feat1, std_feat1, n_bkg)
bkg_feat2 = np.random.normal(mean_feat2, std_feat2, n_bkg)
bkg_coord = np.column_stack((bkg_feat1, bkg_feat2))

scaler = StandardScaler()
bkg_coord_scaled = scaler.fit_transform(bkg_coord).astype('float32')

#splitting in valudation and training and save file with training sample and validation sample per il check dopo UNA VOLTA SOLA PER TUTTO 


# ------------------
# Set seeds for reproducibility
# ------------------

# Dataset randomness — stays constant for a given bootstrap_seed
np.random.seed(args.bootstrap_seed)  # Fixes dataset sampling and performs bootstrapped sampling in np.random.choice()

# Model randomness — stays constant for a given bootstrap_seed
torch.manual_seed(args.bootstrap_seed) # Fixes model initialization (deterministic weight initialization)
torch.cuda.manual_seed_all(args.bootstrap_seed) # Fixes model initialization on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Bootstrap sample
n_samples = bkg_coord_scaled.shape[0]
bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
bootstrapped_data = bkg_coord_scaled[bootstrap_indices]

# ------------------
# Model Definition
# ------------------
num_features = 2
num_context = None

def make_flow(num_features, num_context, perm=True):
    base_dist = StandardNormal(shape=(num_features,))
    transforms = []
    if num_context == 0:
        num_context = None
    for i in range(args.num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform( #randomly initialized by PyTorch number generator
            features=num_features,
            context_features=num_context,
            hidden_features=args.hidden_features,
            num_bins=args.num_bins,
            num_blocks=args.num_blocks,
            tail_bound=10.0,
            tails='linear',
            dropout_probability=0.2,
            use_batch_norm=False
        ))
        if i < args.num_layers - 1 and perm:
            transforms.append(ReversePermutation(features=num_features))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

flow = make_flow(num_features, num_context, perm=True)

# ------------------
# Training Preparation
# ------------------
opt = optim.Adam(flow.parameters(), args.learning_rate)
scheduler = CosineAnnealingLR(opt, T_max=args.n_epochs, eta_min=1e-3)

# Data preparation

# Sample a subset of 100k points, differently per run_id
rng = np.random.default_rng(seed=args.run_id)
subset_indices = rng.choice(len(bootstrapped_data), size=100000, replace=False)
subset_data = bootstrapped_data[subset_indices]

# Reproducibility check
first_point = subset_data[0]
print(f"[bootstrap_seed={args.bootstrap_seed}, run_id={args.run_id}] First datapoint:", first_point)

y = torch.from_numpy(subset_data)
train_size = int(0.8 * len(y))
train_data, val_data = y[:train_size], y[train_size:]
train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=args.batch_size)

# ------------------
# Training Loop
# ------------------
train_losses, val_losses = [], []
min_loss = np.inf
patience_counter = 0

model_dir = os.path.join(args.outdir, f"seed_{args.bootstrap_seed:04d}", f"run_{args.run_id:02d}")
os.makedirs(model_dir, exist_ok=True)

for epoch in range(args.n_epochs):
    flow.train()
    total_train_loss = 0.0
    for batch in train_loader:
        batch_data = batch[0]
        opt.zero_grad()
        train_loss = -flow.log_prob(batch_data).mean()
        train_loss.backward()
        opt.step()
        total_train_loss += train_loss.item() * batch_data.size(0)
    avg_train_loss = total_train_loss / len(train_data)
    train_losses.append(avg_train_loss)

    flow.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            val_loss = -flow.log_prob(val_batch[0]).mean()
            total_val_loss += val_loss.item() * val_batch[0].size(0)
    avg_val_loss = total_val_loss / len(val_data)
    val_losses.append(avg_val_loss)
    scheduler.step()

    if avg_val_loss < min_loss:
        min_loss = avg_val_loss
        torch.save(flow.state_dict(), os.path.join(model_dir, "model.pth"))
        # Save config too
        config = {
            "bootstrap_seed": args.bootstrap_seed,
            "run_id": args.run_id,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "hidden_features": args.hidden_features,
            "num_blocks": args.num_blocks,
            "num_bins": args.num_bins,
            "num_layers": args.num_layers,
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping triggered")
            break

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("Training complete.")
