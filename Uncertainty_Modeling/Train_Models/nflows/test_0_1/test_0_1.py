#!/usr/bin/env python
# coding: utf-8

import os, json, argparse
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import mplhep as hep
hep.style.use("CMS")
from utils import make_flow

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--outdir", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------
# Load data
# ------------------
target_data = np.load(args.data_path) #load data from generate_target_data.py 
target_tensor = torch.from_numpy(target_data)

# ------------------
# Training function
# ------------------
def train_flow(data, num_layers, hidden_features, num_bins, num_blocks, tail_bound, learning_rate, n_epochs, batch_size, model_flag):

    flow = make_flow(num_layers, hidden_features, num_bins, num_blocks, tail_bound)
    opt=optim.Adam(flow.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-3)
    
    train_size = int(0.8 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)

    train_losses, val_losses = [], []
    min_loss = np.inf
    patience_counter = 0

    flow.to(device)  # move once you're sure data is on GPU

    for epoch in range(n_epochs):
        flow.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch[0].to(device)
            opt.zero_grad()
            train_loss = -flow.log_prob(batch).mean()
            train_loss.backward()
            opt.step()
            total_train_loss += train_loss.item() * batch.size(0)
        avg_train_loss = total_train_loss / len(train_data)
        train_losses.append(avg_train_loss)

        flow.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch[0].to(device)
                val_loss = -flow.log_prob(val_batch).mean()
                total_val_loss += val_loss.item() * val_batch.size(0)
        avg_val_loss = total_val_loss / len(val_data)
        val_losses.append(avg_val_loss)
        scheduler.step()
        
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            model_dir = os.path.join(args.outdir, f"{model_flag}_model")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(flow.state_dict(), os.path.join(model_dir, "model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    # Save model-specific architecture config at the end of training
    model_dir = os.path.join(args.outdir, f"{model_flag}_model")
    os.makedirs(model_dir, exist_ok=True)  # In case no checkpoint was saved earlier

    arch_config = {
        "num_layers": num_layers,
        "hidden_features": hidden_features,
        "num_bins": num_bins,
        "num_blocks": num_blocks,
        "tail_bound": tail_bound
    }
    with open(os.path.join(model_dir, "architecture_config.json"), "w") as f:
        json.dump(arch_config, f, indent=4)
    return flow

lr=5e-6
epochs=1001
batch=512

max_abs = torch.abs(target_tensor).max().item()
tail_bound = 1.2 * max_abs
print(f"[DEBUG] Computed tail_bound = {tail_bound:.3f}")

good_model=train_flow(target_tensor, 2, 64, 6, 2, tail_bound, lr, epochs, batch, model_flag='good')

bad_model = make_flow(1, 4, 2, 1, 10)
bad_model.to(device)
bad_model_dir = os.path.join(args.outdir, "bad_model")
os.makedirs(bad_model_dir, exist_ok=True)
torch.save(bad_model.state_dict(), os.path.join(bad_model_dir, "model.pth"))

# Save matching architecture config
bad_arch_config = {
    "num_layers": 1,
    "hidden_features": 4,
    "num_bins": 2,
    "num_blocks": 1,
    "tail_bound": 10
}
with open(os.path.join(bad_model_dir, "architecture_config.json"), "w") as f:
    json.dump(bad_arch_config, f, indent=4)

# ------------------
# Save State Dicts + Initial Weights
# ------------------
f_i_models = [good_model.state_dict(), bad_model.state_dict()]
torch.save(f_i_models, os.path.join(args.outdir, "f_i_averaged.pth"))

w_i_initial = np.array([0.5, 0.5])
np.save(os.path.join(args.outdir, "w_i_initial.npy"), w_i_initial)