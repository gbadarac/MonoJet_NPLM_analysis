#!/usr/bin/env python
# coding: utf-8

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from utils import make_flow
import gc

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
parser.add_argument("--hidden_features", type=int, default=64)
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--num_bins", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=5)
parser.add_argument("--collect_all", action="store_true", help="Collect all trained models into f_i.pth")
parser.add_argument("--num_models", type=int, default=1, help="Used with --collect_all to collect N models")
args = parser.parse_args()

# --- Conditional required arguments ---
if not args.collect_all:
    if args.data_path is None or args.seed is None:
        parser.error("--data_path and --seed are required unless --collect_all is used")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------
# If collecting, skip training
# ------------------
if args.collect_all:
    f_i = []
    for i in range(args.num_models):
        model_path = os.path.join(args.outdir, f"model_{i:03d}", "model.pth")
        print(f"Loading model {i:03d}...")
        state_dict = torch.load(model_path, map_location=device)
        f_i.append(state_dict)

        # Immediately clean up memory to avoid accumulation
        del state_dict
        gc.collect()

    torch.save(f_i, os.path.join(args.outdir, "f_i.pth"))
    print(f"Saved f_i.pth with {len(f_i)} models to {args.outdir}")
    exit(0)

# ------------------
# Load data
# ------------------
target_data = np.load(args.data_path) #load data from generate_target_data.py 
target_tensor = torch.from_numpy(target_data)

# ------------------
# Training function
# ------------------
def train_flow(data, model_seed, bootstrap_seed):
    torch.manual_seed(model_seed) #ensure same initialization for all j for a fixed i 

    flow = make_flow(args.num_layers, args.hidden_features, args.num_bins, args.num_blocks)
    
    opt=optim.Adam(flow.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(opt, T_max=args.n_epochs, eta_min=1e-3)
    
    train_size = int(0.8 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=args.batch_size)

    train_losses, val_losses = [], []
    min_loss = np.inf
    patience_counter = 0

    flow.to(device)  # move once you're sure data is on GPU

    for epoch in range(args.n_epochs):
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
            model_dir = os.path.join(args.outdir, f"model_{model_seed:03d}")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(flow.state_dict(), os.path.join(model_dir, "model.pth"))
            metadata = {"seed": model_seed,"bootstrap_seed": bootstrap_seed}
            with open(os.path.join(model_dir, "info.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
            
    return flow

# ------------------
# Loop: model seeds and bootstraps
# ------------------
model_seed = args.seed
bootstrap_seed = args.seed + 10000

#Note: different models are trained on the same bootstrapped datasets, which means (randmness coming from bootstrapping is the same for every trained model) 

print(f"Training model f_{model_seed:03d}...")

np.random.seed(bootstrap_seed)  # Fix bootstrap randomness

boot_indices = np.random.choice(len(target_tensor), size=len(target_tensor), replace=True) #the size of bootstrap dataset is the same as the loaded training set, i.e. 400k 
boot_data = target_tensor[boot_indices]

train_flow(boot_data, model_seed, bootstrap_seed) #consequently the statististical power of the normalizing flow correpsonds to the size of the bnootstrapped dataset, i.e. 400k

# Save architecture config and averaged model if bootstrap_id == 0
if args.seed==0:
    trial_config = {
        "backend": "nflows",
        "num_layers": args.num_layers,
        "hidden_features": args.hidden_features,
        "num_bins": args.num_bins,
        "num_blocks": args.num_blocks
    }
    config_path = os.path.join(args.outdir, "architecture_config.json")
    with open(config_path, "w") as f:
        json.dump(trial_config, f, indent=4)

    

