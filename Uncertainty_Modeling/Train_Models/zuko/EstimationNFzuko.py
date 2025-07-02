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
from utils import make_flow_zuko


# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--n_epochs", type=int, default=1001)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--hidden_features", type=int, default=64)
parser.add_argument("--num_blocks", type=int, default=2)
parser.add_argument("--num_bins", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=5)
parser.add_argument("--bayesian", action="store_true")  
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------
# Load data
# ------------------
target_data = np.load(args.data_path) #load data from generate_target_data.py 
target_tensor = torch.from_numpy(target_data)
print('target_tensor shape:', target_tensor.shape)

# ------------------
# Training function
# ------------------
def train_flow(data, seed, indices):
    print("DEBUG: Input data shape to flow:", data.shape)

    torch.manual_seed(seed) #ensure same initialization for all j for a fixed i 

    flow = make_flow_zuko(
        num_layers=args.num_layers,
        hidden_features=args.hidden_features,
        num_bins=args.num_bins,
        num_blocks=args.num_blocks,
        num_features=2,
        num_context=0,  
        bayesian=args.bayesian
    )

    print("DEBUG: Flow object:", flow)
    print("DEBUG: flow._transform", flow.transform)

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
            batch = batch[:, :2]
            print("DEBUG: Batch shape:", batch.shape)
            opt.zero_grad()
            # Get first transform (MaskedAutoregressiveTransform)
            first_transform = flow.transform.transforms[0]
            print("DEBUG: First transform:", first_transform)

            # Get its hypernetwork (MaskedMLP)
            hypernet = first_transform.hyper
            print("DEBUG: Hypernet structure:", hypernet)

            # Get the first MaskedLinear layer in the MLP
            first_layer = hypernet[0]
            print("DEBUG: First MaskedLinear layer:", first_layer)

            # Print expected input/output shapes of the first layer
            print("DEBUG: Expected in_features:", first_layer.in_features)
            print("DEBUG: Expected out_features:", first_layer.out_features)

            # Print batch shape going into the model
            print("DEBUG: Input batch shape:", batch.shape)

            train_loss = -flow(batch).log_prob(batch).mean()
            if args.bayesian:
                train_loss += 1e-3 * total_KL_divergence(flow)  # KL scaling factor is a hyperparameter
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
                val_loss = -flow(val_batch).log_prob(val_batch).mean()
                total_val_loss += val_loss.item() * val_batch.size(0)
        avg_val_loss = total_val_loss / len(val_data)
        val_losses.append(avg_val_loss)
        scheduler.step()
        
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            model_dir = os.path.join(args.outdir, f"model_{seed:03d}")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(flow.state_dict(), os.path.join(model_dir, "model.pth"))
            np.save(os.path.join(model_dir, "bootstrap_indices.npy"), indices)
            with open(os.path.join(model_dir, "info.json"), "w") as f:
                json.dump({"seed": seed}, f, indent=4)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
            
    return flow

# ------------------
# Loop: model seeds and bootstraps
# ------------------
seed = args.seed

#Note: different models are trained on the same bootstrapped datasets, which means (randmness coming from bootstrapping is the same for every trained model) 

print(f"Training model f_{seed:03d}...")

np.random.seed(seed)  # Fix bootstrap randomness
boot_indices = np.random.choice(len(target_tensor), size=len(target_tensor), replace=True) #the size of bootstrap dataset is the same as the loaded training set, i.e. 400k 
boot_data = target_tensor[boot_indices].float()
print("Bootstrapped dataset size:", boot_data.shape)

train_flow(boot_data, seed, boot_indices) #consequently the statististical power of the normalizing flow correpsonds to the size of the bnootstrapped dataset, i.e. 400k

# Save architecture config and averaged model if bootstrap_id == 0
if args.seed==0:
    trial_config = {
        "num_layers": args.num_layers,
        "hidden_features": args.hidden_features,
        "num_bins": args.num_bins,
        "num_blocks": args.num_blocks
    }
    config_path = os.path.join(args.outdir, "architecture_config.json")
    with open(config_path, "w") as f:
        json.dump(trial_config, f, indent=4)

    # Save f_i_averaged.pth as single model in list (no real averaging)
    model_dir = os.path.join(args.outdir, f"model_{args.seed:03d}")
    model_file = os.path.join(model_dir, "model.pth")
    f_i = [torch.load(model_file, map_location=device)]
    torch.save(f_i, os.path.join(args.outdir, "f_i.pth"))





