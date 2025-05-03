#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import torch
from nflows.flows import Flow
from utils import make_flow 

def average_state_dicts(state_dicts, weights):
    """Weighted average of PyTorch state dicts"""
    avg_dict = {}
    for key in state_dicts[0]:
        avg_dict[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts)) #computes weighted averga eof each parameter across the models 
    return avg_dict

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets device to GPU if available 
    os.makedirs(args.trial_dir, exist_ok=True)

    model_dirs = sorted([d for d in os.listdir(args.trial_dir) if d.startswith("model_")]) # loops through models
    f_i_models = []
    w_i_values = []

    for model_dir in model_dirs:
        model_path = os.path.join(args.trial_dir, model_dir)
        bootstrap_dirs = sorted([d for d in os.listdir(model_path) if d.startswith("bootstrap_")]) #loops throgh bootstrapped datasets 

        state_dicts = []
        for b_dir in bootstrap_dirs: 
            model_file = os.path.join(model_path, b_dir, "model.pth")
            state_dicts.append(torch.load(model_file, map_location=device)) # Loads all trained bootstrap models under model_i

        # Uniform weights over j
        weights_j = [1.0 / len(state_dicts)] * len(state_dicts) # uses uniform weights over j 
        avg_state_dict = average_state_dicts(state_dicts, weights_j) # averages all bootstrap models to get f_i 
 
        # Load model architecture config from first bootstrap
        with open(os.path.join(model_path, bootstrap_dirs[0], "config.json")) as f:
            config = json.load(f) # Grabs saved architecture/hyperparameters so we can re-initialize the model with the correct shape

        # Create and load flow_i
        flow_i = make_flow(
            num_layers=config["num_layers"],
            hidden_features=config["hidden_features"],
            num_blocks=config["num_blocks"],
            num_bins=config["num_bins"],
            num_features=2,
            num_context=None,
            perm=True
        ).to(device) #construct f_i with the correct architecture 
        flow_i.load_state_dict(avg_state_dict) #loads in the averaged weights 
        f_i_models.append(flow_i) #stores models f_i in memory 

        # Assign uniform model-level weight
        w_i_values.append(1.0)
        
    # Check if models were collected
    if not f_i_models:
        raise RuntimeError("No models were collected. Please check the trial directory and ensure training was successful.")

    # Optionally: print how many and what type
    print(f"\nCollected {len(f_i_models)} models of type {type(f_i_models[0])}")

    # Normalize and save w_i weights
    w_i_initial = np.array(w_i_values)
    w_i_initial /= np.sum(w_i_initial)

    # Save all f_i 
    f_i_averaged = [f.state_dict() for f in f_i_models]
    torch.save(f_i_averaged, os.path.join(args.trial_dir, "f_i_averaged.pth"))

    # Save normalized weights
    np.save(os.path.join(args.trial_dir, "w_i_initial.npy"), w_i_initial)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir", type=str, required=True, help="Path to trial directory (e.g., .../trial_000)")
    args = parser.parse_args()
    main(args)