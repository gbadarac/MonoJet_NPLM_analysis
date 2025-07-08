#!/usr/bin/env python

import os
import json
import torch
import numpy as np
import argparse
import sys
import gc
import matplotlib
matplotlib.use('Agg')  # Needed for non-interactive environments
import matplotlib.pyplot as plt
import mplhep as hep
import torch.nn.functional as F
from torchmin import minimize

# Custom imports
sys.path.append("/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling")
from utils import make_flow, probs


def main(trial_dir, arch_config_path, out_dir, N_generated):
    os.makedirs(out_dir, exist_ok=True)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load architecture configuration
    with open(os.path.join(arch_config_path, "architecture_config.json")) as f:
        config = json.load(f)

    # Load model state_dicts
    f_i_file = os.path.join(trial_dir, "f_i.pth")
    f_i_statedicts = torch.load(f_i_file, map_location=device)

    # Reconstruct flows and sample from each
    sampled_mean = []

    for i, state_dict in enumerate(f_i_statedicts):
        flow = make_flow(
            num_layers=config["num_layers"],
            hidden_features=config["hidden_features"],
            num_bins=config["num_bins"],
            num_blocks=config["num_blocks"],
        ).to(device)

        flow.load_state_dict(state_dict)
        flow.eval()

        with torch.no_grad():
            gen_data = flow.sample(N_generated).cpu().numpy()

        mean = np.mean(gen_data, axis=0)
        sampled_mean.append(mean)
        print(f"[{i+1}/{len(f_i_statedicts)}] mean: {mean}")

        # --- Crucial cleanup ---
        del flow, gen_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sampled_mean = np.stack(sampled_mean)
    print("Final shape of sampled_mean:", sampled_mean.shape)

    out_file = os.path.join(
        out_dir,
        f"generated_sampled_mean_{len(f_i_statedicts)}_models_{config['num_layers']}_{config['num_blocks']}_{config['hidden_features']}_{config['num_bins']}_v2.npy"
    )
    np.save(out_file, sampled_mean)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save means of samples from Normalizing Flows")
    parser.add_argument("--trial_dir", type=str, required=True)
    parser.add_argument("--arch_config_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--N_generated", type=int, default=500000)
    args = parser.parse_args()

    main(args.trial_dir, args.arch_config_path, args.out_dir, args.N_generated)
