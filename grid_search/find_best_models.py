import os
import numpy as np

base_dir='/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs'

def find_best_models(base_dir):
    # Create a list to store the KL divergence and model parameters
    kl_results = []

    # Walk through all subdirectories (job directories)
    for job_dir in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_dir)
        
        if os.path.isdir(job_path):
            # Look for KL divergence file in each job directory
            kl_file = [f for f in os.listdir(job_path) if f.startswith("kl_divergence")]
            if kl_file:
                # Load KL divergence
                kl_div = np.load(os.path.join(job_path, kl_file[0]))

                # Extract model parameters from the directory name
                # Example format: "job_6_8_75_15_12345"
                # Extract the parts of the directory name
                parts = job_dir.split('_')
                
                # Assuming the order: num_layers, num_blocks, hidden_features, num_bins
                try:
                    params = {
                        "num_layers": int(parts[1]),
                        "num_blocks": int(parts[2]),
                        "hidden_features": int(parts[3]),
                        "num_bins": int(parts[4]),
                    }
                except IndexError:
                    # In case the directory structure is not as expected, skip this directory
                    print(f"Skipping invalid directory: {job_dir}")
                    continue

                # Append to results
                kl_results.append((kl_div, params, job_path))

    # Sort by KL divergence (ascending)
    sorted_kl_results = sorted(kl_results, key=lambda x: x[0])

    # Get the top 5 models
    best_5_models = sorted_kl_results[:5]

    # Print out the results
    for idx, (kl_div, params, job_path) in enumerate(best_5_models):
        print(f"Rank {idx+1}: KL Divergence = {kl_div:.9f}")
        print(f"Model Parameters: {params}")
        print(f"Job Directory: {job_path}")
        print("-" * 40)

    return best_5_models

# Example usage
base_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs"
best_models = find_best_models(base_dir)
