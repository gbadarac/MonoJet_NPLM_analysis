import numpy as np
import os

# Set random seed
np.random.seed(1234)

# Output file paths
output_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/4_dim"
os.makedirs(output_dir, exist_ok=True)
training_file = os.path.join(output_dir, "100k_target_training_set.npy")

# Parameters
n_bkg = 100000

# Set per-feature means and stds (edit these as you like)
means = np.array([-0.5, 0.6, 0.2, -0.1], dtype=np.float32)
stds  = np.array([ 0.25, 0.4, 0.3,  0.2], dtype=np.float32)

# Generate 4D target distribution: shape (n_bkg, 4)
bkg_coord = np.random.normal(loc=means, scale=stds, size=(n_bkg, 4)).astype('float32')

# Save target moment for coverage (the mean vector)
mu_target = means.astype(np.float32)
np.save(os.path.join(output_dir, "mu_target.npy"), mu_target)

# Save samples
np.save(training_file, bkg_coord)


