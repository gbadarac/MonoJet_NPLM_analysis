import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Set random seed
np.random.seed(1234)

# Output file paths
output_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Generate_Data/saved_generated_target_data"
training_file = os.path.join(output_dir, "10k_target_training_set.npy")

# Parameters
n_bkg = 10000
mean_feat1, std_feat1 = -0.5, 0.25
mean_feat2, std_feat2 = 0.6, 0.4

# Generate target distribution
bkg_feat1 = np.random.normal(mean_feat1, std_feat1, n_bkg)
bkg_feat2 = np.random.normal(mean_feat2, std_feat2, n_bkg)
bkg_coord = np.column_stack((bkg_feat1, bkg_feat2)).astype('float32')
'''
#save target moment for coverage 
mu_target = np.array([mean_feat1, mean_feat2], dtype=np.float32)
np.save(os.path.join(output_dir, "mu_target.npy"), mu_target)
'''

# Save to files
np.save(training_file, bkg_coord)

