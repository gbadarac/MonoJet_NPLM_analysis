import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Set random seed
np.random.seed(1234)

# Output file paths
output_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/saved_generated_target_data"
training_file = os.path.join(output_dir, "target_training_set.npy")
coverage_file = os.path.join(output_dir, "target_coverage_check_set.npy")

# Parameters
n_bkg = 800000
mean_feat1, std_feat1 = -0.5, 0.25
mean_feat2, std_feat2 = 0.6, 0.4

# Generate target distribution
bkg_feat1 = np.random.normal(mean_feat1, std_feat1, n_bkg)
bkg_feat2 = np.random.normal(mean_feat2, std_feat2, n_bkg)
bkg_coord = np.column_stack((bkg_feat1, bkg_feat2))

# Scale features
scaler = StandardScaler()
bkg_coord_scaled = scaler.fit_transform(bkg_coord).astype('float32')

# Split into train and coverage check sets
train_set = bkg_coord_scaled[:400000]
coverage_check_set = bkg_coord_scaled[400000:]

# Save to files
np.save(training_file, train_set)
np.save(coverage_file, coverage_check_set)

