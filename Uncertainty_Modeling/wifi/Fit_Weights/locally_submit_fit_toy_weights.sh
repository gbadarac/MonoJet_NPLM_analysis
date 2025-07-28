#!/bin/bash

# Activate environment
source ~/miniforge3/bin/activate
conda activate myenv

# -----------------------
# Parameters
# -----------------------
trial_dir=/Users/jafu/cernbox/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/average_bootstraps/EstimationNFnflows_outputs/N_100000_seeds_60_bootstraps_1_4_16_128_15

data_path=/Users/jafu/Documents/VSCode_Projects/MastersThesis/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/100k_target_training_set.npy
plot=True

# -----------------------
# Derive output path
# -----------------------
trial_name=$(basename "$trial_dir")  # e.g., "trial_000" #extracts the folder name from the full path 
out_dir=/Users/jafu/Documents/VSCode_Projects/MastersThesis/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/${trial_name}
mkdir -p "$out_dir" #creates a new directory with the correspondent trial name

# Copy trial files to output dir so script runs in isolation
cp "$trial_dir"/f_i.pth "$out_dir"/

# -----------------------
# Ensure Python can find utils_wifi.py
# -----------------------
export PYTHONPATH=/Users/jafu/Documents/VSCode_Projects/MastersThesis/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi:$PYTHONPATH
export PYTHONPATH=/Users/jafu/Documents/VSCode_Projects/MastersThesis/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH

# -----------------------
# Run the Python script
# -----------------------
python fit_NF_ensemble_weights.py \
    --trial_dir "$trial_dir" \
    --data_path "$data_path" \
    --out_dir "$out_dir" \
    --plot "$plot"