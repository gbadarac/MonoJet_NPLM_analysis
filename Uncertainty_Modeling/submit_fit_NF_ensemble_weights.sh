#!/bin/bash
#SBATCH --job-name=fit_weights
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/result_weights_NF/logs/fit_weights_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/result_weights_NF/logs/fit_weights_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Activate environment
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# -----------------------
# Parameters
# -----------------------
trial_dir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/EstimationNF_gaussians_bootstrap_outputs/trial_001
data_path=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/saved_generated_target_data/unscaled_reduced_target_training_set.npy
plot=True

# -----------------------
# Derive output path
# -----------------------
trial_name=$(basename "$trial_dir")  # e.g., "trial_000" #extracts the folder name from the full path 
out_dir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/result_weights_NF/${trial_name}  
mkdir -p "$out_dir" #creates a new directory with the correspondent trial name

# Copy trial files to output dir so script runs in isolation
cp "$trial_dir"/f_i_averaged.pth "$out_dir"/

# -----------------------
# Run the Python script
# -----------------------
python fit_NF_ensemble_weights.py \
    --trial_dir "$trial_dir" \
    --data_path "$data_path" \
    --out_dir "$out_dir" \
    --plot "$plot"