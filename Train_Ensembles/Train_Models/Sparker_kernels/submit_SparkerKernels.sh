#!/bin/bash
#SBATCH --job-name=Kernels_job
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/logs/kernels_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/logs/kernels_error_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ======================================
# Activate environment
# ======================================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env   # or a dedicated env for kernels

# ======================================
# USER PARAMETERS
# ======================================

# Seed and problem dimensionality
seed=0
num_features=2

# Model hyperparameters (adapt to your kernels script)
n_epochs=1001
learning_rate=5e-4
batch_size=2048

# Example kernel specific hyperparameters
num_centers=256
length_scale=0.5
regularization=1e-3

# Base directory for this study
base_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Kernels/outputs/2_dim/2d_bimodal_gaussian_heavy_tail"

# Dataset path
dataset_path="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy"

# Get dataset size (for folder naming, optional)
num_events=$(python -c "import numpy as np; print(np.load('${dataset_path}').shape[0])")

# ======================================
# Create trial folder
# ======================================
trial_dir="${base_dir}/N_${num_events}_dim_${num_features}_seed_${seed}_nc_${num_centers}_ls_${length_scale}_reg_${regularization}"
mkdir -p "${trial_dir}"

# Make sure Python can find your modules
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models:$PYTHONPATH

# ======================================
# Run training once
# ======================================
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Kernels/TrainKernels.py \
    --data_path "${dataset_path}" \
    --outdir "${trial_dir}" \
    --seed "${seed}" \
    --n_epochs "${n_epochs}" \
    --learning_rate "${learning_rate}" \
    --batch_size "${batch_size}" \
    --num_features "${num_features}" \
    --num_centers "${num_centers}" \
    --length_scale "${length_scale}" \
    --regularization "${regularization}"
