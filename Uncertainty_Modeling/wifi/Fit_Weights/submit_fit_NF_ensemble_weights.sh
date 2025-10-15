#!/bin/bash
#SBATCH --job-name=fit_weights
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/logs/fit_weights_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/logs/fit_weights_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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
trial_dir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_60_4_16_128_15

data_path=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy
plot=True

# -----------------------
# Derive output path
# -----------------------
trial_name=$(basename "$trial_dir")  # e.g., "trial_000" #extracts the folder name from the full path 
out_dir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/${trial_name}_bimodal_gaussian_heavy_tail
mkdir -p "$out_dir" #creates a new directory with the correspondent trial name

# Copy trial files to output dir so script runs in isolation
cp "$trial_dir"/f_i.pth "$out_dir"/

# -----------------------
# Ensure Python can find utils_wifi.py
# -----------------------
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi:$PYTHONPATH
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# -----------------------
# Run the Python script
# -----------------------
python fit_NF_ensemble_weights_2d.py \
    --trial_dir "$trial_dir" \
    --data_path "$data_path" \
    --out_dir "$out_dir" \
    --plot "$plot"