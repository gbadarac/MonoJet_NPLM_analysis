#!/bin/bash
#SBATCH --job-name=wifi_kernels
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/logs/wifi_kernels_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/logs/wifi_kernels_%j.err

#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

# >>> Conda setup for non-interactive shell <<<
# Option A (recommended): use conda.sh
source /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate kernels_env

# If for some reason conda.sh didn't exist, you could instead do:
# source /work/gbadarac/miniforge3/bin/activate kernels_env

# Go to the directory where the WiFi script lives
cd /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights

# Make sure logs directory exists
mkdir -p results_fit_weights_kernel/logs

FOLDER_PATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking
DATA_PATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy

# -----------------------
# Ensure Python can find utils_wifi.py
# -----------------------
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

python fit_kernel_ensemble_weights_2d.py \
  --folder_path "${FOLDER_PATH}" \
  --data_path "${DATA_PATH}" \
  --n_wifi_components 60 \
  --epochs 2000 \
  --patience 100 \
  --lr 0.001 \
  --seed_bootstrap 1234 \
  --compute_covariance
