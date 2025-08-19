#!/bin/bash
#SBATCH --job-name=generated_samples_means
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/generate_sampled_means/results_generated_sampled_means/logs/gen_sampled_means_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/generate_sampled_means/results_generated_sampled_means/logs/gen_sampled_means_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu 
#SBATCH --nodes=1

# Activate environment
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# ============
# Parameters
# ============
TRIAL_NAME="N_100000_dim_2_seeds_60_4_8_64_8"

TRIAL_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/${TRIAL_NAME}"
ARCH_CONFIG_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/2_dim/${TRIAL_NAME}"
OUT_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/generate_sampled_means/results_generated_sampled_means"

# ============
# Run script
# ============
python generate_sampled_means.py \
  --trial_dir "$TRIAL_DIR" \
  --arch_config_path "$ARCH_CONFIG_DIR" \
  --out_dir "$OUT_DIR"