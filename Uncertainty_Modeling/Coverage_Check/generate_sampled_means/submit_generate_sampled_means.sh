#!/bin/bash
#SBATCH --job-name=generated_samples_means
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/generate_sampled_means/results_generated_sampled_means/logs/gen_sampled_means_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/generate_sampled_means/results_generated_sampled_means/logs/gen_sampled_means_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --nodes=1

# Activate environment
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

python generate_sampled_means.py \
  --trial_dir /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Fit_Weights/results_fit_weights_NF/N_100000_seeds_60_4_16_128_15 \
  --arch_config_path /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/nflows/EstimationNFnflows_outputs/N_100000_seeds_60_4_16_128_15 \
  --out_dir /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/generate_sampled_means/results_generated_sampled_means 