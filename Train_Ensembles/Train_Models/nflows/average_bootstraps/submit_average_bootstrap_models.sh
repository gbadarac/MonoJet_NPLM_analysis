#!/bin/bash
#SBATCH --job-name=avg_ensemble
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/logs/avg_ensemble_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/logs/avg_ensemble_%j.err
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

# ======================================
# Ensure Python can find utils.py
# ======================================
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH

# Run the script
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/average_bootstrap_models.py \
--trial_dir /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/N_100000_seeds_60_bootstraps_1_4_16_128_15

