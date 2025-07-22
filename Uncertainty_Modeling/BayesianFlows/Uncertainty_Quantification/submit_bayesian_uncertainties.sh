#!/bin/bash
#SBATCH --job-name=bayesian_flows
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/BayesianFlows/Uncertainty_Quantification/results_BFs/logs/BFs_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/BayesianFlows/Uncertainty_Quantification/results_BFs/logs/BFs_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
# #SBATCH --gres=gpu:1
# #SBATCH --account=gpu_gres
#SBATCH --partition=standard
#SBATCH --nodes=1

# Activate environment
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# -----------------------
# Parameters
# -----------------------
trial_dir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko_outputs/N_10000_seeds_1_4_16_128_15_bayesian

data_path=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/100k_target_training_set.npy
plot=True

# -----------------------
# Derive output path
# -----------------------
trial_name=$(basename "$trial_dir")  # e.g., "trial_000" #extracts the folder name from the full path 
out_dir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/BayesianFlows/Uncertainty_Quantification/results_BFs/${trial_name}
mkdir -p "$out_dir" #creates a new directory with the correspondent trial name

# -----------------------
# Ensure Python can find utils_wifi.py
# -----------------------
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/BayesianFlows:$PYTHONPATH
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH

# -----------------------
# Run the Python script
# -----------------------
python bayesian_uncertainties.py \
    --trial_dir "$trial_dir" \
    --data_path "$data_path" \
    --out_dir "$out_dir" \
    --plot "$plot"