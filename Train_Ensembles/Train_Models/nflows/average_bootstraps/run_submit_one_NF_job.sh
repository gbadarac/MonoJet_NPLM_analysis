#!/bin/bash
#SBATCH --job-name=NF_launcher
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/logs/launcher_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/logs/launcher_error_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --partition=gpu
#SBATCH --account=gpu_gres

# ======================================
# Activate environment
# ======================================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# ======================================
# USER PARAMETERS
# ======================================

# setup
model_seeds=60
bootstrap_runs=1

# Model hyperparameters
n_epochs=1001
learning_rate=5e-6
batch_size=512
hidden_features=128
num_blocks=16
num_bins=15
num_layers=4

# Base directory
base_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs"

#Get dataset size 
dataset_path="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/100k_target_training_set.npy"
num_events=$(python -c "import numpy as np; print(np.load('${dataset_path}').shape[0])")

# ======================================
# Create new trial folder automatically
# ======================================
trial_dir="${base_dir}/N_${num_events}_seeds_${model_seeds}_bootstraps_${bootstrap_runs}_${num_layers}_${num_blocks}_${hidden_features}_${num_bins}"
mkdir -p "${trial_dir}"

# ======================================
# Submit jobs
# ======================================
for seed in $(seq 0 $((0+model_seeds-1))); do
  for run_id in $(seq 0 $((bootstrap_runs-1))); do
    sbatch --export=ALL,MODEL_SEED=${seed},BOOTSTRAP_ID=${seed},TRIAL_DIR=${trial_dir},N_EPOCHS=${n_epochs},LR=${learning_rate},BATCH_SIZE=${batch_size},HIDDEN_FEATURES=${hidden_features},NUM_BLOCKS=${num_blocks},NUM_BINS=${num_bins},NUM_LAYERS=${num_layers} submit_one_NF_job.sh
  done
done

