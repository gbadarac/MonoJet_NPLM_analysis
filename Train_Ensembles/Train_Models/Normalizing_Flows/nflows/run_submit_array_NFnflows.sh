#!/bin/bash
#SBATCH --job-name=NF_launcher
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/logs/launcher_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/logs/launcher_error_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


# ======================================
# Activate environment
# ======================================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# ======================================
# USER PARAMETERS
# ======================================
# setup
model_seeds=80
num_features=2

# Model hyperparameters
n_epochs=1001
learning_rate=5e-6
batch_size=512
hidden_features=256
num_blocks=16
num_bins=15
num_layers=4

# Base directory
base_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail"
#Get dataset size 
dataset_path="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy"
num_events=$(python -c "import numpy as np; print(np.load('${dataset_path}').shape[0])")

# ======================================
# Create new trial folder automatically
# ======================================
trial_dir="${base_dir}/N_${num_events}_dim_${num_features}_seeds_${model_seeds}_${num_layers}_${num_blocks}_${hidden_features}_${num_bins}"
mkdir -p "${trial_dir}"

# === Submit array job with all parameters
array_range="0-$((${model_seeds}-1))"

array_job_id=$(sbatch \
  --parsable \
  --export=ALL,MODEL_SEEDS=${model_seeds},TRIAL_DIR=${trial_dir},DATA_PATH=${dataset_path},N_EPOCHS=${n_epochs},LR=${learning_rate},BATCH_SIZE=${batch_size},HIDDEN_FEATURES=${hidden_features},NUM_BLOCKS=${num_blocks},NUM_BINS=${num_bins},NUM_LAYERS=${num_layers},NUM_FEATURES=${num_features} \
  --array=${array_range} \
  submit_array_NFnflows.sh | cut -d'_' -f1)

# === Submit collection job after array finishes
sbatch --job-name=NF_collect \
  --dependency=afterok:${array_job_id} \
  --output=${base_dir}/logs/collect_output_%j.out \
  --error=${base_dir}/logs/collect_error_%j.err \2
  --time=00:10:00 \
  --mem=4G \
  --partition=standard \
  --wrap="source /work/gbadarac/miniforge3/bin/activate && conda activate nf_env && export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models:\$PYTHONPATH && \
         python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows.py \
           --outdir ${trial_dir} --collect_all --num_models ${model_seeds} --num_features ${num_features}"