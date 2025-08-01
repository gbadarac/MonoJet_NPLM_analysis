#!/bin/bash
#SBATCH --job-name=NF_launcher
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko_outputs/logs/launcher_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko_outputs/logs/launcher_error_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=96G
#SBATCH --partition=standard
# #SBATCH --gres=gpu:1


# ======================================
# Activate environment
# ======================================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# ======================================
# USER PARAMETERS
# ======================================
# setup
model_seeds=1
# Model hyperparameters
n_epochs=1001
learning_rate=5e-6
batch_size=512
hidden_features=128
num_blocks=16
num_bins=15
num_layers=4
bayesian=True  # Set to True if you want to use Bayesian NF

# Base directory
base_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko_outputs"
#Get dataset size 
dataset_path="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/10k_target_training_set.npy"
num_events=$(python -c "import numpy as np; print(np.load('${dataset_path}').shape[0])")

# Add flag to indicate Bayesian training in folder name
bayes_tag=""
if [ "$bayesian" = true ] || [ "$bayesian" = True ]; then
    bayes_tag="_bayesian"
fi

# ======================================
# Create new trial folder automatically
# ======================================
trial_dir="${base_dir}/N_${num_events}_seeds_${model_seeds}_${num_layers}_${num_blocks}_${hidden_features}_${num_bins}${bayes_tag}"
mkdir -p "${trial_dir}"


# === Submit array job with all parameters
array_range="0-$((${model_seeds}-1))"

array_job_id=$(sbatch \
  --parsable \
  --export=ALL,MODEL_SEEDS=${model_seeds},TRIAL_DIR=${trial_dir},DATA_PATH=${dataset_path},N_EPOCHS=${n_epochs},LR=${learning_rate},BATCH_SIZE=${batch_size},HIDDEN_FEATURES=${hidden_features},NUM_BLOCKS=${num_blocks},NUM_BINS=${num_bins},NUM_LAYERS=${num_layers} \
  --array=${array_range} \
  submit_array_NFzuko.sh | cut -d'_' -f1)

# === Submit collection job after array finishes
sbatch --job-name=NF_collect \
  --dependency=afterok:${array_job_id} \
  --output=${base_dir}/logs/collect_output_%j.out \
  --error=${base_dir}/logs/collect_error_%j.err \
  --time=00:10:00 \
  --mem=4G \
  --partition=standard \
  --wrap="source /work/gbadarac/miniforge3/bin/activate && conda activate nf_env && export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:\$PYTHONPATH && python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko.py --outdir ${trial_dir} --collect_all --num_models ${model_seeds}"