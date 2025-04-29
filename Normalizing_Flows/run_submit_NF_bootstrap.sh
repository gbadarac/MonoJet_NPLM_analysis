#!/bin/bash

# ======================================
# USER PARAMETERS
# ======================================

# Trial setup
n_seeds=2
n_runs=2

# Model hyperparameters
n_epochs=1001
learning_rate=5e-6
batch_size=512
hidden_features=64
num_blocks=4
num_bins=8
num_layers=4

# Base directory
base_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_gaussians_bootstrap_outputs"

# ======================================
# Create new trial folder automatically
# ======================================

trial_id=0
while [ -d "${base_dir}/trial_$(printf "%03d" ${trial_id})" ]; do
    trial_id=$((trial_id+1))
done
trial_dir=${base_dir}/trial_$(printf "%03d" ${trial_id})
mkdir -p ${trial_dir}
echo "Saving to trial ${trial_dir}"

# ======================================
# Submit jobs
# ======================================

for seed in $(seq 0 $((n_seeds-1))); do
  for run_id in $(seq 0 $((n_runs-1))); do
    sbatch --export=ALL,BOOTSTRAP_SEED=${seed},RUN_ID=${run_id},TRIAL_DIR=${trial_dir},N_EPOCHS=${n_epochs},LR=${learning_rate},BATCH_SIZE=${batch_size},HIDDEN_FEATURES=${hidden_features},NUM_BLOCKS=${num_blocks},NUM_BINS=${num_bins},NUM_LAYERS=${num_layers} submit_NF_bootstrap.sh
  done
done

