#!/bin/bash
#SBATCH --job-name=NF_launcher
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/EstimationNF_gaussians_bootstrap_outputs/logs/launcher_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/EstimationNF_gaussians_bootstrap_outputs/logs/launcher_error_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=2G

# ======================================
# USER PARAMETERS
# ======================================

# setup
model_seeds=2
bootstrap_runs=2

# Model hyperparameters
n_epochs=1001
learning_rate=5e-6
batch_size=512
hidden_features=64
num_blocks=4
num_bins=8
num_layers=4

# Base directory
base_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/EstimationNF_gaussians_bootstrap_outputs"

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

for seed in $(seq 0 $((model_seeds-1))); do
  for run_id in $(seq 0 $((bootstrap_runs-1))); do
    sbatch --export=ALL,MODEL_SEED=${seed},BOOTSTRAP_ID=${run_id},TRIAL_DIR=${trial_dir},N_EPOCHS=${n_epochs},LR=${learning_rate},BATCH_SIZE=${batch_size},HIDDEN_FEATURES=${hidden_features},NUM_BLOCKS=${num_blocks},NUM_BINS=${num_bins},NUM_LAYERS=${num_layers} submit_one_NF_job.sh
  done
done

