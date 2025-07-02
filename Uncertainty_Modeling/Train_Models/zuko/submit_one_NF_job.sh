#!/bin/bash
#SBATCH --job-name=NF_job
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/zuko/EstimationNFzuko_outputs/logs/job_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/zuko/EstimationNFzuko_outputs/logs/job_error_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=standard
# #SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G

# ======================================
# Activate environment
# ======================================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# ======================================
# Set data loading path
# ======================================
data_path=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Generate_Data/saved_generated_target_data/100k_target_training_set.npy

# ======================================
# Parameters 
# ======================================
seed=${SEED}
n_epochs=${N_EPOCHS}
learning_rate=${LR}
batch_size=${BATCH_SIZE}
hidden_features=${HIDDEN_FEATURES}
num_blocks=${NUM_BLOCKS}
num_bins=${NUM_BINS}
num_layers=${NUM_LAYERS}
bayesian=${BAYESIAN}

# ======================================
# Optional Bayesian flag
# ======================================
extra_flags=""
if [ "$bayesian" = true ] || [ "$bayesian" = True ]; then
  extra_flags="--bayesian"
fi

# ======================================
# Set output directories
# ======================================
outdir=${TRIAL_DIR}

# ======================================
# Ensure Python can find utils.py
# ======================================
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling:$PYTHONPATH

echo "Using GPU ID (as set by SLURM): $CUDA_VISIBLE_DEVICES"

# ======================================
# Run training script
# ======================================
export PYTHONUNBUFFERED=TRUE

python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/zuko/EstimationNFzuko.py \
    --data_path ${data_path} \
    --outdir ${outdir} \
    --seed ${seed} \
    --n_epochs ${n_epochs} \
    --learning_rate ${learning_rate} \
    --batch_size ${batch_size} \
    --hidden_features ${hidden_features} \
    --num_blocks ${num_blocks} \
    --num_bins ${num_bins} \
    --num_layers ${num_layers} \
    ${extra_flags}

export PYTHONUNBUFFERED=FALSE

echo "job submitted"

