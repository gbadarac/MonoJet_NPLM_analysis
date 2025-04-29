#!/bin/bash
#SBATCH --job-name=bootstrap_nf
#SBATCH --output=logs/job_output_%j.out
#SBATCH --error=logs/job_error_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G

# ======================================
# Activate environment
# ======================================
source /t3home/gbadarac/miniforge3/bin/activate my_project

# ======================================
# Parameters 
# ======================================

bootstrap_seed=${BOOTSTRAP_SEED}
run_id=${RUN_ID}
trial_dir=${TRIAL_DIR}

n_epochs=${N_EPOCHS}
learning_rate=${LR}
batch_size=${BATCH_SIZE}
hidden_features=${HIDDEN_FEATURES}
num_blocks=${NUM_BLOCKS}
num_bins=${NUM_BINS}
num_layers=${NUM_LAYERS}

# ======================================
# Set output directories
# ======================================

job_outdir=${trial_dir}/seed_$(printf "%04d" ${bootstrap_seed})/run_$(printf "%02d" ${run_id})
mkdir -p ${job_outdir}

# ======================================
# Run training script
# ======================================

export PYTHONUNBUFFERED=TRUE
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows_gaussians_bootstrap.py \
    --n_epochs ${n_epochs} \
    --learning_rate ${learning_rate} \
    --batch_size ${batch_size} \
    --outdir ${trial_dir} \
    --hidden_features ${hidden_features} \
    --num_blocks ${num_blocks} \
    --num_bins ${num_bins} \
    --num_layers ${num_layers} \
    --bootstrap_seed ${bootstrap_seed} \
    --run_id ${run_id}
export PYTHONUNBUFFERED=FALSE

# ======================================
# Move logs into job folder
# ======================================

mv ./logs/job_output_${SLURM_JOB_ID}.out ${job_outdir}
mv ./logs/job_error_${SLURM_JOB_ID}.err ${job_outdir}

