#!/bin/bash
#SBATCH --job-name=sparker_2gmm
#SBATCH --array=0-9
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# >>> adapt these two lines to your cluster if needed <<<
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/logs/sparker_%A_%a.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/logs/sparker_%A_%a.err

mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/logs

# =============================
# Activate environment
# =============================
source /work/gbadarac/miniforge3/bin/activate
conda activate kernels_env

# =============================
# USER PARAMETERS (like NF)
# =============================
DATA_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy"

BASE_OUTDIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail"
mkdir -p "${BASE_OUTDIR}"

N_LAYERS=15
KERNELS_PER_LAYER=10

N_MODELS=${SLURM_ARRAY_TASK_COUNT}

# =============================
# Run
# =============================
cd /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels

echo "SLURM job id: $SLURM_JOB_ID, array task: $SLURM_ARRAY_TASK_ID"
echo "Running seed = $SLURM_ARRAY_TASK_ID"

python EstimationKernels.py \
    --data_path "${DATA_PATH}" \
    --outdir "${BASE_OUTDIR}" \
    --seed "${SLURM_ARRAY_TASK_ID}"\
    --n_layers "${N_LAYERS}" \
    --kernels_per_layer "${KERNELS_PER_LAYER}" \
    --n_models "${N_MODELS}"
