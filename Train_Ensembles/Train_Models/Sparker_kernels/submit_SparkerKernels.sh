#!/bin/bash
#SBATCH --job-name=sparker_kernels
#SBATCH --array=0-0
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# >>> adapt these two lines to your cluster if needed <<<
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/4_dim/4d_embedding_qcd/logs/sparker_%A_%a.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/4_dim/4d_embedding_qcd/logs/sparker_%A_%a.err

mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/4_dim/4d_embedding_qcd/logs

# =============================
# Activate environment
# =============================
source /work/gbadarac/miniforge3/bin/activate
conda activate kernels_env

# =============================
# USER PARAMETERS (like NF)
# =============================
DATA_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/data/4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_train.npy"

BASE_OUTDIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/4_dim/4d_embedding_qcd"
mkdir -p "${BASE_OUTDIR}"

# 4D-QCD Option A: rebalanced toward the fine layers (peak-builders) vs the 2D
# schedule "80,70,60,50,40". Coarse layer halved (can't hog mass), fine layer ~4x
# (resolves the sharp F3/F2/F4 peaks). Total 500 kernels (~1.7x the old 300).
CENTROIDS_PER_LAYER="40,60,90,130,180"
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
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --centroids_per_layer "${CENTROIDS_PER_LAYER}" \
    --n_models "${N_MODELS}"
