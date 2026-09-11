#!/bin/bash
#SBATCH --job-name=nf_nflows
# Ensemble size is set HERE: use --array=0-(M-1). MODEL_SEEDS below auto-derives
# from the array count, so this is the ONLY line to change to resize the ensemble.
# --array=0-255 = 256-member production ensemble on the 4D embedding (arch
# L6/blk8/h128/bins10, validated end-to-end by the single-seed dry run).
# Use --array=0 for a single-seed dry run.
#SBATCH --array=0-255
# per-array-task budget (one flow member); the biggest 4D grid config finished <25 min
# at 24G, so 6h/32G is a safe per-member envelope and stays valid when you scale --array.
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd/logs/nf_%A_%a.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd/logs/nf_%A_%a.err

# ⚠ PART 2 TODO (loader symmetrization not finished): this trainer no longer
# produces f_i.pth — the ensemble IS the per-member model_*/ subdirs. Downstream
# loaders (fit_ensemble_weights.py nf branch, LRT.py, generate_hit_or_miss_NFs.py,
# coverage_check_*, NPLM/toy.py) STILL read f_i.pth and must be switched to glob
# model_*/model.pth before the NF wifi-fit / LRT / sampling steps will work.

# =============================
# Activate environment
# =============================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models:$PYTHONPATH

# =============================
# USER PARAMETERS (like the kernel submit)
# =============================
DATA_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/data/4d_embeddings/4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_train.npy"
NUM_FEATURES=4
MODEL_SEEDS=${SLURM_ARRAY_TASK_COUNT}   # ensemble size = number of array tasks (see --array above)

# Model hyperparameters
N_EPOCHS=1001
LR=1e-5          # match the grid search that selected this architecture (was 5e-6)
BATCH_SIZE=512
HIDDEN_FEATURES=128
NUM_BLOCKS=8
NUM_BINS=10
NUM_LAYERS=6

NFLOWS_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows"
ESTIMATOR="${NFLOWS_DIR}/EstimationNFnflows.py"
BASE_DIR="${NFLOWS_DIR}/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd"

# Trial dir (num_events from the data; deterministic -> identical across all tasks)
NUM_EVENTS=$(python -c "import numpy as np; print(np.load('${DATA_PATH}').shape[0])")
TRIAL_DIR="${BASE_DIR}/N_${NUM_EVENTS}_dim_${NUM_FEATURES}_seeds_${MODEL_SEEDS}_${NUM_LAYERS}_${NUM_BLOCKS}_${HIDDEN_FEATURES}_${NUM_BINS}"
mkdir -p "${TRIAL_DIR}" "${BASE_DIR}/logs"

# =============================
# Train this ensemble member
#   -> writes <TRIAL_DIR>/model_{SLURM_ARRAY_TASK_ID:03d}/model.pth (+ info.json).
#   The ensemble = these per-member subdirs (like the kernels' seed{NNN}/).
#   No collect/bundle step: downstream iterates model_*/ directly.
# =============================
echo "SLURM job id: $SLURM_JOB_ID, array task: $SLURM_ARRAY_TASK_ID / ${MODEL_SEEDS}"
echo "Trial dir: ${TRIAL_DIR}"

python "${ESTIMATOR}" \
    --data_path "${DATA_PATH}" \
    --outdir "${TRIAL_DIR}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --n_epochs "${N_EPOCHS}" \
    --learning_rate "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --hidden_features "${HIDDEN_FEATURES}" \
    --num_blocks "${NUM_BLOCKS}" \
    --num_bins "${NUM_BINS}" \
    --num_layers "${NUM_LAYERS}" \
    --num_features "${NUM_FEATURES}"
