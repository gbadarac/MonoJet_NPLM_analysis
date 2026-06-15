#!/bin/bash
#SBATCH --job-name=fit_weights
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/logs/fit_weights_%j.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/logs/fit_weights_%j.err

set -euo pipefail

REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# ─── Mode toggles (EDIT THESE) ───────────────────────────────────────────────
MODEL_TYPE=nf   # kernels | nf
NDIM=2          # 2 | 4  (kernels only supports 2; nf supports both)

# ─── Common optimiser settings ────────────────────────────────────────────────
EPOCHS=2000
LR=0.1
PATIENCE=10

# ─── Per-mode paths ───────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh

if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CONDA_ENV=kernels_env
    FOLDER_PATH="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking"
    DATA_PATH="$REPO_ROOT/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy"
    N_WIFI=32
    trial_name=$(basename "$FOLDER_PATH")
    dataset_tag=$(basename "$(dirname "$FOLDER_PATH")")
    OUT_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/${trial_name}_${dataset_tag}_ensemblecomponents${N_WIFI}"
    EXTRA_ARGS="--folder_path $FOLDER_PATH"

elif [[ "$MODEL_TYPE" == "nf" ]]; then
    CONDA_ENV=nf_env
    N_WIFI=60
    if [[ "$NDIM" == "2" ]]; then
        TRIAL_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_60_4_16_128_15"
        DATA_PATH="$REPO_ROOT/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy"
        dataset_tag=$(basename "$(dirname "$TRIAL_DIR")")
        OUT_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/$(basename "$TRIAL_DIR")_${dataset_tag}"
    elif [[ "$NDIM" == "4" ]]; then
        TRIAL_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/N_100000_dim_4_seeds_60_4_16_128_15"
        DATA_PATH="$REPO_ROOT/Train_Ensembles/Generate_Data/saved_generated_target_data/4_dim/100k_4d_gaussian_target_set.npy"
        OUT_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/$(basename "$TRIAL_DIR")"
    else
        echo "Unknown NDIM=$NDIM for nf"; exit 1
    fi
    EXTRA_ARGS="--trial_dir $TRIAL_DIR"

else
    echo "Unknown MODEL_TYPE=$MODEL_TYPE"; exit 1
fi

conda activate "$CONDA_ENV"
mkdir -p "$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/logs"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"
echo "[$(date)] MODEL_TYPE=${MODEL_TYPE}, NDIM=${NDIM}, N_WIFI=${N_WIFI}, host=${HOSTNAME}"

python -u Uncertainty_Modeling/wifi/Fit_Weights/fit_ensemble_weights.py \
    --model_type        "$MODEL_TYPE" \
    --data_path         "$DATA_PATH" \
    --out_dir           "$OUT_DIR" \
    --n_wifi_components "$N_WIFI" \
    --epochs            "$EPOCHS" \
    --patience          "$PATIENCE" \
    --lr                "$LR" \
    $EXTRA_ARGS

echo "[$(date)] Done."
