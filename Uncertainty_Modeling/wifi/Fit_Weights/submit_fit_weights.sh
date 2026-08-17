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
NDIM=2          # 2 | 4  (NF fitter+plotter are dim-agnostic; only an NDIM with a TRIAL_DIR below runs. kernels is 2D-only here.)

# ─── Per-mode paths ───────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh

if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CONDA_ENV=kernels_env
    FOLDER_PATH="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_gmm/N_100000_dim_2_kernels_SparKer_models128_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035"
    DATA_PATH="$REPO_ROOT/data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_train.npy"
    N_WIFI=${N_WIFI:-128}   # ensemble size (<= 128 members); override per-job: sbatch --export=ALL,N_WIFI=64 ...
    trial_name=$(basename "$FOLDER_PATH")
    dataset_tag=$(basename "$(dirname "$FOLDER_PATH")")
    OUT_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernels/${trial_name}_${dataset_tag}_ensemblecomponents${N_WIFI}"
    EXTRA_ARGS="--folder_path $FOLDER_PATH"

elif [[ "$MODEL_TYPE" == "nf" ]]; then
    CONDA_ENV=nf_env
    N_WIFI=${N_WIFI:-128}   # ensemble size (<= available members); override per-job: sbatch --export=ALL,N_WIFI=64 ...
    # The NF fitter/plotter are dimension-agnostic; wiring a new NDIM only needs its
    # TRIAL_DIR (dir of model_*/model.pth members + architecture_config.json) + DATA_PATH.
    if [[ "$NDIM" == "2" ]]; then
        TRIAL_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_128_4_4_64_8"
        DATA_PATH="$REPO_ROOT/data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_train.npy"
    elif [[ "$NDIM" == "4" ]]; then
        # TODO: fill in once a 4D NF ensemble is trained in the per-member (model_*/model.pth) layout.
        TRIAL_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/<FILL_IN_4D_NF_ENSEMBLE>"
        DATA_PATH="$REPO_ROOT/data/4d_embeddings/<FILL_IN_4D_TARGET>.npy"
    else
        echo "Unknown NDIM=$NDIM for nf"; exit 1
    fi
    [[ -d "$TRIAL_DIR" ]] || { echo "NF trial dir for NDIM=$NDIM not found (not trained yet?): $TRIAL_DIR"; exit 1; }
    dataset_tag=$(basename "$(dirname "$TRIAL_DIR")")
    OUT_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/$(basename "$TRIAL_DIR")_${dataset_tag}_ensemblecomponents${N_WIFI}"
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
    $EXTRA_ARGS

echo "[$(date)] Done."
