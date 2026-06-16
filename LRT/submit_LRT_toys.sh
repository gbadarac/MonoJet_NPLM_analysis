#!/bin/bash
#SBATCH --job-name=LRT
#SBATCH --array=0-99
#SBATCH --time=08:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs/%x-%A_%a.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs/%x-%A_%a.err

set -euo pipefail

# ─── Mode toggles (EDIT THESE) ───────────────────────────────────────
MODEL_TYPE=kernels   # kernels | nf

CALIBRATION=1        # 1 = null toys (SIR + calib pool)  |  0 = observed (target data)
FIX_WIFI_WEIGHTS=false
FREE_WIFI_WEIGHTS=false

NTEST=100000
FIRSTSEED=12345

# ─── Per-model paths (EDIT THESE) ────────────────────────────────────
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/LRT/LRT.py"

if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CONDA_ENV=kernels_env
    NENSEMBLE=32
    ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking"
    W_PATH="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents32/w_i_fitted.npy"
    W_COV_PATH="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents32/cov_w.npy"
    CALIB_DATA="$REPO_ROOT/Generate_Ensemble_Samples/Sparker_kernels/saved_generated_kernel_ensemble_data/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents32"
    TARGET_DATA="$REPO_ROOT/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/500k_2d_gaussian_heavy_tail_target_set.npy"

elif [[ "$MODEL_TYPE" == "nf" ]]; then
    CONDA_ENV=nf_env
    NENSEMBLE=60
    NF_TRAIN_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_gaussian/N_100000_dim_2_seeds_60_4_16_128_15"
    NF_WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15_2d_gaussian"
    FI_PATH="$NF_TRAIN_DIR/f_i.pth"
    ARCH_CONFIG="$NF_TRAIN_DIR/architecture_config.json"
    W_PATH="$NF_WIFI_DIR/w_i_fitted.npy"
    W_COV_PATH="$NF_WIFI_DIR/cov_w.npy"
    CALIB_DATA="$REPO_ROOT/Generate_Ensemble_Samples/Normalizing_Flows/saved_generated_NFs_ensemble_data/N_100000_dim_2_seeds_60_4_16_128_15_2d_gaussian"
    TARGET_DATA="$REPO_ROOT/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/500k_2d_gaussian_target_set.npy"

else
    echo "Unknown MODEL_TYPE=$MODEL_TYPE"; exit 1
fi

# ─── Environment ─────────────────────────────────────────────────────
export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

CUDA_PATH=$(python - <<'PY'
import torch; print(torch.version.cuda or "")
PY
)
if [[ -n "$CUDA_PATH" ]]; then
  export CUDA_HOME=/usr/local/cuda-$CUDA_PATH
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# ─── Run ─────────────────────────────────────────────────────────────
OUT_BASE="$REPO_ROOT/LRT/results"
mkdir -p "$OUT_BASE/logs"

TOY_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((FIRSTSEED + TOY_ID + 1))

cd "$REPO_ROOT"
echo "[$(date)] MODEL_TYPE=${MODEL_TYPE}, CALIBRATION=${CALIBRATION}, Task=${TOY_ID}, seed=${SEED}, host=${HOSTNAME}"

CMD=(python -u "$PY"
  --model_type    "$MODEL_TYPE"
  --w_path        "$W_PATH"
  --w_cov_path    "$W_COV_PATH"
  --out_base      "$OUT_BASE"
  -n              "$NTEST"
  -e              "$NENSEMBLE"
  -s              "$SEED"
  --toy_id        "$TOY_ID"
  -c              "$CALIBRATION"
)

if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CMD+=(--ensemble_dir "$ENSEMBLE_DIR" --seed_format "seed%03d")
elif [[ "$MODEL_TYPE" == "nf" ]]; then
    CMD+=(--fi_path "$FI_PATH" --arch_config "$ARCH_CONFIG")
fi

[[ "$FIX_WIFI_WEIGHTS"  == "true" ]] && CMD+=(--fix_wifi_weights)
[[ "$FREE_WIFI_WEIGHTS" == "true" ]] && CMD+=(--free_wifi_weights)

if [[ "$CALIBRATION" -eq 1 ]]; then
  CMD+=(--calib_data "$CALIB_DATA")
else
  CMD+=(--target_data "$TARGET_DATA")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
RC=$?
echo "[$(date)] Task ${TOY_ID} finished with exit code ${RC}"
exit $RC
