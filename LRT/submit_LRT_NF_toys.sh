#!/bin/bash
#SBATCH --job-name=LRT_NF
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

export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nf_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate nf_env

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

# ─── Paths (EDIT THESE) ──────────────────────────────────────────────
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/LRT/LRT.py"

# NF ensemble: use the simple 2D Gaussian ensemble (the one that works well,
# matches the NeurIPS paper setup: Z_frozen~4.13, Z_with_unc~0.92).
# The bimodal+heavy-tail NF ensemble is known to be a poor fit and is NOT used here.
NF_WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15"
FI_PATH="$NF_WIFI_DIR/f_i.pth"
ARCH_CONFIG="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_gaussian/N_100000_dim_2_seeds_60_4_16_128_15/architecture_config.json"

W_PATH="$NF_WIFI_DIR/w_i_fitted.npy"
W_COV_PATH="$NF_WIFI_DIR/cov_w.npy"

CALIB_DATA="$REPO_ROOT/Generate_Ensemble_Samples/Normalizing_Flows/saved_generated_NFs_ensemble_data/N_100000_dim_2_seeds_60_4_16_128_15/concatenated_ensemble_generated_samples_4_16_128_15_N_1000000.npy"
TARGET_DATA="$REPO_ROOT/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/500k_2d_gaussian_target_set.npy"
OUT_BASE="$REPO_ROOT/LRT/results"

# ─── Mode toggles (EDIT THESE) ───────────────────────────────────────
CALIBRATION=0
FIX_WIFI_WEIGHTS=false
FREE_WIFI_WEIGHTS=false

NTEST=100000
NENSEMBLE=60
FIRSTSEED=12345

mkdir -p "$OUT_BASE/logs"

TOY_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((FIRSTSEED + TOY_ID + 1))

cd "$REPO_ROOT"
echo "[$(date)] Task ${TOY_ID}, seed=${SEED}, host=${HOSTNAME}"

CMD=(python -u "$PY"
  --model_type nf
  --fi_path "$FI_PATH"
  --arch_config "$ARCH_CONFIG"
  --w_path "$W_PATH"
  --w_cov_path "$W_COV_PATH"
  --out_base "$OUT_BASE"
  -n "$NTEST"
  -e "$NENSEMBLE"
  -s "$SEED"
  --toy_id "$TOY_ID"
  -c "$CALIBRATION"
)

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
