#!/bin/bash
#SBATCH --job-name=LRT_toys
#SBATCH --array=0-0
#SBATCH --time=08:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
# #SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/results/logs/%x-%A_%a.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/results/logs/%x-%A_%a.err

set -euo pipefail

# -------------------------
# Environment
# -------------------------
export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

. /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate nf_env

CUDA_PATH=$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)
if [[ -n "$CUDA_PATH" ]]; then
  export CUDA_HOME=/usr/local/cuda-$CUDA_PATH
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# -------------------------
# Paths / config (EDIT THESE)
# -------------------------
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/LRT/Sparker_kernels/LRT.py"

# Train_Ensembles output with config.json + seed*/ histories
ENSEMBLE_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking"

# WiFi fitted weights (kernel WiFi)
W_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents60_no_bootstrapping/final_weights.npy"

W_COV_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents60_no_bootstrapping/cov_weights.npy"

# Output base (script will create subfolders inside)
OUT_BASE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/results"
mkdir -p "$OUT_BASE"
mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/results/logs

# Data paths
# If CALIBRATION=1, provide CALIB_DATA (dir of *.npy or single .npy)
# If CALIBRATION=0, provide TARGET_DATA (single .npy)
CALIBRATION=0
#CALIB_DATA="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/<PATH_TO_ENSEMBLE_GENERATED_SAMPLES_DIR_OR_FILE>"
TARGET_DATA="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/500k_2d_gaussian_heavy_tail_target_set.npy"

NTEST=100000
NENSEMBLE=60

# seed folders are seed000, seed001, ...
SEED_FORMAT="seed%03d"

FIRSTSEED=12345

# -------------------------
# Per-task variables
# -------------------------
TOY_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((FIRSTSEED + TOY_ID + 1))

cd "$REPO_ROOT"

echo "[$(date)] Task ${TOY_ID}, seed=${SEED}, host=${HOSTNAME}"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available(), "Device count:", torch.cuda.device_count())
PY

CMD=(python -u "$PY"
  --ensemble_dir "$ENSEMBLE_DIR"
  --w_path "$W_PATH"
  --w_cov_path "$W_COV_PATH"
  --out_base "$OUT_BASE"
  --seed_format "$SEED_FORMAT"
  -n "$NTEST"
  -e "$NENSEMBLE"
  -s "$SEED"
  -c "$CALIBRATION"
)

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