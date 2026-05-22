#!/bin/bash
#SBATCH --job-name=LRT_one_model
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

# -------------------------
# Environment
# -------------------------
export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

. /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate kernels_env

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
PY="$REPO_ROOT/LRT/LRT_one_model.py"

# Train_Ensembles output with config.json + seed*/ histories
MODEL_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking"

# Which single model to use (0 -> seed000)
MODEL_SEED=53

# Output base
OUT_BASE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results"
mkdir -p "$OUT_BASE"
mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs

# -------------------------
# Mode (EDIT THIS)
# -------------------------
# CALIBRATION=1: samples Ntest events from the single GMM (no external file needed)
# CALIBRATION=0: loads target data from TARGET_DATA
CALIBRATION=1
TARGET_DATA="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/500k_2d_gaussian_heavy_tail_target_set.npy"

NTEST=100000
SEED_FORMAT="seed%03d"

FIRSTSEED=12345

# -------------------------
# Per-task variables
# -------------------------
TOY_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((FIRSTSEED + TOY_ID + 1))

cd "$REPO_ROOT"

echo "[$(date)] Task ${TOY_ID}, seed=${SEED}, model_seed=${MODEL_SEED}, host=${HOSTNAME}"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available(), "Device count:", torch.cuda.device_count())
PY

CMD=(python -u "$PY"
  --model_dir "$MODEL_DIR"
  --model_seed "$MODEL_SEED"
  --out_base "$OUT_BASE"
  --seed_format "$SEED_FORMAT"
  -n "$NTEST"
  -s "$SEED"
  --toy_id "$TOY_ID"
  -c "$CALIBRATION"
)

if [[ "$CALIBRATION" -eq 0 ]]; then
  CMD+=(--target_data "$TARGET_DATA")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
RC=$?

echo "[$(date)] Task ${TOY_ID} finished with exit code ${RC}"
exit $RC
