#!/bin/bash
#SBATCH --job-name=LRT_1model_profiling
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
PY="$REPO_ROOT/LRT/LRT_1model_profiling.py"

# Train_Ensembles output with config.json + seed*/ histories
MODEL_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/4_dim/4d_embedding_qcd/N_100000_dim_4_kernels_SparKer_models1_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm"
#MODEL_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking"

# Which single model to use (0 -> seed000)
MODEL_SEED=0

# Output base
OUT_BASE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/kernels/4d_embedding_qcd_joint_norm"
#OUT_BASE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/kernels/2d_no_masking_control"

mkdir -p "$OUT_BASE"
mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs

# -------------------------
# Mode (EDIT THIS)
# -------------------------
# CALIBRATION=1: samples Ntest events from the single GMM (no external file needed)
# CALIBRATION=0: loads target data from TARGET_DATA
# Can be overridden at submit time: sbatch --export=ALL,CALIBRATION=0 ...
CALIBRATION=${CALIBRATION:-1}
TARGET_DATA="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/data/4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_test.npy"
#TARGET_DATA="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy"


NTEST=100000
SEED_FORMAT="seed%03d"
FIRSTSEED=12345

# -------------------------
# Tilt regularization (EDIT / OVERRIDE AT SUBMIT TIME)
# -------------------------
# LAM_PERT = L2 ridge on the tilt coeffs b. Must be > 0: lam_pert=0 separates (tilt
# runaway under the null). lam_pert=1.0 keeps max|b|<1 (effective DOF ~35 2D / ~96 4D).
# CLIP_B is NON-functional (its L-BFGS-B path does not converge) — use LAM_PERT.
#   sbatch --export=ALL,LAM_PERT=0.1,CALIBRATION=1 submit_LRT_1model_profiling_toys.sh
LAM_PERT=${LAM_PERT:-1.0}
CLIP_B=${CLIP_B:-}                # empty => no box on b

# -------------------------
# Numerator kernel basis (EDIT / OVERRIDE AT SUBMIT TIME)
# -------------------------
# M and sigma drive test power. NPLM guideline: M >= sqrt(N)=316 for 100k. Each
# (M,sigma) is a NEW null -> recalibrate; keep sigma FIXED across a calib+test pair.
#   sbatch --export=ALL,N_KERNELS=500,KERNEL_SIGMA=0.5,CALIBRATION=1 submit_LRT_1model_profiling_toys.sh
N_KERNELS=${N_KERNELS:-100}
KERNEL_SIGMA=${KERNEL_SIGMA:-0.3}

# -------------------------
# Normalization Z (EDIT / OVERRIDE AT SUBMIT TIME)
# -------------------------
# Z is estimated by MC over a reference sample ~ DEN-optimal model (sample-Z). The
# finite reference inflates the null (conservative); verdict from the EMPIRICAL null.
N_REF=${N_REF:-}                  # override # reference points (default: Ntest, i.e. 1:1)

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
  --lam_pert "$LAM_PERT"
  --n_kernels "$N_KERNELS"
  --kernel_sigma "$KERNEL_SIGMA"
)

if [[ "$CALIBRATION" -eq 0 ]]; then
  CMD+=(--target_data "$TARGET_DATA")
fi

if [[ -n "$CLIP_B" ]]; then
  CMD+=(--clip_b "$CLIP_B")
fi

if [[ -n "$N_REF" ]]; then
  CMD+=(--n_ref "$N_REF")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
RC=$?

echo "[$(date)] Task ${TOY_ID} finished with exit code ${RC}"
exit $RC
