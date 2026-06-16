#!/bin/bash
#SBATCH --job-name=generate_hit_or_miss_kernels
#SBATCH --array=0-199
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Samples/Sparker_kernels/saved_generated_kernel_ensemble_data/logs/job_out_%j.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Samples/Sparker_kernels/saved_generated_kernel_ensemble_data/logs/job_err_%j.err

set -euo pipefail

source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# ─── Paths (EDIT THESE) ──────────────────────────────────────────────
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/Generate_Ensemble_Samples/Sparker_kernels/generate_hit_or_miss_Sparker.py"

ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking"
W_PATH="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents32/w_i_fitted.npy"
OUT_DIR="$REPO_ROOT/Generate_Ensemble_Samples/Sparker_kernels/saved_generated_kernel_ensemble_data"

# ─── Sampling settings (EDIT THESE) ──────────────────────────────────
NGENERATE=5000   # events per seed; 200 seeds × 5000 = 1M total
NENSEMBLE=32
# Bounds matching the bimodal+heavy-tail target distribution:
#   x1: bimodal Gaussian (modes at -0.70 and -0.30, sigma=0.12)
#   x2: skew-normal (loc=1.0, scale=0.75, alpha=8.0, strong right skew)
BOUNDS="-1.5 0.5 0.5 4.5"

mkdir -p "$OUT_DIR/logs"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}, host=${HOSTNAME}"

python -u "$PY" \
  --ensemble_dir "$ENSEMBLE_DIR" \
  --w_path       "$W_PATH" \
  --out_dir      "$OUT_DIR" \
  -n             "$NGENERATE" \
  -e             "$NENSEMBLE" \
  -s             "$SLURM_ARRAY_TASK_ID" \
  --bounds       $BOUNDS

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} done."
