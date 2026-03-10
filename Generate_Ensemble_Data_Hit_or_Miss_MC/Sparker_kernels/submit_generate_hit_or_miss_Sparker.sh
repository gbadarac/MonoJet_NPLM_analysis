#!/bin/bash
#SBATCH --job-name=generate_hit_or_miss_kernels
#SBATCH --array=1-199
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/Sparker_kernels/saved_generated_kernel_ensemble_data/logs/job_out_%j.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/Sparker_kernels/saved_generated_kernel_ensemble_data/logs/job_err_%j.err

# -------------------------
# Environment
# -------------------------
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# -------------------------
# Paths
# -------------------------
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/Generate_Ensemble_Data_Hit_or_Miss_MC/Sparker_kernels/generate_hit_or_miss_Sparker.py"

ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking"

W_PATH="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernel/N_100000_dim_2_kernels_SparKer_models60_L5_K75_M270_Nboot100000_lr0.05_clip_10000000_no_masking_2d_bimodal_gaussian_heavy_tail_ensemblecomponents60_fix_normalization/final_weights.npy"

OUT_DIR="$REPO_ROOT/Generate_Ensemble_Data_Hit_or_Miss_MC/Sparker_kernels/saved_generated_kernel_ensemble_data"

NGENERATE=5000
NENSEMBLE=60

# -------------------------
# Create log dir
# -------------------------
mkdir -p "$OUT_DIR/logs"

# -------------------------
# Run
# -------------------------
echo "[$(date)] Job ${SLURM_ARRAY_TASK_ID}, seed=${SLURM_ARRAY_TASK_ID}, host=${HOSTNAME}"

python -u "$PY" \
  --ensemble_dir "$ENSEMBLE_DIR" \
  --w_path       "$W_PATH" \
  --out_dir      "$OUT_DIR" \
  -n             "$NGENERATE" \
  -e             "$NENSEMBLE" \
  -s             "$SLURM_ARRAY_TASK_ID"

echo "[$(date)] Job ${SLURM_ARRAY_TASK_ID} done."
