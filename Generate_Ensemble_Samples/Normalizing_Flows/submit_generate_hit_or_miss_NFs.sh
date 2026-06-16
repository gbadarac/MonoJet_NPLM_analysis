#!/bin/bash
#SBATCH --job-name=generate_hit_or_miss_NFs
#SBATCH --array=0-199
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Samples/Normalizing_Flows/saved_generated_NFs_ensemble_data/logs/job_out_%j.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Samples/Normalizing_Flows/saved_generated_NFs_ensemble_data/logs/job_err_%j.err

set -euo pipefail

source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# ─── Paths (EDIT THESE) ──────────────────────────────────────────────
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/Generate_Ensemble_Samples/Normalizing_Flows/generate_hit_or_miss_NFs.py"

TRIAL_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_gaussian/N_100000_dim_2_seeds_60_4_16_128_15"
W_PATH="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15_2d_gaussian/w_i_fitted.npy"
OUT_DIR="$REPO_ROOT/Generate_Ensemble_Samples/Normalizing_Flows/saved_generated_NFs_ensemble_data"

# ─── Sampling settings (EDIT THESE) ──────────────────────────────────
NGENERATE=5000     # events per seed; 200 seeds × 5000 = 1M total
TAIL_BOUND=3.0     # hit-or-miss bounds: [-TAIL_BOUND, TAIL_BOUND] per dim

mkdir -p "$OUT_DIR/logs"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}, host=${HOSTNAME}"

python -u "$PY" \
  --trial_dir   "$TRIAL_DIR" \
  --w_path      "$W_PATH" \
  --out_dir     "$OUT_DIR" \
  -n            "$NGENERATE" \
  -s            "$SLURM_ARRAY_TASK_ID" \
  --tail_bound  "$TAIL_BOUND"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} done."
