#!/bin/bash
#SBATCH --job-name=LRT_100toys
#SBATCH --array=0-99
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
# #SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/results/logs/%x-%A_%a.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/results/logs/%x-%A_%a.err

# -------------------------
# Environment
# -------------------------
export LD_LIBRARY_PATH=/work/gbadarac/miniforge3/envs/nplm_env/lib:$LD_LIBRARY_PATH
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh && conda activate nf_env

# Optional: set CUDA_HOME if torch reports a CUDA version
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

# PyTorch CUDA allocator tweak
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# Make sure Python finds your helpers
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH

# -------------------------
# Static config
# -------------------------
CALIBRATION=False
BASE_OUT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/results/N_100000_dim_2_seeds_60_4_16_128_15_toys_100_N_sampled_100k_20_kernels_no_softmax"

w="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15_trial/w_i_fitted.npy"
w_cov="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15_trial/cov_w.npy"
hit_or_miss_data="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data/N_100000_dim_2_seeds_60_4_16_128_15/concatenated_ensemble_generated_samples_4_16_128_15.npy"
ensemble_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/2_dim/N_100000_dim_2_seeds_60_4_16_128_15"

PY=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/toys_LRT_with_unc.py

# -------------------------
# Per-task variables
# -------------------------
TOY_ID=${SLURM_ARRAY_TASK_ID}
OUT_DIR="${BASE_OUT}"   # Python will append /<mode>/toy_<seed>
mkdir -p "$OUT_DIR"
mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/results/logs

echo "[$(date)] Task ${TOY_ID} starting on ${HOSTNAME}"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available(), "Device count:", torch.cuda.device_count())
PY

# -------------------------
# Build and run command
# (Your Python uses -t/--toys as the seed.)
# -------------------------
CMD=(python -u "$PY"
     -t "$TOY_ID"
     -c "$CALIBRATION"
     --out_dir "$OUT_DIR"
     --w_path "$w"
     --w_cov_path "$w_cov"
     --ensemble_dir "$ensemble_dir")

if [[ "$CALIBRATION" == "True" || "$CALIBRATION" == "true" ]]; then
  CMD+=(--hit_or_miss_data "$hit_or_miss_data")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
RC=$?

echo "[$(date)] Task ${TOY_ID} finished with exit code ${RC}"
exit $RC
