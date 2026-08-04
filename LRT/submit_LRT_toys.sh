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

CALIBRATION=${CALIBRATION:-1}              # 1 = null toys (SIR + calib pool)  |  0 = observed (target data)
FIX_WIFI_WEIGHTS=${FIX_WIFI_WEIGHTS:-false}   # respects sbatch --export override
FREE_WIFI_WEIGHTS=${FREE_WIFI_WEIGHTS:-false} # respects sbatch --export override

NTEST=100000
FIRSTSEED=12345

# ─── Numerator form (EDIT / OVERRIDE AT SUBMIT TIME) ─────────────────
# NUMERATOR=additive       : f_ens + sum c_j G_j (current; convex; SGD).
# NUMERATOR=multiplicative : f_ens*exp(tau)/Z (NPLM exp-tilt; matches LRT_one_model
#                            + classifier; log-space, clean chi2). Currently requires
#                            FROZEN weights (FIX_WIFI_WEIGHTS=true); Z_MODE defaults to
#                            sample (importance from q; any d) — set Z_MODE=grid only for
#                            a 2D cross-check. Constrained/free (joint w,b) is the next step.
#   sbatch --export=ALL,NUMERATOR=multiplicative,FIX_WIFI_WEIGHTS=true,CALIBRATION=1 submit_LRT_toys.sh
NUMERATOR=${NUMERATOR:-additive}
N_KERNELS=${N_KERNELS:-100}
KERNEL_SIGMA=${KERNEL_SIGMA:-0.3}
LAM_PERT=${LAM_PERT:-1.0}          # [multiplicative] L2 ridge on tilt coeffs b
Z_MODE=${Z_MODE:-sample}           # [multiplicative] sample (importance from q; any d, DEFAULT) | grid (2D cross-check)
GRID_POINTS=${GRID_POINTS:-300}    # [multiplicative grid] points per dimension
GRID_PAD=${GRID_PAD:-0.2}          # [multiplicative grid] padding beyond data range
N_REF=${N_REF:-}                   # [multiplicative sample] # importance refs (default Ntest; >>Ntest in 4D)
# 4D: set NUMERATOR=multiplicative, Z_MODE=sample, FIX_WIFI_WEIGHTS=true; the multiplicative
# path builds its own null (SIR-from-q) so CALIB_DATA is ignored (no 4D pool needed).

# ─── Per-model paths (EDIT THESE) ────────────────────────────────────
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/LRT/LRT.py"

if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CONDA_ENV=kernels_env
    NENSEMBLE=110
    ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/4_dim/4d_embedding_qcd/N_100000_dim_4_kernels_SparKer_models160_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm"
    W_PATH="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernels/N_100000_dim_4_kernels_SparKer_models160_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_4d_embedding_qcd_ensemblecomponents110/w_i_fitted.npy"
    W_COV_PATH="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernels/N_100000_dim_4_kernels_SparKer_models160_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_4d_embedding_qcd_ensemblecomponents110/cov_w.npy"
    CALIB_DATA=""   # unused by the multiplicative path (self-generated SIR-from-q null); only the additive path reads it
    TARGET_DATA="$REPO_ROOT/data/4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_test.npy"

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
  --numerator     "$NUMERATOR"
  --n_kernels     "$N_KERNELS"
  --kernel_sigma  "$KERNEL_SIGMA"
)

if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CMD+=(--ensemble_dir "$ENSEMBLE_DIR" --seed_format "seed%03d")
elif [[ "$MODEL_TYPE" == "nf" ]]; then
    CMD+=(--fi_path "$FI_PATH" --arch_config "$ARCH_CONFIG")
fi

if [[ "$NUMERATOR" == "multiplicative" ]]; then
    CMD+=(--lam_pert "$LAM_PERT" --z_mode "$Z_MODE" --grid_points "$GRID_POINTS" --grid_pad "$GRID_PAD")
    [[ -n "$N_REF" ]] && CMD+=(--n_ref "$N_REF")
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
