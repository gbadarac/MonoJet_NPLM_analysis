#!/bin/bash
#SBATCH --job-name=fit_weights
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/logs/fit_weights_%j.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/logs/fit_weights_%j.err

set -euo pipefail

REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
FW="Uncertainty_Modeling/wifi/Fit_Weights"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh

# ─── Mode toggles (EDIT THESE) ───────────────────────────────────────────────
MODEL_TYPE=nf          # kernels | nf
NDIM=2                 # 2 | 4   (nf only; kernels is 2D here)
SELECTION=uniform      # how the N_WIFI members are chosen:
                       #   uniform    : draw N_WIFI seeds at random from all members  (kernels + nf)
                       #   stratified : (worst,medium,best) draw per M from the marginal ranking (nf)
                       #   pinned     : first N_WIFI of the ranking, best->worst      (nf)
                       #   default    : seed000.. / model_000.. in order (no seed list)
N_WIFI=${N_WIFI:-128}  # ensemble size; per-job: sbatch --export=ALL,N_WIFI=8 ...
RNG_SEED=${RNG_SEED:-0}  # draw seed for uniform/stratified (reproducible; change for another draw)

# ─── The ONLY model-type-specific block: env + paths + member layout ─────────
if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CONDA_ENV=kernels_env
    MODEL_TAG=kernels
    MEMBER_FLAG="--folder_path"          # fitter arg pointing at the ensemble dir
    MEMBER_GLOB="seed[0-9][0-9][0-9]"    # member subdir pattern (for the uniform pool count)
    RANKING_FILE=""                      # kernels have no marginal ranking -> uniform/default only
    ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_gmm/N_100000_dim_2_kernels_SparKer_models128_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035"
    DATA_PATH="$REPO_ROOT/data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_train.npy"
elif [[ "$MODEL_TYPE" == "nf" ]]; then
    CONDA_ENV=nf_env
    MODEL_TAG=NF
    MEMBER_FLAG="--trial_dir"
    MEMBER_GLOB="model_*"
    RANKING_FILE="$REPO_ROOT/$FW/results_fit_weights_NF_uniform/2d_toymodel/seed_ranking_marginal.txt"  # 2D-toy NF ranking (stratified/pinned only; uniform ignores it)
    if [[ "$NDIM" == "2" ]]; then
        ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_128_4_4_64_8"
        DATA_PATH="$REPO_ROOT/data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_train.npy"
    elif [[ "$NDIM" == "4" ]]; then
        # TODO: fill in once a 4D NF ensemble is trained in the per-member (model_*/model.pth) layout.
        # Keep the SAME 3-level depth as 2D (.../<N>_dim/<dataset>/<ensemble>) so the GROUP
        # derivation below reads "4_dim" and results land under .../4d_embedding/.
        ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/<FILL_IN_4D_DATASET>/<FILL_IN_4D_NF_ENSEMBLE>"
        DATA_PATH="$REPO_ROOT/data/4d_embeddings/<FILL_IN_4D_TARGET>.npy"
    else
        echo "Unknown NDIM=$NDIM for nf"; exit 1
    fi
else
    echo "Unknown MODEL_TYPE=$MODEL_TYPE (kernels|nf)"; exit 1
fi
[[ -d "$ENSEMBLE_DIR" ]] || { echo "ensemble dir not found (not trained yet?): $ENSEMBLE_DIR"; exit 1; }

# ─── Common: output dir, env, workdir (uniform naming across model types) ─────
dataset_tag=$(basename "$(dirname "$ENSEMBLE_DIR")")
# Group results by dimensionality so 2D-toy and 4D-embedding runs never mix. Derived
# from the ensemble dir's "<N>_dim" component (robust to the NDIM toggle).
dim_dir=$(basename "$(dirname "$(dirname "$ENSEMBLE_DIR")")")   # "2_dim" | "4_dim"
case "$dim_dir" in
    2_dim) GROUP=2d_toymodel ;;
    4_dim) GROUP=4d_embedding ;;
    *) echo "cannot derive result group from ensemble dir dim component '$dim_dir'"; exit 1 ;;
esac
OUT_DIR="$REPO_ROOT/$FW/results_fit_weights_${MODEL_TAG}_${SELECTION}/${GROUP}/$(basename "$ENSEMBLE_DIR")_${dataset_tag}_ensemblecomponents${N_WIFI}"
conda activate "$CONDA_ENV"
mkdir -p "$REPO_ROOT/$FW/logs" "$OUT_DIR"
cd "$REPO_ROOT"
echo "[$(date)] MODEL_TYPE=$MODEL_TYPE NDIM=$NDIM SELECTION=$SELECTION N_WIFI=$N_WIFI host=$HOSTNAME"

# ─── Resolve the member seeds once (model-type agnostic) ─────────────────────
SEED_ARGS=""
SINGLE_SEED="0"   # used only when N_WIFI=1 (default falls back to seed 0)
if [[ "$SELECTION" != "default" ]]; then
    if [[ "$SELECTION" == "uniform" ]]; then
        N_MEMBERS=$(find "$ENSEMBLE_DIR" -maxdepth 1 -type d -name "$MEMBER_GLOB" | wc -l)
        POOL_ARG="--n_members $N_MEMBERS"
    else
        [[ -n "$RANKING_FILE" && -f "$RANKING_FILE" ]] || {
            echo "SELECTION=$SELECTION needs a marginal ranking (NF-only); use SELECTION=uniform for $MODEL_TYPE"; exit 1; }
        POOL_ARG="--ranking_file $RANKING_FILE"
    fi
    MEMBER_SEEDS=$(python "$FW/select_member_seeds.py" --mode "$SELECTION" $POOL_ARG \
        --M "$N_WIFI" --rng_seed "$RNG_SEED")
    echo "[select] $SELECTION M=$N_WIFI rng_seed=$RNG_SEED -> $MEMBER_SEEDS"
    SINGLE_SEED="${MEMBER_SEEDS%%,*}"                 # first seed (for N_WIFI=1)
    [[ "$N_WIFI" != "1" ]] && SEED_ARGS="--member_seeds $MEMBER_SEEDS"
fi

# ─── Run ─────────────────────────────────────────────────────────────────────
if [[ "$N_WIFI" == "1" ]]; then
    # Single component: no ensemble, no weight fit, no uncertainty band.
    [[ "$MODEL_TYPE" == "nf" ]] || { echo "N_WIFI=1 single-component plot is NF-only"; exit 1; }
    echo "[single-seed] N_WIFI=1 -> seed $SINGLE_SEED, marginal WITHOUT uncertainty band"
    python -u "$FW/plot_single_seed_marginal.py" \
        --trial_dir "$ENSEMBLE_DIR" --seed "$SINGLE_SEED" \
        --data_path "$DATA_PATH" --out_dir "$OUT_DIR"
else
    python -u "$FW/fit_ensemble_weights.py" \
        --model_type "$MODEL_TYPE" --data_path "$DATA_PATH" \
        --out_dir "$OUT_DIR" --n_wifi_components "$N_WIFI" \
        $MEMBER_FLAG "$ENSEMBLE_DIR" $SEED_ARGS
fi

echo "[$(date)] Done."
