#!/bin/bash
#SBATCH --job-name=nf_gridsearch
# "even smaller" factorial: num_layers{2,4} x num_blocks{4,8} x hidden{32,64} x num_bins{8,10}
# -> 16 configs, ALL strictly smaller than the previous winner (L4/blk16/h128/bins15).
# One seed (0) each. The eval globs ALL config dirs, so the previous 12 (larger) stay in
# the unified ranking as the reference. Change --array to 0-(N-1) if you edit the configs.
#SBATCH --array=0-15
# small footprint on purpose: a single 2D flow needs little -> highly backfillable,
# so these slot into gaps behind big arrays instead of waiting for the whole queue.
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/gridsearch/logs/grid_%A_%a.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/gridsearch/logs/grid_%A_%a.err

# =============================
# Environment
# =============================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models:$PYTHONPATH

# =============================
# Fixed settings
# =============================
DATA_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/data_train.npy"
NUM_FEATURES=2
LR=1e-5
N_EPOCHS=1001
BATCH_SIZE=512

NFLOWS_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows"
ESTIMATOR="${NFLOWS_DIR}/EstimationNFnflows.py"
GRID_BASE="${NFLOWS_DIR}/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/gridsearch"

# =============================
# Grid: "num_layers num_blocks hidden_features num_bins"
# "even smaller" corner (all below the winner L4/blk16/h128/bins15):
#   layers{2,4} x blocks{4,8} x hidden{32,64} x bins{8,10}
# =============================
CONFIGS=(
  "2 4 32 8"  "2 4 32 10"  "2 4 64 8"  "2 4 64 10"
  "2 8 32 8"  "2 8 32 10"  "2 8 64 8"  "2 8 64 10"
  "4 4 32 8"  "4 4 32 10"  "4 4 64 8"  "4 4 64 10"
  "4 8 32 8"  "4 8 32 10"  "4 8 64 8"  "4 8 64 10"
)
read NUM_LAYERS NUM_BLOCKS HIDDEN_FEATURES NUM_BINS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
TAG="L${NUM_LAYERS}_blk${NUM_BLOCKS}_h${HIDDEN_FEATURES}_bins${NUM_BINS}"
OUTDIR="${GRID_BASE}/${TAG}"
mkdir -p "${OUTDIR}" "${GRID_BASE}/logs"

echo "arch: layers=${NUM_LAYERS} blocks=${NUM_BLOCKS} hidden=${HIDDEN_FEATURES} bins=${NUM_BINS}  lr=${LR}"
echo "outdir: ${OUTDIR}"

# --seed 0 => same bootstrap/val split for every architecture (fair comparison).
# Writes <OUTDIR>/model_000/model.pth + <OUTDIR>/architecture_config.json.
python "${ESTIMATOR}" \
    --data_path "${DATA_PATH}" \
    --outdir "${OUTDIR}" \
    --seed 0 \
    --n_epochs "${N_EPOCHS}" \
    --learning_rate "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --hidden_features "${HIDDEN_FEATURES}" \
    --num_blocks "${NUM_BLOCKS}" \
    --num_bins "${NUM_BINS}" \
    --num_layers "${NUM_LAYERS}" \
    --num_features "${NUM_FEATURES}"
