#!/bin/bash
#SBATCH --job-name=nf_gridsearch
# 4D embedding (JetClass QCD) architecture scan: a TINY ladder of LARGER archs than
# the 2D winner (L4/blk4/h64/bins8), since 4D has cross-feature correlations that the
# 2D product-of-marginals toy did not. NB the 2D grid already showed that maxing out
# (blk16/h256/bins64) OVERFITS -> we scan UPWARD from the 2D winner, not to the extreme.
# One seed (0) each -> same bootstrap/val split for a fair comparison. eval_gridsearch.py
# globs the config dirs and ranks by held-out (seed-42 test) NLL. Change --array to
# 0-(N-1) if you edit the configs below.
#SBATCH --array=0-5
# single-seed flows are small -> highly backfillable; slot into gaps behind big arrays.
# Empirically the biggest 2D config finished in <25 min; 3h gives comfortable 4D margin.
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd/gridsearch/logs/grid_%A_%a.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd/gridsearch/logs/grid_%A_%a.err

# =============================
# Environment
# =============================
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models:$PYTHONPATH

# =============================
# Fixed settings
# =============================
DATA_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/data/4d_embeddings/4d_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_train.npy"
NUM_FEATURES=4
LR=1e-5
N_EPOCHS=1001
BATCH_SIZE=512

NFLOWS_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows"
ESTIMATOR="${NFLOWS_DIR}/EstimationNFnflows.py"
GRID_BASE="${NFLOWS_DIR}/EstimationNFnflows_outputs/4_dim/4d_embedding_qcd/gridsearch"

# =============================
# Grid: "num_layers num_blocks hidden_features num_bins"
# Tiny ladder of LARGER archs than the 2D winner (L4/blk4/h64/bins8), scanning UPWARD:
#   #0 is the 2D winner as a baseline anchor (does scaling up actually help in 4D?);
#   we add layers (dim mixing via the interleaved ReversePermutations) before brute
#   width, and cap below the 2D "worse" extreme (blocks<=8, hidden<=256, bins<=12).
# =============================
CONFIGS=(
  "4 4 64 8"     # 2D winner -- baseline anchor
  "4 8 128 8"    # wider + deeper blocks
  "6 8 128 10"   # +layers (dim mixing) +bins
  "8 8 128 10"   # more layers
  "6 8 256 10"   # wider
  "8 8 256 12"   # largest of the ladder
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
