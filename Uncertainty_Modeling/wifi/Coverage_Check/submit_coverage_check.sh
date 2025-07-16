#!/bin/bash
#SBATCH --job-name=coverage_array
# #SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/coverage_outputs/logs/coverage_array_%A_%a.out
# #SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/coverage_outputs/logs/coverage_array_%A_%a.err
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-299     # ← Launch 300 jobs
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --nodes=1

# ===============================
# Define the trial configuration
# ===============================
TRIAL_NAME="N_100000_seeds_60_4_16_256_15"
TRIAL_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/${TRIAL_NAME}"
N_SAMPLED=110000       # Number of target samples for coverage test
COVERAGE_BASE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/coverage_outputs"
OUT_DIR_NAME="${TRIAL_NAME}_N_sampled_${N_SAMPLED}"
OUT_DIR="${COVERAGE_BASE}/${OUT_DIR_NAME}"
MU_TARGET_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/mu_target.npy"
MU_I_FILE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/generate_sampled_means/results_generated_sampled_means/generated_sampled_means_${TRIAL_NAME}.npy"

# Activate env
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# Create output folders if needed
mkdir -p "${OUT_DIR}"
mkdir -p "${COVERAGE_BASE}/logs"

# Copy trial files to output dir so script runs in isolation
cp "$TRIAL_DIR"/f_i.pth "$OUT_DIR"/

# -----------------------
# Ensure Python can find utils_wifi.py
# -----------------------
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi:$PYTHONPATH
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH

# Call script
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/coverage_check.py \
    --toy_seed ${SLURM_ARRAY_TASK_ID} \
    --trial_dir ${TRIAL_DIR} \
    --out_dir ${OUT_DIR} \
    --mu_target_path ${MU_TARGET_PATH} \
    --mu_i_file ${MU_I_FILE} \
    --n_points ${N_SAMPLED}