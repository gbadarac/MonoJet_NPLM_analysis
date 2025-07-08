#!/bin/bash
#SBATCH --job-name=launcher_coverage_check
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/coverage_outputs/logs/launcher_coverage_check_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/coverage_outputs/logs/launcher_coverage_check_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
# #SBATCH --gres=gpu:1
# #SBATCH --account=gpu_gres
#SBATCH --partition=standard
#SBATCH --nodes=1

# ===============================
# Define the trial configuration
# ===============================
TRIAL_NAME="N_100000_seeds_60_bootstraps_1_4_16_128_15"
TRIAL_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/nflows/average_bootstraps/EstimationNFnflows_outputs/${TRIAL_NAME}"
COVERAGE_BASE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/coverage_outputs"
OUT_DIR_NAME="N_20000_seeds_60_bootstraps_1_4_16_128_15"
OUT_DIR="${COVERAGE_BASE}/${OUT_DIR_NAME}"
MU_TARGET_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Generate_Data/saved_generated_target_data/mu_target.npy"
MU_I_FILE="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/generate_sampled_means/results_generated_sampled_means/generated_sampled_mean_60_models_4_16_128_15.npy"

# Create output folders if needed
mkdir -p "${OUT_DIR}"
mkdir -p "${COVERAGE_BASE}/logs"

# Copy trial files to output dir so script runs in isolation
cp "$TRIAL_DIR"/f_i.pth "$OUT_DIR"/

# -----------------------
# Ensure Python can find utils.py
# -----------------------
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling:$PYTHONPATH

# ===============================
# Launch SLURM jobs for 300 toys
# ===============================
toy_seed=300
for seed in $(seq 0 $((toy_seed - 1))); do
    sbatch --export=ALL,TOY_SEED=${seed},TRIAL_DIR=${TRIAL_DIR},OUT_DIR=${OUT_DIR},MU_TARGET_PATH=${MU_TARGET_PATH},MU_I_FILE=${MU_I_FILE} submit_one_coverage_check.sh
done

echo "Submitted coverage jobs for trial: ${TRIAL_NAME}"

