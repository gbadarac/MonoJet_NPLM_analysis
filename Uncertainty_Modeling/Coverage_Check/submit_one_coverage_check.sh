#!/bin/bash
#SBATCH --job-name=coverage_check
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/coverage_outputs/logs/coverage_check_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/coverage_outputs/logs/coverage_check_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Activate environment
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# Run the coverage check
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Coverage_Check/coverage_check.py \
  --toy_seed ${TOY_SEED} \
  --trial_dir ${TRIAL_DIR} \
  --out_dir ${OUT_DIR} \
  --mu_target_path ${MU_TARGET_PATH} \
  --mu_i_file ${MU_I_FILE} 