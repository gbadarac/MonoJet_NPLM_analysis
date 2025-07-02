#!/bin/bash
#SBATCH --job-name=train_2NFs
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/EstimationNF_gaussians_bootstrap_outputs/logs/job_out_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/EstimationNF_gaussians_bootstrap_outputs/logs/job_err_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Activate conda environment
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# -----------------------
# Parameters
# -----------------------
data_path=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Generate_Data/saved_generated_target_data/100k_target_training_set.npy

# Dynamically extract dataset size
num_events=$(python -c "import numpy as np; print(np.load('${data_path}').shape[0])")

# Set output directory using dataset size
outdir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/Train_Models/EstimationNF_gaussians_outputs/0_1_test_N_${num_events}

# Add parent directory to PYTHONPATH so utils.py can be found
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling:$PYTHONPATH

# -----------------------
# Run script
# -----------------------
python test_0_1.py --data_path "$data_path" --outdir "$outdir"
