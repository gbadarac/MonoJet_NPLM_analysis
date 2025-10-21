#!/bin/bash
#SBATCH --job-name=generate_hit_or_miss
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data/logs/job_out_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data/logs/job_err_%j.err
#SBATCH --array=0-199
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1

# Activate conda environment
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# Create log dir if it doesn't exist
mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data/logs

python generate_hit_or_miss.py --seed $SLURM_ARRAY_TASK_ID