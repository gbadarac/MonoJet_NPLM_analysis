#!/bin/bash
#SBATCH --job-name=find_best_models
#SBATCH --output=logs/find_best_models.out
#SBATCH --error=logs/find_best_models.err
#SBATCH --time=1-00:00:00                       # Time limit (1 day)
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --account=gpu_gres                      # Account to access GPU resources
#SBATCH --partition=gpu                         # Partition for GPU jobs
#SBATCH --nodes=1                               # Request to run job on a single node
#SBATCH --ntasks=1                              # Request 1 task (1 CPU)
#SBATCH --mem=32G                               # Request 32GB of memory

#SBATCH --output=logs/job_output_%j.out         # Output file path
#SBATCH --error=logs/job_error_%j.err           # Error file path


# Load required modules or activate conda environment if necessary
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# Run the Python script
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Grid_Search/find_best_models.py