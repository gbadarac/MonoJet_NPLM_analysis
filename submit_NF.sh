#!/bin/bash

#SBATCH --job-name=normalizing_flow           # Job name
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs/job_$SLURM_JOB_ID/job_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs/job_$SLURM_JOB_ID/job_error_%j.err
#SBATCH --time=1-00:00:00                 # Time limit (1 day)
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --account=gpu_gres                     # Account to access GPU resources
#SBATCH --partition=gpu                         # Partition for GPU jobs
#SBATCH --nodes=1                              # Request to run job on a single node
#SBATCH --ntasks=1                             # Request 1 task (1 CPU)

# Load required modules or activate conda environment if necessary
source /t3home/gbadarac/miniforge3/bin/activate my_project  # Modify this line to match your setup

# Check GPU availability
srun nvidia-smi

# Create the output directory if it doesn't exist
mkdir -p /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs/job_$SLURM_JOB_ID/

# Run the Python script, making sure it saves outputs to the correct directory
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows.py # Adjust to point to your actual script


