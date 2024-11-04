#!/bin/bash
#SBATCH --job-name=normalizing_flow             # Job name
#SBATCH --output=logs/job_output_%j.out   # Output file path
#SBATCH --error=logs/job_error_%j.err    # Error file path
#SBATCH --time=1-00:00:00                       # Time limit (1 day)
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --account=gpu_gres                      # Account to access GPU resources
#SBATCH --partition=gpu                         # Partition for GPU jobs
#SBATCH --nodes=1                               # Request to run job on a single node
#SBATCH --ntasks=1                              # Request 1 task (1 CPU)

# Ensure the logs directory exists
mkdir -p logs

# Load required modules or activate conda environment if necessary
source /t3home/gbadarac/miniforge3/bin/activate my_project  # Modify this line to match your setup

# Default parameters for your normalizing flow training
N_EPOCHS=4001
LEARNING_RATE=5e-4
OUTDIR='/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs'
#hidden_features 
#num_blocks
#batch

python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows.py --n_epochs $N_EPOCHS --learning_rate $LEARNING_RATE --outdir $OUTDIR

