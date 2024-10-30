#!/bin/bash

#SBATCH --job-name=normalizing_flow           # Job name
#SBATCH --account=gpu_gres                     # Account to access GPU resources
#SBATCH --partition=gpu                         # Partition for GPU jobs
#SBATCH --nodes=1                              # Request to run job on a single node
#SBATCH --ntasks=1                             # Request 1 task (1 CPU)
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs/job_$SLURM_JOB_ID/job_output_%j.out


# Set output directory
OUTPUT_DIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs/job_${SLURM_JOB_ID}"  # Create job-specific directory
mkdir -p $OUTPUT_DIR  # Create the output directory if it doesn't exist


# Load required modules or activate conda environment if necessary
source /t3home/gbadarac/miniforge3/bin/activate my_project  # Modify this line to match your setup

# Set KERAS_BACKEND for TensorFlow
export KERAS_BACKEND=tensorflow

# Run the Python script and redirect output and errors to the job-specific log file
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows.py > ${OUTPUT_DIR}/job_output_${SLURM_JOB_ID}.out 2>&1





#!/bin/bash

#SBATCH --job-name=normalizing_flow           # Job name
#SBATCH --account=gpu_gres                     # Account to access GPU resources
#SBATCH --partition=gpu                         # Partition for GPU jobs
#SBATCH --nodes=1                              # Request to run job on a single node
#SBATCH --ntasks=1                             # Request 1 task (1 CPU)
#SBATCH --gres=gpu:1                           # Request 1 GPU
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs/job_%j/job_output_%j.out  # SLURM output path

# Load required modules or activate conda environment if necessary
source /t3home/gbadarac/miniforge3/bin/activate my_project  # Modify this line to match your setup

# Set KERAS_BACKEND for TensorFlow
export KERAS_BACKEND=tensorflow

# Run the Python script, making sure it saves outputs to the correct directory
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows.py

