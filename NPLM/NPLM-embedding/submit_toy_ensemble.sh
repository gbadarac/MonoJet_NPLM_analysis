#!/bin/bash
#SBATCH -c 1                 # Number of cores (-c)
#SBATCH --gpus 1
#SBATCH -t 0-01:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to     
#SBATCH --account=gpu_gres   # Account to access GPU resources
#SBATCH --mem=20000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_ensemble/logs/%x-%j.out  # Keep logs in 'logs'
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_ensemble/logs/%x-%j.err  # Keep error logs in 'logs'

# Set the LD_LIBRARY_PATH to include the directory with libcusolver.so.11
export LD_LIBRARY_PATH=/work/gbadarac/miniforge3/envs/nplm_env/lib:$LD_LIBRARY_PATH

# Activate the conda environment
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh && conda activate nplm_env

# Find CUDA home dynamically through torch
CUDA_PATH=$(python -c "import torch; print(torch.version.cuda)")
echo "CUDA version detected: $CUDA_PATH"

# Set the environment variables dynamically (assuming torch has the right CUDA version)
export CUDA_HOME=/usr/local/cuda-$CUDA_PATH  # Dynamically set CUDA path based on detected version
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Define the output directory for results
CALIBRATION=True # Change this if needed

# Run the Python script and capture its output
TEMP_LOG=/tmp/py_output_$SLURM_JOB_ID.log
python -u /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM-embedding/toy_ensemble.py -g 2000 -r 10000 -t 100 -c $CALIBRATION -M 500 | tee $TEMP_LOG
PYTHON_OUTPUT=$(cat $TEMP_LOG)

# Extract the SLURM_OUTPUT_DIR from the Python script output
SLURM_OUTPUT_DIR=$(echo "$PYTHON_OUTPUT" | grep "Output directory set to:" | awk -F': ' '{print $2}')
echo "Extracted SLURM_OUTPUT_DIR: $SLURM_OUTPUT_DIR"

# Remove temporary log file (not needed anymore)
rm -f "$TEMP_LOG"

# Ensure the output directory exists
mkdir -p "$SLURM_OUTPUT_DIR"

echo "Job completed. Outputs stored in $SLURM_OUTPUT_DIR"
