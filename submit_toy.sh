#!/bin/bash
#SBATCH -c 1                 # Number of cores (-c)
#SBATCH --gpus 1
#SBATCH -t 0-01:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu               # Partition to submit to     
#SBATCH --account=gpu_gres   # Account to access GPU resources
#SBATCH --mem=20000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_outputs/%x-%j.out  # SLURM output file path
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_outputs/%x-%j.err  # SLURM error file path

# Set the LD_LIBRARY_PATH to include the directory with libcusolver.so.11
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/t3home/gbadarac/miniforge3/envs/nplm_env/lib/

# Activate the conda environment
source /t3home/gbadarac/miniforge3/bin/activate nplm_env

# Find CUDA home dynamically through torch
CUDA_PATH=$(python -c "import torch; print(torch.version.cuda)")
echo "CUDA version detected: $CUDA_PATH"

# Set the environment variables dynamically (assuming torch has the right CUDA version)
export CUDA_HOME=/usr/local/cuda-$CUDA_PATH  # Dynamically set CUDA path based on detected version
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Define the output directory for results
CALIBRATION=True  # Change this if needed

# Run the Python script and capture its output
PYTHON_OUTPUT=$(python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM-embedding/toy.py -m best_model -g 2000 -r 10000 -t 100 -c $CALIBRATION)

# Extract the SLURM_OUTPUT_DIR from the Python script output
SLURM_OUTPUT_DIR=$(echo "$PYTHON_OUTPUT" | grep "Output directory set to:" | awk -F': ' '{print $2}')

# Debug statement to check if SLURM_OUTPUT_DIR is set
echo "SLURM_OUTPUT_DIR is set to: $SLURM_OUTPUT_DIR"

# Capture the job name and job ID
JOB_NAME=${SLURM_JOB_NAME}
JOB_ID=${SLURM_JOB_ID}

# Define the SLURM output file paths
OUT_FILE_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_outputs/${JOB_NAME}-${JOB_ID}.out"
ERR_FILE_PATH="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_outputs/${JOB_NAME}-${JOB_ID}.err"

# Move the .out and .err files to the output directory if SLURM_OUTPUT_DIR is set
if [ -n "$SLURM_OUTPUT_DIR" ]; then
    mv "$OUT_FILE_PATH" "$SLURM_OUTPUT_DIR"
    mv "$ERR_FILE_PATH" "$SLURM_OUTPUT_DIR"
else
    echo "SLURM_OUTPUT_DIR is not set. Skipping file move."
fi