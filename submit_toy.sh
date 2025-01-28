#!/bin/bash                                                                                                       
#SBATCH -c 1                 # Number of cores (-c)                                                               
#SBATCH --gpus 1                                                                                                  
#SBATCH -t 0-01:00           # Runtime in D-HH:MM, minimum of 10 minutes                                          
#SBATCH -p gpu          # Partition to submit to     
#SBATCH --account=gpu_gres                      # Account to access GPU resources
#SBATCH --mem=20000           # Memory pool for all cores (see also --mem-per-cpu)                                 
#SBATCH -o toy_myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid            
#SBATCH -e toy_myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid      

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

# Run your Python script                                                                                                        
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM-embedding/toy.py -m best_model -g 2000 -r 10000 -t 100 -c True
