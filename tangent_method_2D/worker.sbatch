#!/bin/bash
#SBATCH --job-name=gof2d
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G                                                                                                                                
#SBATCH --time=06:00:00                                                                                                                          
#SBATCH -p shared
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
python worker.py --task-id "$SLURM_ARRAY_TASK_ID"
