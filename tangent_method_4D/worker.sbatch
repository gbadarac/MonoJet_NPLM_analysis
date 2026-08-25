#!/bin/bash
#SBATCH --job-name=gof4d
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
# add your partition/account:  #SBATCH -p <partition>   #SBATCH -A <account>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
python worker.py --task-id "$SLURM_ARRAY_TASK_ID"
