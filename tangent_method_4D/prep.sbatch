#!/bin/bash
#SBATCH --job-name=gof4d_prep
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
# add your partition/account
export OMP_NUM_THREADS=4
mkdir -p logs
python prep.py
