#!/bin/bash
#SBATCH --job-name=gof2d_prep
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                                                                                                                                
#SBATCH --time=06:00:00                                                                                                                          
#SBATCH -p shared
export OMP_NUM_THREADS=4
mkdir -p logs
python prep.py
