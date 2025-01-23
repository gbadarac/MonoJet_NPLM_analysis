#!/bin/bash                                                                                                       
#SBATCH -c 1                 # Number of cores (-c)                                                               
#SBATCH --gpus 1                                                                                                  
#SBATCH -t 0-01:00           # Runtime in D-HH:MM, minimum of 10 minutes                                          
#SBATCH -p gpu          # Partition to submit to                                                             
#SBATCH --mem=20000           # Memory pool for all cores (see also --mem-per-cpu)                                 
#SBATCH -o toy_myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid            
#SBATCH -e toy_myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid            

# load modules                                                                                                    
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01

# run code                                                                                                        
python toy.py -m h_4 -s 0 -b 2000 -r 10000 -t 100 -l 5
