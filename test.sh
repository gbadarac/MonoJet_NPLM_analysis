#!/bin/bash
#SBATCH --job-name=normalizing_flow
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Load the necessary modules or activate your environment
source /t3home/gbadarac/miniforge3/bin/activate my_project  # Adjust this line based on your setup

# Set the parameters
N_EPOCHS=4001
LEARNING_RATE=5e-4
OUTDIR="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs"

# Run the Python script with the specified parameters
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows.py --n_epochs $N_EPOCHS --learning_rate $LEARNING_RATE --outdir $OUTDIR


'''
# Create a job-specific output directory
job_outdir=${outdir}/job_%j
mkdir -p ${job_outdir}  # Ensure the output directory exists

# Gather parameters given by user.
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

#Pass arguments to the python script:

# Run the script with print flushing instantaneous.
export PYTHONUNBUFFERED=TRUE
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows.py --n_epochs ${n_epochs} --learning_rate ${learning_rate} --outdir ${job_outdir} 
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
mv ./logs/job_output_%j.out ${job_outdir}
'''