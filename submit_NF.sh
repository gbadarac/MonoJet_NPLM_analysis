#!/bin/bash
#SBATCH --job-name=normalizing_flow             # Job name
#SBATCH --output=logs/job_output_%j.out         # Output file path
#SBATCH --error=logs/job_error_%j.err           # Error file path
#SBATCH --time=1-00:00:00                       # Time limit (1 day)
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --account=gpu_gres                      # Account to access GPU resources
#SBATCH --partition=gpu                         # Partition for GPU jobs
#SBATCH --nodes=1                               # Request to run job on a single node
#SBATCH --ntasks=1                              # Request 1 task (1 CPU)
#SBATCH --mem=64G                               # Request 32GB of memory

# Load required modules or activate conda environment if necessary
source /work/gbadarac/envs/my_project/bin/activate my_project  # Modify this line to match your setup

# Default parameters for your normalizing flow training
n_epochs=1001
learning_rate=5e-6
batch_size=512 #number of data samples processed before updating the model's parameters
outdir=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs
hidden_features=50
num_blocks=6
num_bins=10
num_layers=4

# Create a job-specific output directory
#job_outdir=${outdir}/job_${num_layers}_layers_${num_blocks}_transformations_${hidden_features}_neurons_${num_bins}_bins_${SLURM_JOB_ID}
job_outdir=${outdir}/job_${num_layers}_${num_blocks}_${hidden_features}_${num_bins}_best_model_${SLURM_JOB_ID}

echo ${job_outdir}
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
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNFnflows.py --n_epochs ${n_epochs} --learning_rate ${learning_rate} --outdir ${job_outdir} --hidden_features ${hidden_features} --num_blocks ${num_blocks} --num_bins ${num_bins} --num_layers ${num_layers}
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
mv ./logs/job_output_${SLURM_JOB_ID}.out ${job_outdir}
mv ./logs/job_error_${SLURM_JOB_ID}.err ${job_outdir}


