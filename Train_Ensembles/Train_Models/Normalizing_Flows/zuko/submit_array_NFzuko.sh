#!/bin/bash
#SBATCH --job-name=NF_job
#SBATCH --array=0-0
#SBATCH --output=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko_outputs/logs/job_output_%j.out
#SBATCH --error=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko_outputs/logs/job_error_%j.err
# #SBATCH --output=/dev/null
# #SBATCH --error=/dev/null
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
# #SBATCH --gres=gpu:1

# === Activate env
source /work/gbadarac/miniforge3/bin/activate
conda activate nf_env

# === Parameters
seed=$SLURM_ARRAY_TASK_ID
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH

# ======================================
# Optional Bayesian flag
# ======================================
extra_flags=""
if [ "$bayesian" = true ] || [ "$bayesian" = True ]; then
  extra_flags="--bayesian"
fi

# === Run training
python /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/zuko/EstimationNFzuko.py \
    --data_path ${DATA_PATH} \
    --outdir ${TRIAL_DIR} \
    --seed ${seed} \
    --n_epochs ${N_EPOCHS} \
    --learning_rate ${LR} \
    --batch_size ${BATCH_SIZE} \
    --hidden_features ${HIDDEN_FEATURES} \
    --num_blocks ${NUM_BLOCKS} \
    --num_bins ${NUM_BINS} \
    --num_layers ${NUM_LAYERS} \
    ${extra_flags}
