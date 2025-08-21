#!/bin/bash
#SBATCH -c 1
#SBATCH --gpus 1
#SBATCH -t 0-01:00
#SBATCH -p gpu
#SBATCH --account=gpu_gres
#SBATCH --mem=20000
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/results/logs/%x-%j.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/results/logs/%x-%j.err

export LD_LIBRARY_PATH=/work/gbadarac/miniforge3/envs/nplm_env/lib:$LD_LIBRARY_PATH
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh && conda activate nf_env

CUDA_PATH=$(python -c "import torch; print(torch.version.cuda)")
echo "CUDA version detected: $CUDA_PATH"
export CUDA_HOME=/usr/local/cuda-$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Ensure Python can find utils_flows.py
export PYTHONPATH=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

CALIBRATION=False
N_TOYS=100
BASE_OUT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/results/N_100000_dim_2_seeds_60_4_16_256_15_freezed_train_centers_L2_10_lr_1e-4_kernels_100_sigma_1e-1"

w="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_256_15/w_i_fitted.npy"
w_cov="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_256_15/cov_w.npy"
hit_or_miss_data="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data/N_100000_dim_2_seeds_60_4_16_256_15/concatenated_ensemble_generated_samples_4_16_256_15.npy"
ensemble_dir="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/2_dim/N_100000_dim_2_seeds_60_4_16_256_15"

PY=/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT_with_unc/toy_LRT_with_unc.py
CMD=(python -u "$PY"
     -t "$N_TOYS"
     -c "$CALIBRATION"
     --out_dir "$BASE_OUT"
     --w_path "$w"
     --w_cov_path "$w_cov"
     --ensemble_dir "$ensemble_dir")

# only pass hit_or_miss_data in calibration
if [ "$CALIBRATION" = "True" ] || [ "$CALIBRATION" = "true" ]; then
  CMD+=(--hit_or_miss_data "$hit_or_miss_data")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done."
