import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
from SampleUtils import *
import scipy.stats
from scipy.stats import rel_breitwigner
import torch
from nflows import distributions, flows, transforms
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.flows import Flow
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets device to GPU if available 

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=int, help="number of data events", required=True) #generated distribution (signal e bkg insieme)
parser.add_argument('-r', '--reference', type=int, help="reference (number of reference events, must be larger than data)", required=True) #target distribution 
parser.add_argument('-t', '--toys', type=int, help="toys", required=True)
parser.add_argument("-c", "--calibration", type=str, default="True", help="Enable calibration mode (True/False)")
parser.add_argument('-M', '--M', type=int, help="Effective number of samples used for configuring the log-likelihood estimation (should be selected based on the dataset size to optimize performance)", required=True)
args = parser.parse_args()

N_ref = args.reference
N_data = args.data

# Convert string to boolean
calibration = args.calibration.lower() == "true"

# Load real reference and data
data_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/500k_2d_gaussian_heavy_tail_target_set.npy"
reference_path = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Generate_Ensemble_Data_Hit_or_Miss_MC/saved_generated_ensemble_data/N_100000_dim_2_seeds_60_4_16_128_15_bimodal_gaussian_heavy_tail/concatenated_ensemble_generated_samples_4_16_128_15_bimodal_gaussian_heavy_tail_N_1000000.npy"

reference_dataset = np.load(reference_path).astype('float32')
data_dataset = np.load(data_path).astype('float32')

reference = torch.as_tensor(reference_dataset[:N_ref], dtype=torch.float32)

# hyper parameters of the model
M=args.M
flk_sigma_perc=90 #%
lam =1e-6
iterations=1000000
Ntoys = args.toys

## Get SLURM job ID
job_id = os.getenv('SLURM_JOB_ID', 'local')

# Define output folder based on calibration flag
if calibration:
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_ensemble/calibration/'
else:
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_ensemble/comparison/'

# Create unique job directory
job_id = os.getenv('SLURM_JOB_ID', 'local')
job_dir = f"nplm_ensemble_NR{args.reference}_NG{args.data}_M{M}_lam1e-6_iter1000000_job{job_id}/"
output_dir = os.path.join(folder_out, job_dir)
os.makedirs(output_dir, exist_ok=True)

# Export output directory path for SLURM script to access
os.environ['SLURM_OUTPUT_DIR'] = output_dir

# Print the directory path for debugging
print(f"Output directory set to: {output_dir}")


######## standardizes data
#### compute sigma hyper parameter from data

flk_sigma = candidate_sigma(reference_dataset[:2000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

# run toys
print('Start running toys')
ts_list = []  # Use a list to accumulate t-values

# Pre-calculate seeds 
base_seed = int(datetime.datetime.now().timestamp() * 1e6) % (2**32 - 1)
seeds = base_seed + np.arange(Ntoys)

for i in range(Ntoys):
    seed = int(seeds[i])
    rng = np.random.default_rng(seed=seed)
    N_data_p = int(rng.poisson(lam=N_data, size=1)[0])
    num_samples = int(N_data_p + N_ref)

    # Shuffle background data at the beginning of each iteration (if needed)
    np.random.shuffle(reference_dataset)
    np.random.shuffle(data_dataset)

    if calibration:
        data = torch.from_numpy(reference_dataset[:num_samples])
        label_D = np.ones(N_data_p, dtype=np.float32)
        label_R = np.zeros(N_ref, dtype=np.float32)
        w_ref = N_data_p / N_ref  # Reference weight
    else:
        # Sample points from the trained flow
        ref = reference_dataset[:N_ref]
        data_gen = data_dataset[:N_data_p]

        num_data = data_gen.shape[0]
        num_ref = ref.shape[0]

        # Combine the filtered generated data with the filtered reference data
        data = np.concatenate((data_gen, ref), axis=0)

        # Create labels corresponding to the filtered generated data and reference data
        label_D = np.ones(num_data, dtype=np.float32)  # "Generated" labels
        label_R = np.zeros(num_ref, dtype=np.float32)

        # Recalculate the reference weight based on the filtered reference samples
        w_ref = num_data / num_ref if num_ref > 0 else 1.0  # Avoid division by zero
  

    # Convert to float32
    labels = np.concatenate((label_D, label_R), axis=0).astype(np.float32)
    
    data = np.float32(data)

    # Enable plotting every 20 iterations
    plot_reco = (i % 20 == 0)
    verbose = plot_reco
    if verbose:
        print(f"Plotting iteration {i}")

    # Get FLK configuration
    flk_config = get_logflk_config(M, flk_sigma, [lam], weight=w_ref, iter=[iterations], seed=None, cpu=False)

    # Set xlabels
    xlabels = ['Feature 1', 'Feature 2']

    def load_stacked_marginals(marginals_dir, num_features):
        f_binned_list = []
        f_err_list = []
        bin_centers_list = []

        for i in range(num_features):
            data = np.load(f"{marginals_dir}/marginal_feature_{i+1}_data.npz")
            f_binned_list.append(data["f_binned"])
            f_err_list.append(data["f_err"])
            bin_centers_list.append(data["bin_centers"])

        # Stack into shape (N, num_features)
        f_binned_stacked = np.stack(f_binned_list, axis=1)
        f_err_stacked = np.stack(f_err_list, axis=1)
        bin_centers_stacked = np.stack(bin_centers_list, axis=1)

        return f_binned_stacked, f_err_stacked, bin_centers_stacked
    
    marginals_dir='/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15_bimodal_gaussian_heavy_tail'
    f_bins_centroids, ferr_bins_centroids, bins_centroids = load_stacked_marginals(marginals_dir, num_features=2)

    # Run the toy experiment
    test_label='ensemble'
    t, pred = run_toy(test_label, data, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, bins_centroids=bins_centroids, f_bins_centroids=f_bins_centroids, ferr_bins_centroids=ferr_bins_centroids, 
                      output_path=output_dir, plot=plot_reco, savefig=plot_reco,
                      verbose=verbose, xlabels=xlabels)
    ts_list.append(t)


# Collect previous toys if they exist
seeds_past, ts_past = np.array([]), np.array([])

# Update path to use output_dir
h5_file_path = os.path.join(output_dir, f'tvalues_flksigma{flk_sigma}.h5')  # Cleaner path definition

if os.path.exists(h5_file_path):
    print('Collecting previous t-values')
    with h5py.File(h5_file_path, 'r') as f:
        seeds_past = np.array(f.get('seed_toy'))
        ts_past = np.array(f.get(str(flk_sigma)))

# Append past values to current results
ts = np.append(ts_past, ts_list)
seeds = np.append(seeds_past, seeds)

# Save updated results
with h5py.File(h5_file_path, 'w') as f:
    f.create_dataset(str(flk_sigma), data=ts, compression='gzip')
    f.create_dataset('seed_toy', data=seeds, compression='gzip')