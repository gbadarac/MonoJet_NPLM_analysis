import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils_model import *
from SampleUtils import *
from sklearn.preprocessing import StandardScaler
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
from utils_flows import make_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)

# give a name to each model and provide a path to where the model's prediction for bkg and signal classes are stored
folders = {
    'model': '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/2_dim/N_100000_dim_2_seeds_60_4_16_128_15',
}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--manifold', type=str, help="manifold type (must be in folders.keys())", required=True)
parser.add_argument('-g', '--generated', type=int, help="generated distribution (number of generated events)", required=True) #generated distribution (signal e bkg insieme)
parser.add_argument('-r', '--reference', type=int, help="reference (number of reference events, must be larger than background)", required=True) #target distribution 
parser.add_argument('-t', '--toys', type=int, help="toys", required=True)
parser.add_argument("-c", "--calibration", type=str, default="True", help="Enable calibration mode (True/False)")
parser.add_argument('-M', '--M', type=int, help="Effective number of samples used for configuring the log-likelihood estimation (should be selected based on the dataset size to optimize performance)", required=True)
args = parser.parse_args()

#parser arguments 
manifold = args.manifold

N_ref = args.reference
N_generated = args.generated

# Convert string to boolean
calibration = args.calibration.lower() == "true"

#GROUND TRUTH DISTRIBTION
# Generate background 
n_bkg = 200000

# Energy: single Gaussian
mean_feat1 = -0.5  # adjust as needed
std_feat1 = 0.25
bkg_feat1 = np.random.normal(loc=mean_feat1, scale=std_feat1, size=n_bkg)

# b-tag score: single Gaussian
mean_feat2 = 0.6  # adjust as needed
std_feat2 = 0.4
bkg_feat2 = np.random.normal(loc=mean_feat2, scale=std_feat2, size=n_bkg)

num_features=2 #dimensionality of the data being transformed.
hidden_features=128
num_bins=15
num_blocks=16
num_layers=4

# In this case: b-tagging score and background energy

# Note: the bkg distribution is the posterior/target distribution which the Normalizing Flow should learn to approximate.

# Combining energy and b-tagging score for both bkg and signal 
bkg_coord = np.column_stack((bkg_feat1, bkg_feat2)).astype('float32')  # Combine btag and bkg for training

reference = torch.as_tensor(bkg_coord[:N_ref], dtype=torch.float32)

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
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_one_model/calibration/'
else:
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_one_model/comparison/'

# Create unique job directory
job_id = os.getenv('SLURM_JOB_ID', 'local')
job_dir = f"{args.manifold}_NR{args.reference}_NG{args.generated}_M{M}_lam1e-6_iter1000000_job{job_id}/"
output_dir = os.path.join(folder_out, job_dir)
os.makedirs(output_dir, exist_ok=True)

# Export output directory path for SLURM script to access
os.environ['SLURM_OUTPUT_DIR'] = output_dir

# Print the directory path for debugging
print(f"Output directory set to: {output_dir}", flush=True)
############ begin load data

# NORMALIZING FLOW GENERATED DISTRIBUTION
# Define the function to recreate the flow
flow = make_flow(num_layers, hidden_features, num_bins, num_blocks, num_features)

print('Load data')

models_path = os.path.join(folders[manifold], "f_i.pth")

# load only to CPU first
all_models = torch.load(models_path, map_location="cpu")

# choose the index you want (e.g. 0)
flow.load_state_dict(all_models[0])

flow.to(device)

flow.eval()
print("Model loaded successfully.")
############ end load data
    
######## standardizes data
#### compute sigma hyper parameter from data

flk_sigma = candidate_sigma(bkg_coord[:2000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

# run toys
print('Start running toys')
ts_list = []  # Use a list to accumulate t-values

# Pre-calculate seeds 
base_seed = int(datetime.datetime.now().timestamp() * 1e6) % (2**32 - 1)
seeds = base_seed + np.arange(Ntoys)

for i in range(Ntoys):
    print(f"\n\n========== Starting toy iteration {i} ==========", flush=True)

    seed = int(seeds[i])
    print(f"Seed: {seed}", flush=True)

    rng = np.random.default_rng(seed=seed)
    N_generated_p = int(rng.poisson(lam=N_generated, size=1)[0])
    num_samples = int(N_generated_p + N_ref)
    print(f"N_generated_p: {N_generated_p} | num_samples (gen + ref): {num_samples}", flush=True)

    np.random.shuffle(bkg_coord)
    print("Shuffled background", flush=True)

    if calibration:
        data = torch.from_numpy(bkg_coord[:num_samples])
        label_D = np.ones(N_generated_p, dtype=np.float32)
        label_R = np.zeros(N_ref, dtype=np.float32)
        w_ref = N_generated_p / N_ref
    else:
        ref = torch.from_numpy(bkg_coord[:N_ref])
        data_gen = flow.sample(N_generated_p).detach().cpu().numpy()
        print(f"Generated {data_gen.shape[0]} samples from NF", flush=True)
        # Free up memory: delete flow after sampling

        num_gen = data_gen.shape[0]
        num_ref = ref.shape[0]
        data = np.concatenate((data_gen, ref), axis=0)

        label_D = np.ones(num_gen, dtype=np.float32)
        label_R = np.zeros(num_ref, dtype=np.float32)

        w_ref = num_gen / num_ref if num_ref > 0 else 1.0
        print(f"Weight w_ref: {w_ref}", flush=True)

    labels = np.concatenate((label_D, label_R), axis=0).astype(np.float32)
    data = np.float32(data)
    print(f"Data and labels prepared. Data shape: {data.shape}", flush=True)

    plot_reco = (i % 20 == 0)
    verbose = plot_reco
    if verbose:
        print(f"Plotting iteration {i}", flush=True)

    print("Getting FLK config...", flush=True)
    flk_config = get_logflk_config(M, flk_sigma, [lam], weight=w_ref, iter=[iterations], seed=None, cpu=False)
    print("FLK config ready.", flush=True)

    xlabels = ['Feature 1', 'Feature 2']

    print("Calling run_toy...", flush=True)
    t, pred = run_toy(manifold, data, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path=output_dir, plot=plot_reco, savefig=plot_reco,
                      verbose=verbose, xlabels=xlabels)
    print(f"run_toy finished. t-value: {t}", flush=True)

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