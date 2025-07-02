import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
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

# give a name to each model and provide a path to where the model's prediction for bkg and signal classes are stored
folders = {
    'best_model': '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_gaussians_outputs/job_4_4_64_8_best_model_580200',
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
n_bkg = 2000000

# Energy: single Gaussian
mean_feat1 = -0.5  # adjust as needed
std_feat1 = 0.25
bkg_feat1 = np.random.normal(loc=mean_feat1, scale=std_feat1, size=n_bkg)

# b-tag score: single Gaussian
mean_feat2 = 0.6  # adjust as needed
std_feat2 = 0.4
bkg_feat2 = np.random.normal(loc=mean_feat2, scale=std_feat2, size=n_bkg)

num_features=2 #dimensionality of the data being transformed.
hidden_features=64
num_bins=8
num_blocks=4
num_layers=4

# In this case: b-tagging score and background energy

# Note: the bkg distribution is the posterior/target distribution which the Normalizing Flow should learn to approximate.

# Combining energy and b-tagging score for both bkg and signal 
bkg_coord = np.column_stack((bkg_feat1, bkg_feat2))  # Combine btag and bkg for training
#Initialize the scaler 
scaler = StandardScaler()
# Scale the target distribution to help the model to converge faster 
bkg_coord_scaled = scaler.fit_transform(bkg_coord)

# Convert to float32 for compatibility with PyTorch
bkg_coord_scaled = bkg_coord_scaled.astype('float32')

reference = torch.as_tensor(bkg_coord_scaled[:N_ref], dtype=torch.float32)

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
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_gaussians_outputs/calibration/'
else:
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_NF_gaussians_outputs/comparison/'

# Create unique job directory
job_id = os.getenv('SLURM_JOB_ID', 'local')
job_dir = f"{args.manifold}_NR{args.reference}_NG{args.generated}_M{M}_lam1e-6_iter1000000_job{job_id}/"
output_dir = os.path.join(folder_out, job_dir)
os.makedirs(output_dir, exist_ok=True)

# Export output directory path for SLURM script to access
os.environ['SLURM_OUTPUT_DIR'] = output_dir

# Print the directory path for debugging
print(f"Output directory set to: {output_dir}")

############ begin load data

# NORMALIZING FLOW GENERATED DISTRIBUTION
# Define the function to recreate the flow

def make_flow(num_features, hidden_features, num_bins, num_blocks, num_layers, num_context, perm=True):
    base_dist = distributions.StandardNormal(shape=(num_features,))

    transforms = []
    if num_context == 0:
        num_context = None
    for i in range(num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                context_features=num_context,
                                                                                hidden_features=hidden_features,
                                                                                num_bins=num_bins,
                                                                                num_blocks=num_blocks,
                                                                                tail_bound=10.0, #range over which the spline trasnformation is defined 
                                                                                tails='linear',
                                                                                dropout_probability=0.2,
                                                                                use_batch_norm=False))                                                       
        if i < num_layers - 1 and perm:
            transforms.append(ReversePermutation(features=num_features)) #Shuffles feature order to increase expressivity
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    return flow

# Parameters for the flow (ensure they match the ones used during training)
num_context = 0

# Recreate the flow model
flow = make_flow(num_features, hidden_features, num_bins, num_blocks, num_layers, num_context, perm=True)

print('Load data')

model_path = os.path.join(folders[manifold], "best_model.pth")
flow.load_state_dict(torch.load(model_path))
flow.eval()  # Set the model to evaluation mode
print("Best model loaded successfully.")
############ end load data
    
######## standardizes data
#### compute sigma hyper parameter from data

flk_sigma = candidate_sigma(bkg_coord_scaled[:2000, :], perc=flk_sigma_perc)
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
    N_generated_p = int(rng.poisson(lam=N_generated, size=1)[0])
    num_samples = int(N_generated_p + N_ref)

    # Shuffle background data at the beginning of each iteration (if needed)
    np.random.shuffle(bkg_coord_scaled)

    if calibration:
        data = torch.from_numpy(bkg_coord_scaled[:num_samples])
        label_D = np.ones(N_generated_p, dtype=np.float32)
        label_R = np.zeros(N_ref, dtype=np.float32)
        w_ref = N_generated_p / N_ref  # Reference weight
    else:
        # Sample points from the trained flow
        ref = torch.from_numpy(bkg_coord_scaled[:N_ref])
        data_gen = flow.sample(N_generated_p).detach().numpy()

        num_gen = data_gen.shape[0]
        num_ref = ref.shape[0]

        # Combine the filtered generated data with the filtered reference data
        data = np.concatenate((data_gen, ref), axis=0)

        # Create labels corresponding to the filtered generated data and reference data
        label_D = np.ones(num_gen, dtype=np.float32)  # "Generated" labels
        label_R = np.zeros(num_ref, dtype=np.float32)

        # Recalculate the reference weight based on the filtered reference samples
        w_ref = num_gen / num_ref if num_ref > 0 else 1.0  # Avoid division by zero
  

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

    # Run the toy experiment
    t, pred = run_toy(manifold, data, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path=output_dir, plot=plot_reco, savefig=plot_reco,
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