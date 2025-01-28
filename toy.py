import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
from SampleUtils import *
from sklearn.preprocessing import StandardScaler
import scipy.stats
from scipy.stats import rel_breitwigner
import torch
import os
from nflows import distributions, flows, transforms
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.flows import Flow
import matplotlib.pyplot as plt

# give a name to each model and provide a path to where the model's prediction for bkg and signal classes are stored
#different normalizing flows, path dei modelli generati 
folders = {
    'best_model': '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Normalizing_Flows/EstimationNF_outputs/job_4_6_50_10_best_model_419803',
}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--manifold', type=str, help="manifold type (must be in folders.keys())", required=True)
parser.add_argument('-g', '--generated', type=int, help="generated distribution (number of generated events)", required=True) #generated distribution (signal e bkg insieme)
parser.add_argument('-r', '--reference', type=int, help="reference (number of reference events, must be larger than background)", required=True) #target distribution 
parser.add_argument('-t', '--toys', type=int, help="toys", required=True)
parser.add_argument('-c', '--calibration', type=str, help="calibration", required=True)
args = parser.parse_args()

#parser arguments 
manifold = args.manifold

N_ref = args.reference
N_generated = args.generated
w_ref = N_generated*1./N_ref

calibration = args.calibration

#GROUND TRUTH DISTRIBTION
# Generate background and signal data
n_bkg = 800000
n_sig = 40
bkg = np.random.exponential(scale=100.0, size=n_bkg)
sig = rel_breitwigner.rvs(450, size=n_sig)
# Adding b-tagging information (a form of event classification)
bkg_btag = np.random.uniform(low=0.0, high=1.0, size=n_bkg)
sig_btag = np.random.normal(0.85, 0.05, n_sig)

num_features=2 #dimensionality of the data being transformed.
# In this case: b-tagging score and background energy

# Note: the bkg distribution is the posterior/target distribution which the Normalizing Flow should learn to approximate.

# Combining energy and b-tagging score for both bkg and signal 
bkg_coord = np.column_stack((bkg_btag, bkg))  # Combine btag and bkg for training
#Initialize the scaler 
scaler = StandardScaler()
# Scale the target distribution to help the model to converge faster 
bkg_coord_scaled = scaler.fit_transform(bkg_coord)

# Shift the entire dataset to make sure all values are positive
shift = -bkg_coord_scaled[:, 1].min() + 1e-6  # Get the absolute value of the minimum across all features

bkg_coord_scaled[:, 1] += shift  # Add the shift to the entire dataset

# Convert to float32 for compatibility with PyTorch
bkg_coord_scaled = bkg_coord_scaled.astype('float32')

reference = torch.as_tensor(bkg_coord_scaled[:N_ref], dtype=torch.float32)

# hyper parameters of the model
M=10000
flk_sigma_perc=90 #%
lam =1e-6
iterations=1000000
Ntoys = args.toys

# details about the save path
if calibration == True:
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_outputs/calibration/'
else:
    folder_out = '/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/NPLM/NPLM_outputs/comparison/'
    
string = ''
NP = '%s%s_NR%i_NG%i_M%i_lam%s_iter%i/'%(manifold, string, N_ref, N_generated,
                                                  M, str(lam), iterations)
if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)

############ begin load data

# NORMALIZING FLOW GENERATED DISTRIBUTION
# Define the function to recreate the flow

def make_flow(num_features,num_context, perm=True):
    base_dist = distributions.StandardNormal(shape=(num_features,))

    transforms = []
    num_layers = 4
    if num_context == 0:
        num_context = None
    for i in range(num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=num_features,
                                                                                context_features=num_context,
                                                                                hidden_features=50,
                                                                                num_bins=10,
                                                                                num_blocks=6,
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
num_features = 2  # Dimensionality of the data
num_context = 0

# Recreate the flow model
flow = make_flow(num_features, num_context, perm=True)

print('Load data')

model_path = os.path.join(folders[manifold], "best_model.pth")
flow.load_state_dict(torch.load(model_path, weights_only=True))
flow.eval()  # Set the model to evaluation mode

print("Best model loaded successfully.")
    
# Sample points from the trained flow
generated = flow.sample(N_generated).detach().numpy()
############ end load data

if calibration == True:
    data = torch.from_numpy(bkg_coord_scaled[:N_generated]) #reference distribution but sample N_generated events and not N_reference
else:
    data = generated
    
######## standardizes data

##USE REFERENCE HERE AND MOVE CALIBRATION THING IN THE LOOP 
print('standardize')
features_mean, features_std = np.mean(data, axis=0), np.std(data, axis=0)
print('Mean:', features_mean)
print('Std Dev:', features_std)

features_data=standardize(data, features_mean, features_std)

#### compute sigma hyper parameter from data
#### (This doesn't need modifications)
flk_sigma = candidate_sigma(features_data[:2000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

# run toys
print('Start running toys')
ts = np.array([])
seeds = np.arange(Ntoys) + datetime.datetime.now().microsecond + datetime.datetime.now().second + datetime.datetime.now().minute
for i in range(Ntoys):
    seed = seeds[i]
    rng = np.random.default_rng(seed=seed)
    N_generated_p = rng.poisson(lam=N_generated, size=1)[0]
    
    # Ensure that N_generated_p + N_ref is a valid sample size
    num_samples = int(N_generated_p + N_ref)

    # Directly sample from the trained normalizing flow instead of using features_data
    
    generated_data = flow.sample(num_samples).detach().numpy()  # Sample from the normalizing flow
    generated_data = np.float32(generated_data)  # Ensure generated data is a float32 numpy array

    
    # Separate signal and background for labels
    label_R = np.zeros((N_ref,))  # Reference (background)
    label_D = np.ones((N_generated_p,))  # Generated (signal)
    labels = np.concatenate((label_D, label_R), axis=0)

    # Optionally plot every 20 toys (can be adjusted)
    plot_reco = False
    verbose = False
    if not i % 20:
        plot_reco = True
        verbose = True
        print(f"Plotting iteration {i}")
        
    # Convert labels to the same dtype as generated_data (in case it's NumPy and generated_data is Tensor)
    if isinstance(generated_data, np.ndarray):
        labels = np.asarray(labels, dtype=np.float32)  # Ensure labels are also numpy arrays of float32
        
    flk_config = get_logflk_config(M, flk_sigma, [lam], weight=w_ref, iter=[iterations], seed=None, cpu=False)
    
    # Initialize xlabels if it's not already defined elsewhere
    xlabels = locals().get('xlabels', None)  # Check if xlabels exists in the local scope, else return None
    
    # Ensure xlabels is initialized
    xlabels = xlabels if xlabels is not None else ['Feature {}'.format(i) for i in range(10)]  # Adjust 10 to match your number of features
    
    # Pass generated data tensor to run_toy
    t, pred = run_toy(manifold, generated_data, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path='./', plot=plot_reco, savefig=plot_reco,
                      verbose=verbose, xlabels=xlabels)
    ts = np.append(ts, t)

# collect previous toys if existing
seeds_past = np.array([])
ts_past = np.array([])
if os.path.exists('%s/%s/tvalues_flksigma%s.h5' % (folder_out, NP, flk_sigma)):
    print('collecting previous tvalues')
    f = h5py.File('%s/%s/tvalues_flksigma%s.h5' % (folder_out, NP, flk_sigma), 'r')
    seeds_past = np.array(f.get('seed_toy'))
    ts_past = np.array(f.get(str(flk_sigma)))
    f.close()

ts = np.append(ts_past, ts)
seeds = np.append(seeds_past, seeds)

f = h5py.File('%s/%s/tvalues_flksigma%s.h5' % (folder_out, NP, flk_sigma), 'w')
f.create_dataset(str(flk_sigma), data=ts, compression='gzip')
f.create_dataset('seed_toy', data=seeds, compression='gzip')
f.close()
