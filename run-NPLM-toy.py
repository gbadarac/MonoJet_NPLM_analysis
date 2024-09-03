import glob, h5py, math, time, os, json, argparse, datetime
import numpy as np
from FLKutils import *
from SampleUtils import *

# labels identifying the signal classes in the dataset
sig_labels=[3]
# labels identifying the background classes in the dataset
bkg_labels=[0, 1, 2]

# hyper parameters of the NPLM model based on kernel methods
## number of kernels
M = 1000 

## percentile of the distribution of pair-wise distances between reference-distributed points
flk_sigma_perc=90 

## L2 regularization coefficient
lam =1e-6 

## number of maximum iterations before the training is killed
iterations=1000000 

## number of toys to simulate 
## (multiple experiments allow you to build a statistics for the test and quantify type I and II errors)
Ntoys = 10 


a = np.load('./predictions/simclr_predictions.npz')
target=a['dim8']
features=a['dim8_embedding'].reshape((-1, 4))
# select SIG and BKG classes
mask_SIG = np.zeros_like(target)
mask_BKG = np.zeros_like(target)
for sig_label in sig_labels:
1000000    mask_SIG += 1*(target==sig_label)
for bkg_label in bkg_labels:
    mask_BKG += 1*(target==bkg_label)

features_SIG = features[mask_SIG>0]
features_BKG = features[mask_BKG>0]


######## standardizes data
print('standardize')
features_mean, features_std = np.mean(features_BKG, axis=0), np.std(features_BKG, axis=0)
print('mean: ', features_mean)
print('std: ', features_std)
features_BKG = standardize(features_BKG, features_mean, features_std)
features_SIG = standardize(features_SIG, features_mean, features_std)

#### compute sigma hyper parameter from data
#### sigma is the gaussian kernels width. 
#### Following a heuristic, we set this hyperparameter to the 90% quantile of the distribution of pair-wise distances between bkg-distributed points
#### (see below)
#### This doesn't need modifications, but one could in principle change it (see https://arxiv.org/abs/2408.12296)
flk_sigma = candidate_sigma(features_BKG[:2000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

N_ref = 10000 # number of reference datapoints (mixture of non-anomalous classes)
N_bkg = 1000 # number of backgroun datapoints in the data (mixture of non-anomalous classes present in the data)
N_sig = 0 # number of signal datapoints in the data (mixture of anomalous classes present in the data)
w_ref = N_bkg*1./N_ref


## run toys
print('Start running toys')
t0=np.array([])
seeds = np.random.uniform(low=1, high=100000, size=(Ntoys,))
for i in range(Ntoys):
    seed = int(seeds[i])
    rng = np.random.default_rng(seed=seed)
    N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
    N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
    rng.shuffle(features_SIG)
    rng.shuffle(features_BKG)
    features_s = features_SIG[:N_sig_p, :]
    features_b = features_BKG[:N_bkg_p+N_ref, :]
    features  = np.concatenate((features_s,features_b), axis=0)

    label_R = np.zeros((N_ref,))
    label_D = np.ones((N_bkg_p+N_sig_p, ))
    labels  = np.concatenate((label_D,label_R), axis=0)
    
    plot_reco=False
    verbose=False
    # make reconstruction plots every 20 toys (can be changed)
    #if not i%20:
    #    plot_reco=True
    #    verbose=True
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
    t, pred = run_toy('t0', features, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path='./', plot=plot_reco, savefig=plot_reco,
                      verbose=verbose)
    
    t0 = np.append(t0, t)


N_ref = 10000 # number of reference datapoints (mixture of non-anomalous classes)
N_bkg = 1000 # number of backgroun datapoints in the data (mixture of non-anomalous classes present in the data)
N_sig = 10 # number of signal datapoints in the data (mixture of anomalous classes present in the data)
w_ref = N_bkg*1./N_ref


## run toys
print('Start running toys')
t1=np.array([])
seeds = np.random.uniform(low=1, high=100000, size=(Ntoys,))
for i in range(Ntoys):
    seed = int(seeds[i])
    rng = np.random.default_rng(seed=seed)
    N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
    N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
    rng.shuffle(features_SIG)
    rng.shuffle(features_BKG)
    features_s = features_SIG[:N_sig_p, :]
    features_b = features_BKG[:N_bkg_p+N_ref, :]
    features  = np.concatenate((features_s,features_b), axis=0)

    label_R = np.zeros((N_ref,))
    label_D = np.ones((N_bkg_p+N_sig_p, ))
    labels  = np.concatenate((label_D,label_R), axis=0)
    
    plot_reco=False
    verbose=False
    # make reconstruction plots every 20 toys (can be changed)
    #if not i%20:
    #    plot_reco=True
    #    verbose=True
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
    t, pred = run_toy('t1', features, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path='./', plot=plot_reco, savefig=plot_reco,
                      verbose=verbose)
    
    t1 = np.append(t1, t)


## details about the save path                                                                                                                               
folder_out = './out/'
sig_string = ''
if N_sig:
    sig_string+='_SIG'
    for s in sig_labels:
        sig_string+='-%i'%(s)
NP = '%s_NR%i_NB%i_NS%i_M%i_lam%s_iter%i/'%(sig_string, N_ref, N_bkg, N_sig,
                                                  M, str(lam), iterations)
if not os.path.exists(folder_out+NP):
    os.makedirs(folder_out+NP)

np.save('./t0.npy', t0)
np.save('./t1.npy', t1)
