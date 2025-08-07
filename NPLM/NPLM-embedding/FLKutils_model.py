# Falkon packages
from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss
from falkon.options import FalkonOptions
opts = FalkonOptions()
device_type = "cpu" if opts.use_cpu else "cuda"
print(f">>> Falkon device: {device_type}", flush=True)
# scipy packages
from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

# plot packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
font = font_manager.FontProperties(family='serif', size=20)

# others
import numpy as np
import os, time
import torch


def candidate_sigma(data, perc=90):
    # this function estimates the width of the gaussian kernel.                          
    # use on a (small) sample of reference data (standardize first if necessary)         
    pairw = pdist(data)
    return np.around(np.percentile(pairw,perc),1)

def get_logflk_config(M,flk_sigma,lam,weight,iter=[1000000],seed=None,cpu=False):
    # it returns logfalkon parameters                                                                                                              
    return {
            'kernel' : GaussianKernel(sigma=flk_sigma),
            'M' : M, #number of Nystrom centers,                                                                                                  
            'penalty_list' : lam, # list of regularization parameters,                                                                            
            'iter_list' : iter, #list of number of CG iterations,                                                                                 
            'options' : FalkonOptions(cg_tolerance=np.sqrt(float(1e-7)), keops_active='no', use_cpu=cpu, debug = False),
            'seed' : seed, # (int or None), the model seed (used for Nystrom center selection) is manually set,                                   
            'loss' : WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=flk_sigma), neg_weight=weight),
            }

def trainer(X,Y,flk_config):
    # trainer for logfalkon model
    Xtorch= torch.from_numpy(X)
    Ytorch= torch.from_numpy(Y)
    model = LogisticFalkon(**flk_config)
    model.fit(Xtorch, Ytorch)
    return model.predict(Xtorch).numpy()

def compute_t(preds,Y,weight):
    # it returns extended log likelihood ratio from predictions
    diff = weight*np.sum(1 - np.exp(preds[Y==0]))
    return 2 * (diff + np.sum(preds[Y==1]))

#single model (not ensemble)

def run_toy(test_label, X_train, Y_train, weight, flk_config, seed,
            plot=False, verbose=False, savefig=False, output_path='', df=10, binsrange=None, yrange=None, xlabels=None):

    if not os.path.exists(output_path):
      os.makedirs(output_path, exist_ok=True)
    dim = X_train.shape[1]                                                                                                        
    flk_config['seed']=seed # select different centers for different toys                                                   
    st_time = time.time()
    preds = trainer(X_train,Y_train,flk_config)
    t = compute_t(preds,Y_train,weight)
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---LRT = {}\n---Time = {} sec\n\t".format(seed,t,dt))
    if plot:
        plot_reconstruction(data=X_train[Y_train.flatten()==1], weight_data=1,
                            ref=X_train[Y_train.flatten()==0], weight_ref=weight,
                            ref_preds=preds[Y_train.flatten()==0],                 
                            yrange=yrange,binsrange=binsrange, xlabels=xlabels,
                            save=savefig, save_path=output_path+'/plots/',
                            file_name=test_label+'.pdf'
                )
    return t, preds


def plot_reconstruction(data, weight_data, ref, weight_ref, ref_preds, xlabels=[], yrange=None, binsrange=None,
                        save=False, save_path='', file_name=''):

    eps = 1e-10

    weight_ref = np.ones(len(ref))*weight_ref
    weight_data = np.ones(len(data))*weight_data

    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    for i in range(data.shape[1]):
        All = np.append(ref[:, i], data[:, i])
        bins = np.linspace(np.min(All),np.max(All),30)
        if not binsrange==None:
            if len(binsrange[xlabels[i]]):
                bins=binsrange[xlabels[i]]
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax1= fig.add_axes([0.15, 0.43, 0.8, 0.5])
        hD = plt.hist(data[:, i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(ref[:, i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
        hN = plt.hist(ref[:, i], weights=np.exp(ref_preds[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)

        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0], edgecolor='black', label='RECO', color='#b2df8a', lw=1, s=30, zorder=4)

        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.15, 0.1, 0.8, 0.3])
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/(hR[0]+eps), yerr=np.sqrt(hD[0])/(hR[0]+eps), ls='', marker='o', label ='DATA/REF', color='black')
        plt.plot(x, hN[0]/(hR[0]+eps), label ='RECO', color='#b2df8a', lw=3)
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)

        if len(xlabels)>0:
            plt.xlabel(xlabels[i], fontsize=22, fontname='serif')
        else:
            plt.xlabel('x', fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')

        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylim(0, 2)

        if len(xlabels):
            if not yrange==None and len(xlabels)>0:
                plt.ylim(yrange[xlabels[i]][0], yrange[xlabels[i]][1])
        plt.grid()
        if save:
            os.makedirs(save_path, exist_ok=True)
            #fig.savefig(save_path+file_name.replace('.pdf', '_%i.pdf'%(i)))
            fig.savefig(save_path+file_name.replace('.pdf', '_%i.png'%(i)))
        plt.close()
    return
