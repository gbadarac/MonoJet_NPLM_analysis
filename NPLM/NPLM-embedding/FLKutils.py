# Falkon packages
from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss
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

def run_toy(test_label, X_train, Y_train, weight, flk_config, seed, bins_centroids, f_bins_centroids, ferr_bins_centroids, 
            plot=False, verbose=False, savefig=False, output_path='', df=10, binsrange=None, yrange=None, xlabels=None, 
            ):
    
    '''                                                                                                                     
    type of signal: "NP0", "NP1", "NP2", "NP3"                                                                              
    output_path: directory (inside ./runs/) where to save results                                                           
    N_0: size of ref sample                                                                                                 
    N0: expected num of bkg events                                                                                          
    NS: expected num of signal events                                                                                       
    flk_config: dictionary of logfalkon parameters                                                                          
    toys: numpy array with seeds for toy generation                                                                         
    plots_freq: how often to plot inputs with learned reconstructions                                                       
    df: degree of freedom of chi^2 for plots                                                                                
    '''

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
        for i in range(X_train.shape[1]):
            print(f"\n[DEBUG] Feature {i+1}")
            print("  f_bins_centroids has NaNs? ", np.any(np.isnan(f_bins_centroids[:, i])))
            print("  f_bins_centroids == 0?    ", np.any(f_bins_centroids[:, i] == 0))
            print("  ferr_bins_centroids has NaNs? ", np.any(np.isnan(ferr_bins_centroids[:, i])))
            print("  ferr_bins_centroids == 0?    ", np.any(ferr_bins_centroids[:, i] == 0))
        plot_reconstruction(data=X_train[Y_train.flatten()==1], weight_data=1,
                            ref=X_train[Y_train.flatten()==0], weight_ref=weight,
                            ref_preds=preds[Y_train.flatten()==0],                 
                            xlabels=xlabels, 
                            bins_centroids=bins_centroids, f_bins_centroids=f_bins_centroids, ferr_bins_centroids=ferr_bins_centroids, 
                            yrange=yrange,
                            save=savefig, save_path=output_path+'/plots/',
                            file_name=test_label+'.pdf'
                )
    return t, preds

def centroids_to_edges(centroids):
    """
    Convert an array of bin centroids to bin edges.
    Parameters:
        centroids (array-like): 1D array of bin centroids (must be sorted).
    Returns:
        np.ndarray: 1D array of bin edges, length = len(centroids) + 1
    """
    centroids = np.asarray(centroids)
    if centroids.ndim != 1:
        raise ValueError("Input must be a 1D array of centroids.")
    if len(centroids) < 2:
        raise ValueError("At least two centroids are required to compute edges.")
    
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    first_edge = centroids[0] - (midpoints[0] - centroids[0])
    last_edge = centroids[-1] + (centroids[-1] - midpoints[-1])
    edges = np.concatenate([[first_edge], midpoints, [last_edge]])
    return edges

def ratio_with_error(a, b, a_err, b_err):
    """
    Compute the element-wise ratio of two arrays with error propagation.
    
    Parameters:
        a (np.ndarray): Numerator values
        b (np.ndarray): Denominator values
        a_err (np.ndarray): Errors in numerator
        b_err (np.ndarray): Errors in denominator
        
    Returns:
        ratio (np.ndarray): Element-wise ratio a / b
        ratio_err (np.ndarray): Propagated uncertainty of the ratio
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.asarray(a_err)
    b_err = np.asarray(b_err)
    if not (a.shape == b.shape == a_err.shape == b_err.shape):
        raise ValueError("All input arrays must have the same shape.")
    ratio = a / b
    ratio_err = ratio * np.sqrt((a_err / a)**2 + (b_err / b)**2)
    return ratio, ratio_err

def plot_reconstruction(data, # 
                        weight_data, ref, weight_ref, ref_preds, 
                        xlabels=[], #[D]
                        bins_centroids = [], #[Nb, D]
                        f_bins_centroids = [], #[Nb, D]
                        ferr_bins_centroids = [], #[Nb, D]
                        yrange=None,
                        save=False, save_path='', file_name=''):
    '''
    Reconstruction of the data distribution learnt by the model.
    df:              (int) chi2 degrees of freedom                                                                         
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)                                      
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample                                                  
    tau_OBS:         (float) value of the tau term after training
    output_tau_ref:  (numpy array, shape (None, 1)) tau prediction of the reference training sample after training
    feature_labels:  (list of string) list of names of the training variables 
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)   
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel 
                     for each training variable (ymax_code.keys()=feature_labels) 
    delta_OBS:       (float) value of the delta term after training (if not given, only tau reconstruction is plotted) 
    output_delta_ref:(numpy array, shape (None, 1)) delta prediction of the reference training sample after training 
                     (if not given, only tau reconstruction is plotted)     
    '''
    eps = 1e-10

    weight_ref = np.ones(len(ref))*weight_ref
    weight_data = np.ones(len(data))*weight_data

    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    for i in range(data.shape[1]):
        #All = np.append(ref[:, i], data[:, i])
        bins = centroids_to_edges(bins_centroids[:,i])#np.linspace(np.min(All),np.max(All),30)
        N = np.sum(weight_ref)
        bins_legth = (bins[1:]-bins[:-1])*N
        
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor('white')
        ax1= fig.add_axes([0.15, 0.43, 0.5, 0.5])
        hD = ax1.hist(data[:, i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = ax1.hist(ref[:, i], weights=weight_ref, color='darkturquoise', ec='darkturquoise', alpha=0.3, bins=bins, lw=1, label='REFERENCE', zorder=1)
        hN = ax1.hist(ref[:, i], weights=np.exp(ref_preds[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)
        #ax1.plot(bins_centroids[:, i], bins_legth*f_bins_centroids[:, i])
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0], edgecolor='black', label='NPLM RECO', color='#b2df8a', lw=1, s=30, zorder=4)

        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=1, frameon=False, 
                          bbox_to_anchor=(1.05, 1),loc='upper left',borderaxespad=0.)
        font = font_manager.FontProperties(family='serif', size=18)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.15, 0.1, 0.5, 0.3])
        x   = bins_centroids[:, i]
        area= np.sum(bins_legth*f_bins_centroids[:, i])
        print("Area: ", area)
        ratio, ratio_err = ratio_with_error(N/area*bins_legth*f_bins_centroids[:, i], hR[0], N/area*bins_legth*ferr_bins_centroids[:, i], np.sqrt(hR[0]))
        ratio_nplm = hN[0]/(hR[0]+eps)
        ratio_nplm_x = x[ratio_nplm>0]
        ratio_nplm = ratio_nplm[ratio_nplm>0]
        ax2.errorbar(x, hD[0]/(hR[0]+eps), yerr=np.sqrt(hD[0])/(hR[0]+eps), ls='', marker='o', label ='DATA/REF', color='black',zorder=1)
        ax2.plot(ratio_nplm_x, ratio_nplm, label ='NPLM RECO/REF', color='#b2df8a', lw=3, )
        
        # Plot the central ratio line
        ax2.plot(x, ratio, label=r'MODEL/REF', color='blue')
        # ±1σ band
        ax2.fill_between(x, ratio - ratio_err, ratio + ratio_err, color='blue', alpha=0.3, label=r'$\pm 1 \sigma$')
        # ±2σ band
        ax2.fill_between(x, ratio - 2*ratio_err, ratio + 2*ratio_err, color='blue', alpha=0.15, label=r'$\pm 2 \sigma$')
        
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font, bbox_to_anchor=(1.05, 1),
        loc='upper left', frameon=False, 
        borderaxespad=0.)

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
        plt.show()
        plt.close()
    return


#single model (not ensemble)
'''
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
'''