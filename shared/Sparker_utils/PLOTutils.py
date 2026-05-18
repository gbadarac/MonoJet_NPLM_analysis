import numpy as np
import os, time

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
from scipy.stats import norm, expon
import os, json, glob, random, h5py


def plot_reconstruction(data, weight_data, ref, weight_ref, ref_preds, ref_preds_labels, t_obs=None, df=None, title='', xlabels=[], bins=[],
                        yrange=None, binsrange=None, centroids=[], widths=[], coeffs=[],
                        save=False, save_path='', file_name=''):
    '''  
    Reconstruction of the data distribution learnt by the model.                           
    df:              (int) chi2 degrees of freedom                                         
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1) 
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)  
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)  
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample        
    tau_OBS:         (float) value of the tau term after training                        
    output_tau_ref:  (numpy array, shape (None, 1)) tau prediction of the reference training sample after training                                                                                 
    feature_labels:  (list of string) list of names of the training variables             
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)                                              
    delta_OBS:       (float) value of the delta term after training (if not given, only tau reconstruction is plotted)                                                                                 
    output_delta_ref:(numpy array, shape (None, 1)) delta prediction of the reference training sample after training (if not given, only tau reconstruction is plotted)                              
    '''
    # used to regularize empty reference bins                                            
    eps = 1e-10

    weight_ref = np.ones(len(ref))*weight_ref
    weight_data = np.ones(len(data))*weight_data
    colors = [
        '#a6cee3',
        '#1f78b4',
        '#b2df8a',
        '#33a02c',
        '#fb9a99',
        '#e31a1c',
        '#fdbf6f',
        '#ff7f00',
    ]
    font = font_manager.FontProperties(family='serif', size=12)
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    for i in range(data.shape[1]):
        if not len(bins):
            bins = np.linspace(np.min(ref[:, i]),np.max(ref[:, i]),50)
        if not binsrange==None:
            if len(binsrange[xlabels[i]]):
                bins=binsrange[xlabels[i]]
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax1= fig.add_axes([0.13, 0.43, 0.8, 0.5])
        hD = plt.hist(data[:, i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(ref[:, i], weights=weight_ref, color=colors[0], ec=colors[1], bins=bins, lw=1, label='REFERENCE', zorder=1)
        hN = []
        for j in range(len(ref_preds)):
            hN.append(plt.hist(ref[:, i], weights=np.exp(ref_preds[j][:, 0])*weight_ref, histtype='step', bins=bins, lw=0))

        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        for j in range(len(ref_preds)):
            plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[j][0], #edgecolor='black',                                                                                                                          
                        label=ref_preds_labels[j]+' RECO', color='none', edgecolor=colors[j+2], lw=2, s=30, zorder=4)
        l    = plt.legend(fontsize=18, prop=font, ncol=2, loc='best')
        title_leg=''
        if not t_obs==None:
            title_leg += 't='+str(np.around(t_obs, 2))
        if not df==None:
            Zscore=norm.ppf(chi2.cdf(t_obs, df))
            title_leg += ', Z-score='+str(np.around(Zscore, 2))
        if not title_leg=='':
            l.set_title(title=title_leg, prop=font)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        if title!='':
            plt.title(title, fontsize=22, fontname='serif')
        if len(xlabels):
            if '_pt' in xlabels[i] or 'm_' in xlabels[i]:
                plt.xlim(bins[0]+0.1, bins[-1])
                plt.xscale('log')

        ax2 = fig.add_axes([0.13, 0.1, 0.8, 0.3])
        x   = 0.5*(bins[1:]+bins[:-1])
        y = hD[0]/(hR[0]+eps)
        yerr = np.sqrt(hD[0])/(hR[0]+eps)
        log_yerr=yerr[y>0]/y[y>0]
        plt.errorbar(x[y>0], np.log(y[y>0]), yerr=log_yerr, ls='', fmt='none', color='black')
        plt.scatter(x[y>0], np.log(y[y>0]), s=30, color='none', edgecolors='black', label='DATA/REF')
        for j in range(len(ref_preds)):
            y = hN[j][0]/(hR[0]+eps)
            plt.plot(x[y>0], np.log(y[y>0]), label =ref_preds_labels[j]+' RECO/REF', color=colors[j+2],
                     #ls='--',                                                                                                                                                                            
                     lw=2)
        if len(centroids):
            plt.scatter(centroids[:, i], np.zeros((centroids.shape[0])), color='red', label='centroids')
        if len(centroids) and len(widths) and len(coeffs):
            y = np.zeros(ref.shape[0])
            for k in range(coeffs.shape[0]):
                if j%5: continue
                x = ref[:, i]
                y= coeffs[k]*norm.pdf(x, loc=centroids[k, 0], scale=widths[k])
                plt.plot(x, y,
                         color='red', lw=1,
                         alpha=(1+k)*1./len(centroids))
        plt.legend(loc='upper left', ncol=3, prop=font)
        if xlabels:
            plt.xlabel(xlabels[i], fontsize=22, fontname='serif')
        else:
            plt.xlabel('x', fontsize=22, fontname='serif')

        plt.ylabel("log-ratio", fontsize=22, fontname='serif')
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        if len(xlabels):
            if '_pt' in xlabels[i] or 'm_' in xlabels[i]:
                plt.xlim(bins[0]+0.1, bins[-1])
                plt.xscale('log')

        if not yrange==None and len(xlabels)>0:
            plt.ylim(yrange[xlabels[i]][0], yrange[xlabels[i]][1]) 
        plt.grid()
        if save:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(save_path+file_name)
        plt.show()
        plt.close()
    return


def plot_training_data(data, weight_data, ref, weight_ref, feature_labels, bins_code, xlabel_code, ymax_code={},
                       save=False, save_path='', file_name=''):
    '''
    Plot distributions of the input variables for the training samples.
    
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample
    feature_labels:  (list of string) list of names of the training variables
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)
    '''
    plt_i = 0
    for key in feature_labels:
        bins = bins_code[key]
        plt.rcParams["font.family"] = "serif"
        plt.style.use('classic')
        fig = plt.figure(figsize=(10, 10)) 
        fig.patch.set_facecolor('white')  
        ax1= fig.add_axes([0.1, 0.43, 0.8, 0.5])        
        hD = plt.hist(data[:, plt_i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=4)
        hR = plt.hist(ref[:, plt_i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE')
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18) 
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3]) 
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/hR[0], yerr=np.sqrt(hD[0])/hR[0], ls='', marker='o', label ='DATA/REF', color='black')
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)
        plt.xlabel(xlabel_code[key], fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')
        if key in list(ymax_code.keys()):
            plt.ylim(0., ymax_code[key])
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.grid()
        if save:
            if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
            else:
                if file_name=='': file_name = 'InputVariable_%s'%(key)
                else: file_name += '_InputVariable_%s'%(key)
                fig.savefig(save_path+file_name+'.pdf')
        plt.show()
        plt.close()
        plt_i+=1
    return