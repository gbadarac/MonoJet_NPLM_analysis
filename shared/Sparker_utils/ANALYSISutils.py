import glob, h5py, math, time, os, json
from scipy.stats import norm, expon, chi2, uniform, chisquare, gamma, beta, kstest
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')

def return_best_chi2dof(tobs):
    """
    Returns the most fitting value for dof assuming tobs follows a chi2_dof distribution,
    computed with a Kolmogorov-Smirnov test, removing NANs and negative values.
    Parameters
    ----------
    tobs : np.ndarray
        observations
    Returns
    -------
        best : tuple
            tuple with best dof and corresponding chi2 test result
    """

    dof_range = np.arange(np.nanmedian(tobs) - 10, np.nanmedian(tobs) + 10, 0.1)
    ks_tests = []
    for dof in dof_range:
        test, pv = kstest(tobs, lambda x:chi2.cdf(x, df=dof))
        ks_tests.append((dof, test, pv))
    ks_tests = [test for test in ks_tests if test[1] != 'nan'] # remove nans
    ks_tests = [test for test in ks_tests if test[0] >= 0] # retain only positive dof
    best = min(ks_tests, key = lambda t: t[1]) # select best dof according to KS test result
    return best

def Z_score_chi2(t,df):
    sf = chi2.sf(t, df)
    Z  = -norm.ppf(sf)
    return Z

def Z_score_gamma(t, a, loc, scale):
    sf = gamma.sf(t, a, loc, scale)
    Z  = -norm.ppf(sf)
    return Z

def Z_score_norm(t,mu, std):
    sf = norm.sf(t, mu, std)
    Z  = -norm.ppf(sf)
    return Z

def plot_1distribution(t, df, xmin=0, xmax=300, nbins=10, save=False, ymax=None, output_path='', save_name='', label=''):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution (df must be specified!). 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    Z_obs     = norm.ppf(chi2.cdf(np.median(t), df))
    t_obs_err = 1.2533*np.std(t)*1./np.sqrt(t.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t)-t_obs_err, df))
    label  = 'sample %s\nsize: %i \nmedian: %s, std: %s\n'%(label, t.shape[0], str(np.around(np.median(t), 2)),str(np.around(np.std(t), 2)))
    label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    binswidth = (xmax-xmin)*1./nbins
    h = plt.hist(t, weights=np.ones_like(t)*1./(t.shape[0]*binswidth), color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'$\chi^2_{%i}$'%(df))
    font = font_manager.FontProperties(family='serif', size=14) 
    plt.legend(prop=font, frameon=False)
    plt.xlabel('t', fontsize=18, fontname="serif")
    plt.ylabel('Probability', fontsize=18, fontname="serif")
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if ymax !=None:
        plt.ylim(0., ymax)
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(output_path+ save_name+'_distribution.pdf')
            print('saved at %s'%(output_path+ save_name+'_distribution.pdf'))
    plt.show()
    plt.close(fig)
    return

def plot_2distribution(t1, t2, df, xmin=0, xmax=300, ymax=None, nbins=10, save=False, output_path='', label1='1', label2='2', save_name='', print_Zscore=True):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution (df must be specified!).
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    binswidth = (xmax-xmin)*1./nbins
    # t1
    Z_obs     = Z_score_chi2(np.median(t1), df)
    t_obs_err = 1.2533*np.std(t1)*1./np.sqrt(t1.shape[0])
    Z_obs_p   = Z_score_chi2(np.median(t1)+t_obs_err, df)
    Z_obs_m   = Z_score_chi2(np.median(t1)-t_obs_err, df)
    label  = '%s \nsize: %i\nmedian: %s, std: %s\n'%(label1, t1.shape[0], str(np.around(np.median(t1), 2)),str(np.around(np.std(t1), 2)))
    if print_Zscore:
        label += 'asymptotic Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    
    h = plt.hist(t1, weights=np.ones_like(t1)*1./(t1.shape[0]*binswidth), color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t1.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    max1 = np.max(h[0])
    # t2
    Z_obs     = Z_score_chi2(np.median(t2), df)
    t_obs_err = 1.2533*np.std(t2)*1./np.sqrt(t2.shape[0])
    Z_obs_p   = Z_score_chi2(np.median(t2)+t_obs_err, df)
    Z_obs_m   = Z_score_chi2(np.median(t2)-t_obs_err, df)
    t_empirical = (1+1.*np.sum(t1>np.mean(t2)))*1./(t1.shape[0]+1)
    empirical_lim = '='
    if t_empirical==0:
        empirical_lim='>'
        t_empirical = 1./t1.shape[0]
    t_empirical_err = t_empirical*np.sqrt(1./np.sum(1.*(t1>np.mean(t2))+1./t1.shape[0]))
    Z_empirical = norm.ppf(1-t_empirical)
    Z_empirical_m = norm.ppf(1-(t_empirical+t_empirical_err))
    Z_empirical_p = norm.ppf(1-(t_empirical-t_empirical_err))
                                          
    label  = '%s \nsize: %i\nmedian: %s, std: %s\n'%(label2, t2.shape[0], str(np.around(np.median(t2), 2)),str(np.around(np.std(t2), 2)))
    if print_Zscore:
        label += 'asymptotic Z = %s (+%s/-%s) \n'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
        label += 'empirical Z %s %s (+%s/-%s)'%(empirical_lim, str(np.around(Z_empirical, 2)), str(np.around(Z_empirical_p-Z_empirical, 2)), str(np.around(Z_empirical-Z_empirical_m, 2)))
    h = plt.hist(t2, weights=np.ones_like(t2)*1./(t2.shape[0]*binswidth), color='#8dd3c7', ec='seagreen',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t2.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='seagreen', marker='o', ls='')
    max2 = np.max(h[0])
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'$\chi^2_{%i}$'%(df))
    font = font_manager.FontProperties(family='serif', size=20) #weight='bold', style='normal', 
    plt.legend(ncol=1, loc='upper right', prop=font, frameon=False)
    plt.xlabel('$t$', fontsize=32, fontname="serif")
    plt.ylabel('Probability', fontsize=32, fontname="serif")
    plt.ylim(0., 1.2*np.maximum(max1, max2))#np.max(chi2.pdf(x, df))*1.3)
    if ymax !=None:
        plt.ylim(0., ymax)
    plt.yticks(fontsize=22, fontname="serif")
    plt.xticks(fontsize=22, fontname="serif")
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(output_path+ save_name+'_2distribution.pdf')
    plt.show()
    plt.close()
    return [Z_obs, Z_obs_p, Z_obs_m], [Z_empirical, Z_empirical_p, Z_empirical_m]

def Plot_Percentiles_ref(tvalues_check, dof, patience=1, title='', wc=None, ymax=300, ymin=0, save=False, output_path=''):
    '''
    The funcion creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    The percentile lines for the target chi2 distribution (dof required!) are shown as a reference.
    patience: interval between two check points (epochs).
    tvalues_check: array of t=-2*loss, shape = (N_toys, N_check_points)
    '''
    colors = [
    'seagreen',
    'mediumseagreen',
    'lightseagreen',
    '#2c7fb8',
    'midnightblue',
    ]
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    epochs_check = []
    nr_check_points = tvalues_check.shape[1]
    for i in range(nr_check_points):
        epoch_check = patience*(i+1)
        epochs_check.append(epoch_check)
    
    fig=plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    quantiles=[2.5, 25, 50, 75, 97.5]
    percentiles=np.array([])
    plt.xlabel('Training Epochs', fontsize=16, fontname="serif")
    plt.ylabel('t', fontsize=16, fontname="serif")
    plt.ylim(ymin, ymax)
    if wc != None:
        plt.title('Weight Clipping = '+wc, fontsize=16,  fontname="serif")
    for i in range(tvalues_check.shape[1]):
        percentiles_i = np.percentile(tvalues_check[:, i], quantiles)
        #print(percentiles_i.shape)
        percentiles_i = np.expand_dims(percentiles_i, axis=1)
        #print(percentiles_i.shape)
        if not i:
            percentiles = percentiles_i.T
        else:
            percentiles = np.concatenate((percentiles, percentiles_i.T))
    legend=[]
    #print(percentiles.shape)
    for j in range(percentiles.shape[1]):
        plt.plot(epochs_check, percentiles[:, j], marker='.', linewidth=3, color=colors[j])
        #print(percentiles[:, j])
        legend.append(str(quantiles[j])+' % quantile')
    for j in range(percentiles.shape[1]):
        plt.plot(epochs_check, chi2.ppf(quantiles[j]/100., df=dof, loc=0, scale=1)*np.ones_like(epochs_check),
                color=colors[j], ls='--', linewidth=1)
        #print( chi2.ppf(quantiles[j]/100., df=dof, loc=0, scale=1))
        if j==0:
            legend.append("Target "+r"$\chi^2(dof=$"+str(dof)+")")
    font = font_manager.FontProperties(family='serif', size=16)         
    plt.legend(legend, prop=font)
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            fig.savefig(output_path+title+'_PlotPercentiles.pdf')
    plt.show()
    plt.close(fig)
    return

def ClopperPearson_interval(total, passed, level):
    low_b, up_b = 0.5*(1-level), 0.5*(1+level)
    low_q=beta.ppf(low_b, passed, total-passed+1, loc=0, scale=1)
    up_q=beta.ppf(up_b, passed, total-passed+1, loc=0, scale=1)
    return np.around(passed*1./total-low_q, 5), np.around(up_q-passed*1./total,5)

def z_to_p(z):
    return norm.sf(z)

def p_to_z(pvals):
    return norm.ppf(1 - pvals)
    
def power_emp(t_ref,t_data,zalpha=[1,2,3]):
    alpha = z_to_p(zalpha)
    #print(alpha)
    quantiles = np.quantile(t_ref,1-alpha,method='higher')
    total = len(t_data) 
    passed_list = [np.sum(t_data>=quantile) for quantile in quantiles]
    err_list = [ClopperPearson_interval(total, passed, level=0.68) for passed in passed_list]
    power_list = [(1+passed)/(1+total) for passed in passed_list]
    return p_to_z(alpha), power_list, err_list

def power_asympt_chi2(t_ref,t_data,df,zalpha=[1,2,3]):
    alpha = z_to_p(zalpha)
    #print(alpha)
    quantiles = chi2.ppf(1-alpha,df)
    total = len(t_data) 
    passed_list = [np.sum(t_data>=quantile) for quantile in quantiles]
    err_list = [ClopperPearson_interval(total, passed, level=0.68) for passed in passed_list]
    power_list = [passed/total for passed in passed_list]
    return p_to_z(alpha), power_list, err_list

def power_asympt_gamma(t_ref,t_data,a,loc=0,scale=1,zalpha=[1,2,3]):
    alpha = z_to_p(zalpha)
    #print(alpha)
    quantiles = gamma.ppf(1-alpha, a, loc, scale)
    total = len(t_data) 
    passed_list = [np.sum(t_data>=quantile) for quantile in quantiles]
    err_list = [ClopperPearson_interval(total, passed, level=0.68) for passed in passed_list]
    power_list = [passed/total for passed in passed_list]
    return p_to_z(alpha), power_list, err_list

def plot_2distribution_new(t1, t2, df, xmin=0, xmax=300, ymax=None, nbins=10, Z_print=[1,2,3], 
                           return_power=True, save=False, output_path='', label1='1', label2='2', 
                           save_name='', print_Zscore=True):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution (df must be specified!).
    Z_print: values of the Z-score for which to compute the power
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    ax= fig.add_axes([0.07, 0.07, 0.85, 0.85])
    
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    binswidth = (xmax-xmin)*1./nbins
    # t1
    label = '%s \nsize: %i, median: %s, std: %s\n'%(label1, t1.shape[0], 
                                                    str(np.around(np.median(t1), 2)),
                                                    str(np.around(np.std(t1), 2)))
    h = plt.hist(t1, weights=np.ones_like(t1)*1./(t1.shape[0]*binswidth), 
                 color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t1.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    max1 = np.max(h[0])
    # t2
    Z_list_emp, power_list_emp, err_list_emp = power_emp(t1,t2,zalpha=Z_print)
    Z_list_asymp, power_list_asymp, err_list_asymp = power_asympt_chi2(t1,t2,df=df,zalpha=Z_print)
    print(Z_list_emp, power_list_emp, err_list_emp)
    string_emp = ''
    for Z, pow, err in zip(Z_list_emp, power_list_emp, err_list_emp):
        string_emp+= r' P(Z>%s)=$%s^{+%s}_{-%s}$'%(str(int(Z)), 
                                                   str(np.around(pow,3)), 
                                                   str(np.around(err[0],2)), 
                                                   str(np.around(err[1],2)))
        string_emp+='\n'

    string_asymp = ''
    for Z, pow, err in zip(Z_list_asymp, power_list_asymp, err_list_asymp):
        string_asymp+= r' P(Z>%s)=$%s^{+%s}_{-%s}$'%(str(int(Z)),
                                                     str(np.around(pow,3)), 
                                                     str(np.around(err[0],2)), 
                                                     str(np.around(err[1],2)))
        string_asymp+='\n'
    
    label  = '%s \nsize: %i, median: %s, std: %s\n'%(label2, t2.shape[0], 
                                                     str(np.around(np.median(t2), 2)),
                                                     str(np.around(np.std(t2), 2)))
    if print_Zscore:
        extra_stats = ''
        extra_stats += 'asymptotic:\n'+string_asymp
        extra_stats += '\n'
        extra_stats += 'empirical:\n'+string_emp 
        ax.text(x=0.7, y=0., s=extra_stats, fontsize=20, fontname='serif', transform=ax.transAxes)
    h = plt.hist(t2, weights=np.ones_like(t2)*1./(t2.shape[0]*binswidth), 
                 color='#8dd3c7', ec='seagreen',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t2.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='seagreen', marker='o', ls='')
    max2 = np.max(h[0])
    
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 100)
    plt.plot(x, chi2.pdf(x, df),'midnightblue', lw=5, alpha=0.8, label=r'$\chi^2_{%i}$'%(df))
    font = font_manager.FontProperties(family='serif', size=20) 
    plt.legend(ncol=1, loc='upper right', prop=font, frameon=False)
    plt.xlabel('$t$', fontsize=32, fontname="serif")
    plt.ylabel('Probability', fontsize=32, fontname="serif")
    plt.ylim(0., 1.2*np.maximum(max1, max2))
    if ymax !=None:
        plt.ylim(0., ymax)
    plt.yticks(fontsize=22, fontname="serif")
    plt.xticks(fontsize=22, fontname="serif")
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(output_path+ save_name+'_2distribution.pdf')
    plt.show()
    plt.close()
    
    if return_power:
        return [Z_list_emp, power_list_emp, err_list_emp], [Z_list_asymp, power_list_asymp, err_list_asymp]
    else:
        return

def plot_2distribution_gamma(t1, t2, a, scale=1, loc=0, xmin=0, xmax=300, ymax=None, nbins=10, save=False, output_path='', label1='1', label2='2', save_name='', print_Zscore=True):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution (df must be specified!).
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    '''
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    binswidth = (xmax-xmin)*1./nbins
    # t1
    Z_obs     = Z_score_gamma(np.median(t1), a, loc=loc, scale=scale)
    t_obs_err = 1.2533*np.std(t1)*1./np.sqrt(t1.shape[0])
    Z_obs_p   = Z_score_gamma(np.median(t1)+t_obs_err, a, loc=loc, scale=scale)
    Z_obs_m   = Z_score_gamma(np.median(t1)-t_obs_err, a, loc=loc, scale=scale)
    label  = '%s \nsize: %i\nmedian: %s, std: %s\n'%(label1, t1.shape[0], str(np.around(np.median(t1), 2)),str(np.around(np.std(t1), 2)))
    if print_Zscore:
        label += 'asymptotic Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    
    h = plt.hist(t1, weights=np.ones_like(t1)*1./(t1.shape[0]*binswidth), color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t1.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    max1 = np.max(h[0])
    # t2
    Z_obs     = Z_score_gamma(np.median(t2), a, loc=loc, scale=scale)
    t_obs_err = 1.2533*np.std(t2)*1./np.sqrt(t2.shape[0])
    Z_obs_p   = Z_score_gamma(np.median(t2)+t_obs_err, a, loc=loc, scale=scale)
    Z_obs_m   = Z_score_gamma(np.median(t2)-t_obs_err, a, loc=loc, scale=scale)
    t_empirical = np.sum(1.*(t1>np.mean(t2)))*1./t1.shape[0]
    empirical_lim = '='
    if t_empirical==0:
        empirical_lim='>'
        t_empirical = 1./t1.shape[0]
    t_empirical_err = t_empirical*np.sqrt(1./np.sum(1.*(t1>np.mean(t2))+1./t1.shape[0]))
    Z_empirical = norm.ppf(1-t_empirical)
    Z_empirical_m = norm.ppf(1-(t_empirical+t_empirical_err))
    Z_empirical_p = norm.ppf(1-(t_empirical-t_empirical_err))
                                          
    label  = '%s \nsize: %i\nmedian: %s, std: %s\n'%(label2, t2.shape[0], str(np.around(np.median(t2), 2)),str(np.around(np.std(t2), 2)))
    if print_Zscore:
        label += 'asymptotic Z = %s (+%s/-%s) \n'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
        label += 'empirical Z %s %s (+%s/-%s)'%(empirical_lim, str(np.around(Z_empirical, 2)), str(np.around(Z_empirical_p-Z_empirical, 2)), str(np.around(Z_empirical-Z_empirical_m, 2)))
    h = plt.hist(t2, weights=np.ones_like(t2)*1./(t2.shape[0]*binswidth), color='#8dd3c7', ec='seagreen',
                 bins=bins, label=label)
    err = np.sqrt(h[0]/(t2.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='seagreen', marker='o', ls='')
    max2 = np.max(h[0])
    # plot reference chi2
    x  = np.linspace(gamma.ppf(0.0001, a, loc=loc, scale=scale), gamma.ppf(0.9999, a, loc=loc, scale=scale), 100)
    plt.plot(x, gamma.pdf(x, a, loc=loc, scale=scale),'midnightblue', lw=5, alpha=0.8, label=r'$\Gamma_{%s}$'%(str(np.around(a,2))))
    font = font_manager.FontProperties(family='serif', size=20) #weight='bold', style='normal', 
    plt.legend(ncol=1, loc='upper right', prop=font, frameon=False)
    plt.xlabel('$t$', fontsize=32, fontname="serif")
    plt.ylabel('Probability', fontsize=32, fontname="serif")
    plt.ylim(0., 1.2*np.maximum(max1, max2))#np.max(chi2.pdf(x, df))*1.3)
    if ymax !=None:
        plt.ylim(0., ymax)
    plt.yticks(fontsize=22, fontname="serif")
    plt.xticks(fontsize=22, fontname="serif")
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(output_path+ save_name+'_2distribution_gamma.pdf')
    plt.show()
    plt.close()
    return [Z_obs, Z_obs_p, Z_obs_m], [Z_empirical, Z_empirical_p, Z_empirical_m]


def plot_2distribution_gamma_new(t1, t2, a, scale=1, loc=0, xmin=0, xmax=300, ymax=None, nbins=10, Z_print=[1,2,3], 
                           return_power=True, save=False, output_path='', label1='1', label2='2', 
                           save_name='', print_Zscore=True, signal_type=-1, injection_ratio=1, embedding_type=""):
    '''
    Plot the histogram of a test statistics sample (t) and the target gamma distribution (a must be specified!).
    Z_print: values of the Z-score for which to compute the power
    '''
    
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor('white')
    ax= fig.add_axes([0.07, 0.07, 0.85, 0.85])
    #plot the title
    plt.plot([], [], ' ', label=embedding_type)
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    binswidth = (xmax-xmin)*1./nbins
    # t1
    label = ('%s \n'+r'$\overline{t}= %s,\,\sigma_t= %s$')%(label1, 
                                                    str(int(np.around(np.mean(t1), 0))),
                                                    str(int(np.around(np.std(t1), 0))))

    h = plt.hist(t1, weights=np.ones_like(t1)*1./(t1.shape[0]*binswidth), 
                  color='lightblue', ec='#2c7fb8',
                 bins=bins, label=label, alpha=.99)
    err = np.sqrt(h[0]/(t1.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    
    plt.errorbar(x, h[0], yerr = err, color='#2c7fb8', marker='o', ls='')
    max1 = np.max(h[0])
    # t2
    Z_list_emp, power_list_emp, err_list_emp = power_emp(t1,t2,zalpha=Z_print)
    Z_list_asymp, power_list_asymp, err_list_asymp = power_asympt_gamma(t1,t2, a, loc, scale,zalpha=Z_print)

    #Set upper bound to zero, in the edge case for P -> 1.0 (due to numerical instabilities)
    string_emp = ''
    for Z, pow, err in zip(Z_list_emp, power_list_emp, err_list_emp):
        if err[1] < 0:
            err = list(err) 
            err[1]=0
        elif np.isnan(err[0]) and np.isnan(err[1]):
            err = list(err)
            err[0]=0
            err[1]=0
        string_emp+= r' P(Z>%.2f)=$%.2f^{+%.2f}_{-%.2f}$'%(np.around(Z,2), 
                                                   np.around(pow,2), 
                                                   np.around(err[1],2), 
                                                   np.around(err[0],2))
        string_emp+='\n'

    string_asymp = ''
    for Z, pow, err in zip(Z_list_asymp, power_list_asymp, err_list_asymp):
        if err[1] < 0: 
            err = list(err)
            err[1]=0
        elif np.isnan(err[0]) and np.isnan(err[1]):
            err = list(err)
            err[0]=0
            err[1]=0
        string_asymp+= r' P(Z>%.2f)=$%.2f^{+%.2f}_{-%.2f}$'%(np.around(Z,2),
                                                     np.around(pow,2), 
                                                     np.around(err[1],2), 
                                                     np.around(err[0],2))
        string_asymp+='\n'
        if signal_type>0:
            label  = ('%s \n'+r'$\overline{t}= %s,\,\sigma_t= %s$')%(label2+f' ({signal_type_dict[signal_type]} {injection_ratio}%)', 
                                                                     str(int(np.around(np.mean(t2), 0))),str(int(np.around(np.std(t2), 0))))
        else:
            label=label2

    if print_Zscore:
        extra_stats = ''
        extra_stats += 'asymptotic:\n'+string_asymp
        extra_stats += '\n'
        extra_stats += 'empirical:\n'+string_emp 
        ax.text(x=0.7, y=0., s=extra_stats, fontsize=20, 
                transform=ax.transAxes)
    h = plt.hist(t2, weights=np.ones_like(t2)*1./(t2.shape[0]*binswidth), 
                 color='#8dd3c7', ec='seagreen',
                 bins=bins, label=label, alpha=.65)
    err = np.sqrt(h[0]/(t2.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    plt.errorbar(x, h[0], yerr = err, color='seagreen',marker='o', ls='')
    max2 = np.max(h[0])
    
    # plot reference gamma
    x  = np.linspace(gamma.ppf(0.0001, a, loc, scale), gamma.ppf(0.9999, a, loc, scale), 100)
    plt.plot(x, gamma.pdf(x, a, loc, scale),'midnightblue', lw=5, alpha=0.8, label=r'$\Gamma_{%s}$'%(str(np.around(a,1))))
    font = font_manager.FontProperties(size=20) 
    legend = plt.legend(ncol=1, loc='upper right', prop=font, frameon=False)
    plt.xlabel('$t$', fontsize=32)
    plt.ylabel('Probability', fontsize=32)
    plt.ylim(0., 1.2*np.maximum(max1, max2))
    if ymax !=None:
        plt.ylim(0., ymax)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.tick_params(axis='x', pad=15)  # Increase pad to move x-ticks down
    plt.tick_params(axis='y', pad=15)  # Increase pad to move y-ticks to the left
    if save:
        if output_path=='':
            print('argument output_path is not defined. The figure will not be saved.')
        else:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, save_name+'_2distribution_power.pdf'), bbox_inches="tight")
    plt.show()
    plt.close()
    
    if return_power:
        return [Z_list_emp, power_list_emp, err_list_emp], [Z_list_asymp, power_list_asymp, err_list_asymp]
    else:
        return



###### multiple test
import numpy as np
import sys
from scipy.stats import norm, beta
from scipy.special import logsumexp

# discrepancies are considered right-sided by default
# hence tests based on the p-value have a minus sign in front 
# because more discrepant results return smaller values

def ClopperPearson_interval(total, passed, level):
    low_b, up_b = 0.5*(1-level), 0.5*(1+level)
    low_q=beta.ppf(low_b, passed, total-passed+1, loc=0, scale=1)
    up_q=beta.ppf(up_b, passed, total-passed+1, loc=0, scale=1)
    return np.around(passed*1./total,5), np.around(passed*1./total-low_q, 5), np.around(up_q-passed*1./total,5)

def power(t_ref,t_data,zalpha=[.5,1,2,2.5]):
    # alpha=np.array([0.309,0.159,0.06681,0.0228,0.00620]))
    alpha = z_to_p(zalpha)
    #print(alpha)
    quantiles = np.quantile(t_ref,1-alpha,method='higher')
    total = len(t_data) 
    passed_list = [np.sum(t_data>=quantile) for quantile in quantiles]
    err_list = [ClopperPearson_interval(total, passed, level=0.68) for passed in passed_list]
    power_list = [passed/total for passed in passed_list]
    return p_to_z(alpha), power_list, err_list
    #return p_to_z(alpha), emp_pvalues_biased(t_data,quantiles)

def median_score(t_ref,t_data):
    quantiles = np.quantile(t_data,[0.16, 0.5, 0.84],method='higher')
    total = len(t_ref) 
    passed_list = [np.sum(t_ref>=quantile) for quantile in quantiles]
    err_list = [ClopperPearson_interval(total, passed, level=0.68) for passed in passed_list]
    power_list = [passed/total for passed in passed_list]
    return p_to_z(np.array(power_list)), power_list, err_list

def min_p(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    
    p_ref, p_data = return_pvalues(ref,data)
    return -np.min(p_ref,axis=1), -np.min(p_data,axis=1)
    #return -np.log(np.min(p_ref,axis=1)), -np.log(np.min(p_data,axis=1))

def fused_p(ref,data,T=1):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array
    
    p_ref, p_data = return_pvalues(ref,data)
    return -np.log(fusion(p_ref,-T)), -np.log(fusion(p_data,-T))


def avg_p(ref,data):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array

    p_ref, p_data = return_pvalues(ref,data)
    return -np.log(np.mean(p_ref,axis=1)), -np.log(np.mean(p_data,axis=1))
    #return -np.mean(p_ref,axis=1), -np.mean(p_data,axis=1)
    

def prod_p(ref,data):
    # ref: n_refxd numpy array
    # data: n_dataxd numpy array

    p_ref, p_data = return_pvalues(ref,data)
    return -np.sum(np.log(p_ref),axis=1), -np.sum(np.log(p_data),axis=1)
    #return -np.log(np.mean(p_ref,axis=1)), -np.log(np.mean(p_data,axis=1))

def fused_t(ref,data,T):
    fused_ref = fusion(ref,T)
    fused_data = fusion(data,T)

    return emp_pvalues(fused_ref,fused_data)

def emp_pvalue_biased(ref,t):
    # this definition is such that p=1/(N+1)!=0 if data is the most extreme value
    p = np.count_nonzero(ref > t) / (len(ref)) #+ (np.count_nonzero(ref > t)==0)*1. / (len(ref)) 
    return p

def emp_pvalue(ref,t):
    # this definition is such that p=1/(N+1)!=0 if data is the most extreme value
    p = (np.count_nonzero(ref > t)+1) / (len(ref)+1)
    return p

def emp_pvalues(ref,data):
    return np.array([emp_pvalue(ref,t) for t in data])

def emp_pvalues_biased(ref,data):
    return np.array([emp_pvalue_biased(ref,t) for t in data])

def p_to_z(pvals):
    return norm.isf(pvals)

def z_to_p(z):
    # sf=1-cdf (sometimes more accurate than cdf)
    return norm.sf(z)

def Zscore(ref,data):
    return p_to_z(emp_pvalues(ref,data))



def fusion(x,T):
    return T * logsumexp(1/T*x, axis=1, b=1/x.shape[1])

'''
#def bootstrap_pn(pn,rng=None):

    rnd = check_rng(rng)

    return rnd.choice(pn,size=len(pn))

#def bootstrap_pval(pn,t,rng=None):

    return emp_pvalue(bootstrap_pn(pn,rng=rng),t)


#def return_pvalues(ref,data,rng=None):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value by bootstrapping col (the value of t is removed first)
        p_ref[:,idx] = np.transpose([bootstrap_pval(np.delete(col,idx2),el,rng=rng) for idx2, el in enumerate(col)])

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data
'''

def return_pvalues(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p_ref[:,idx] = np.transpose([emp_pvalue(np.delete(col,idx2),el) for idx2, el in enumerate(col)])

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data


def return_pvalues_biased(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p_ref[:,idx] = np.transpose([emp_pvalue_biased(np.delete(col,idx2),el) for idx2, el in enumerate(col)])

    # p-values under the alternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues_biased(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data


def check_rng(rnd):
    if isinstance(rnd, np.random._generator.Generator):
        return rnd
    elif rnd is None:
        return np.random.default_rng(seed=rnd)
    elif isinstance(rnd, int):
        return np.random.default_rng(seed=rnd)
    else: 
        sys.exit("rnd must be None, an integer or an instance of np.random._generator.Generator")



def return_pvalues_2(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p=emp_pvalues(col,col)
        p_ref[:,idx] = np.transpose(p)

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data

def return_pvalues_2_biased(ref,data):
    # ref: nxd numpy array
    # data: mxd numpy array
    p_ref = np.zeros_like(ref)
    p_data = np.zeros_like(data)

    # p-values under the null - loop over results for a given test - all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(ref)):
        # for each t in col (all toys for a given sigma), compute p-value with respect to the other ts
        p=emp_pvalues_biased(col,col)
        p_ref[:,idx] = np.transpose(p)

    # p-values under the laternative - computed as usual for all values of t (col) for a given sigma (idx)
    for idx, col in enumerate(np.transpose(data)):
        p=emp_pvalues_biased(np.transpose(ref)[idx],col)
        p_data[:,idx] = np.transpose(p)

    return p_ref, p_data