import torch, json, os
import numpy as np
import ANALYSISutils as an
import GENutils as gen


def save_np_array(array, name, json_path, seed=None):
    # setup output_folder                                                                                            
    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)
    OUTPUT_PATH    = config_json["output_directory"]
    OUTPUT_FILE_ID=''
    if not seed==None:
        OUTPUT_FILE_ID = '/seed%s/'%(seed)
    folder_out = OUTPUT_PATH+OUTPUT_FILE_ID
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    np.save(folder_out+name+'.npy', array)
    print('file saved at ', folder_out+name+'.npy')
    return

def compute_empirical_means(data, labels):
    means = []
    labels_unique=torch.unique(labels)
    for label in labels_unique:
        data_label=data[labels==label]
        means.append(torch.mean(data_label, axis=0))
    return torch.stack(means) # n x d

def pairwise_dist(X, P):
    X2 = (X ** 2).sum(dim=1, keepdim=True) # (n x 1)                                                            
    P2 = (P ** 2).sum(dim=1, keepdim=True) # (n' x 1)                                                            
    XP = X @ P.T # (n x n')                                                                                      
    return X2 + P2.T - 2 * XP # (n x n')  

def compute_empirical_cov_matrix(data, labels, means):
    """
    Compute the covariance matrix from data.
    
    Parameters:
        data : np.ndarray of shape (n_samples, n_features)
            The input data.
        labels : np.ndarray of shape (n_samples,), optional
            The class labels. If provided, compute per-class covariance matrices.
    
    Returns:
        cov : np.ndarray (if labels is None)
            The empirical covariance matrix of the data.
            
        cov_dict : dict[label] = np.ndarray (if labels is not None)
            A dictionary of class-wise covariance matrices.
    """
    if labels is None:
        # Center the data
        X_centered = data - np.mean(data, axis=0)
        cov = np.dot(X_centered.T, X_centered) / (data.shape[0] - 1)
        return cov
    else:
        cov_all = np.array([])
        i=0
        for label in np.unique(labels):
            class_data = data[labels == label]
            class_mean = means[i]#np.mean(class_data, axis=0)
            centered = class_data - class_mean
            cov = np.dot(centered.T, centered) / (class_data.shape[0] - 1)
            if i==0: cov_all = cov
            else: cov_all +=cov
            i+=1
        cov_all = cov_all/len(data)
        return torch.from_numpy(cov_all)
'''
def compute_empirical_cov_matrix(data, labels, means):
    N = len(data)
    empirical_cov = 0
    labels_unique=torch.unique(labels)
    for i in range(labels_unique.shape[0]):
        mean_label = means[i, :].reshape((1, -1)) # [1, d]
        label = i
        data_label = data[labels==label]
        dist_sq = pairwise_dist(data_label, mean_label) 
        empirical_cov += torch.sum(dist_sq)
    return empirical_cov/N
'''
def mahalanobis_test(data, means, cov, eps=1e-16):
    cov_inv = torch.linalg.inv(cov+ eps * torch.eye(cov.shape[-1], device=cov.device))
    dist  = torch.subtract(data[:, None, :], means[None, :, :]) # [n, n', d]
    #dist_sq = -1*torch.sum(dist_sq, dim=2)/cov # [n, n']
    #dist_sq = torch.matmul(dist_sq,  cov_inverse dist_sq.transpose()
    m = -1* torch.einsum('nci,ij,ncj->nc', dist, cov_inv, dist)
    return torch.max(m, dim=1)[0]

def mahalanobis_routine(seed, json_path, calibration=False, rule='sum'):
    '''
    generates a toy using seed.
    computes the mahalnobis test for the dataset.
    ARGS:
    - seed: random seed to extract the toy
    - json_path: path to json file containing the configuration distionary
    - calibration: boolean switching between signal injection and bkg-only toys.
                   if True, the toys are bkg-only
    RETURN:
    - t: the value of the test statistic. 
         If rule='sum', t is obtained summing the Mahalanobis distance of all events.
         If rule='max', t is the maximum Mahalanobis distance among all events.
    - M_data: the Mahalanobis distance for each event in the sample
    '''
    # random seed                                                                                                                    
    if seed==None:
        seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
    np.random.seed(seed)
    print('Random seed: '+str(seed))

    # setup parameters
    with open(json_path, 'r') as jsonfile:
        config_json = json.load(jsonfile)
    plot =  config_json["plot"]
    # problem definition                                                                                                             
    N_ref      = config_json["N_Ref"]
    N_Bkg      = config_json["N_Bkg"]
    N_Sig      = config_json["N_Sig"]
    if calibration: N_Sig=0
    #z_ratio    = config_json["luminosity_ratio"]
    Pois_ON    = config_json["Pois_ON"]
    anomalous_class = config_json["anomalous_class"]
    anomaly_type = config_json["anomaly_type"]
    sig_labels = [anomalous_class]
    #print('SIG classes: ', sig_labels)
    ##### define output path ######################                                                                                  
    OUTPUT_PATH    = config_json["output_directory"]
    OUTPUT_FILE_ID = '/seed%s/'%(seed)
    folder_out = OUTPUT_PATH+OUTPUT_FILE_ID
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    output_folder = folder_out

    files = np.load(config_json['ref_path'])
    ref_all_x, ref_all_y = files['data'], files['labels']
    
    if calibration:
        # use the training sample
        files = np.load(config_json['ref_path'])
        data_all_x, data_all_y = files['data'], files['labels']
    else:
        # use the domain shifted sample
        files = np.load(config_json['data_path'])
        data_all_x, data_all_y = files['data'], files['labels']
    
    # build the dataset 
    ref_all_x = ref_all_x[ref_all_y!=anomalous_class]
    ref_all_y = ref_all_y[ref_all_y!=anomalous_class]
    sig_all_x = data_all_x[data_all_y==anomalous_class]
    sig_all_y = data_all_y[data_all_y==anomalous_class]
    bkg_all_x = data_all_x[data_all_y!=anomalous_class]
    bkg_all_y = data_all_y[data_all_y!=anomalous_class]
    
    # standardize                                                                                                                    
    mean_all, std_all = np.mean(ref_all_x, axis=0), np.std(ref_all_x, axis=0)
    std_all[std_all==0] = 1
    #print(mean_all, std_all)
    ref_all_x = gen.standardize(ref_all_x, mean_all, std_all)
    bkg_all_x = gen.standardize(bkg_all_x, mean_all, std_all)
    sig_all_x = gen.standardize(sig_all_x, mean_all, std_all)
    
    N_bkg_p = np.random.poisson(lam=N_Bkg, size=1)[0]
    N_sig_p = np.random.poisson(lam=N_Sig, size=1)[0]
    
    idx_bkg = np.arange(bkg_all_y.shape[0])
    np.random.shuffle(idx_bkg)
    idx_sig = np.arange(sig_all_y.shape[0])
    np.random.shuffle(idx_sig)
    
    data_x = np.concatenate((sig_all_x[idx_sig[:N_sig_p]], bkg_all_x[idx_bkg[:N_bkg_p]]), axis=0)
    
    # estimate parameters of the bkg model 
    ref_all_x, ref_all_y = torch.from_numpy(ref_all_x), torch.from_numpy(ref_all_y)
    means=compute_empirical_means(ref_all_x, ref_all_y)
    #print(means)
    emp_cov=compute_empirical_cov_matrix(ref_all_x, ref_all_y, means)
    #print(emp_cov)
    
    # compute the Mahalnobis distance on the data
    data_x = torch.from_numpy(data_x)#feature[target[:, 0]==1]
    M_data = mahalanobis_test(data_x, means, emp_cov)
    
    if plot:
        # compute the Mahalnobis distance on the reference 
        # (just for comparison)
        M_ref = mahalanobis_test(ref_x, means, emp_cov)
        # visualize mahalanobis
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        plt.hist([M_ref, M_data], density=True, label=['REF', 'DATA'])
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font, ncol=2, loc='best')
        plt.yscale('log')
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.ylabel("density", fontsize=22, fontname='serif')
        plt.xlabel("mahalanobis metric", fontsize=22, fontname='serif')
        plt.savefig(output_folder+'distribution.pdf')
        plt.show()
    replacement_value = -1e9
    M_data[M_data == -float('inf')] = replacement_value
    # compute the test as the reduce sum of the mahalanobis distance over the dataset
    if rule=='sum':
        t = -1* torch.sum(M_data)
    elif rule=='max':
        t = -1* torch.min(M_data)
    #t_file=open(output_folder+'t.txt', 'w')
    #t_file.write("%f\n"%(t))
    #t_file.close()
    #M_data[np.isinf(-1*M_data)]=-1000000
    #print(torch.min(M_data))
    print('Mahalanobis test: ', "%f"%(t), 'rule: ', rule)
    return t, M_data

def compute_statistics(json_path, Ntoys, output_path=None, power_thr=[1.645,2.33], rule='sum'):
    '''
    if a list of power thresholds are given, computes the power of the test at specified thresholds.
    computes the z-scores at 50%, 16%, 84% flase positive rate.
    ARGS:
    - json_path: path to json file containing the configuration distionary
    - Ntoys: number of toys to run for the test statistic distributions
    - rule: aggregation mode to compute the Mahalanobis test
    - power_thr: list of thresholds to compute the test power (results showed in the plot)
    - output_path: path to store figure. If None, the figure is not saved
    RETURN:
    - z_emp: numpy array of the z-score at 16%, 50%, 84% false negative rate (shape: [3,])
    '''
    if output_path==None: save=False
    else: save=True
    t_null, t_alt = [], []
    for seed in range(Ntoys):
        t_null.append(mahalanobis_routine(seed, json_path, calibration=True, rule=rule)[0])
        t_alt.append(mahalanobis_routine(seed, json_path, calibration=False, rule=rule)[0])
    t_null, t_alt = np.array(t_null), np.array(t_alt)
    save_np_array(t_null, 't_null', json_path, seed=None)
    save_np_array(t_alt, 't_alt', json_path, seed=None)
    if len(power_thr):
        # compute the power of the test at specified thresholds
        z_as, z_emp = an.plot_2distribution_new(t_null, t_alt, df=np.median(t_null), 
                                             xmin=np.min(t_null)-10, xmax=np.max(t_alt)+10, 
                                             nbins=8, save=save, output_path=output_path, 
                                             Z_print=power_thr,
                                             label1='REF', label2='DATA', 
                                             save_name='/combined-pvals_with_powers', 
                                             print_Zscore=True)
        z_as, z_emp = an.plot_2distribution(t_null, t_alt, df=np.mean(t_null),
                                             xmin=np.min(t_null)-10, xmax=np.max(t_alt)+10,
                                             nbins=8, save=save, output_path=output_path,
                                             #Z_print=power_thr,                                                                      
                                             label1='REF', label2='DATA',
                                             save_name='/combined-pvals_with_medianZscore',
                                             print_Zscore=True)
    # compute z-scores at 50%, 16%, 84%
    z_emp = an.median_score(t_null,t_alt)[0]
    print("Z-score(FNR): %s (0.50), %s (0.16), %s (0.84)"%(str(np.around(z_emp[1], 2)), str(z_emp[0]), str(z_emp[2])))
    return z_emp
