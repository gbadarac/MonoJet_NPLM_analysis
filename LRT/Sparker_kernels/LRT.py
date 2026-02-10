import glob, h5py, math, time, os, json, random, yaml, argparse, datetime, sys
from scipy.stats import norm, expon, chi2, uniform, chisquare
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist
from pathlib import Path
import torch

import numpy as np
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches

from torch.autograd.functional import hessian
from torch.autograd import grad

sys.path.insert(1, './utils/')
from SPARKutils import *
from PLOTutils import *
from GENutils import *
from sampler_mixture import *
from losses import *
from regularizers import *
import LRTGOFutils as lrt
import ENSEMBLEutils as ens

parser   = argparse.ArgumentParser()
parser.add_argument('-f', '--folderpath', type=str, help="input folder", required=True)
#parser.add_argument('-j', '--jsonfile', type=str, help="json file", required=True)
parser.add_argument('-e', '--nensemble', type=int, help="number of ensembled models", required=True)
parser.add_argument('-n', '--ntest', type=int, help="number of points in the test data", required=True)
parser.add_argument('-c', '--calibration', type=int, help="is it a calibration toy", required=True)
parser.add_argument('-s', '--seed', type=int, help="toy seed", required=False, default=None)
args     = parser.parse_args()

# random seed
seed = args.seed
if seed==None:
    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().min
#np.random.seed(seed)
print('Random seed:'+str(seed))

# train on GPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize ensemble
folder_path = args.folderpath
Ntest= args.ntest
lambda_regularizer = 1
n_kernels_numerator = 200
epochs = 10000
patience = 1000
kernel_width_numerator=0.4
lambda_L2_numerator = 1
test_id_string='Ntest%i_Lnorm%s/M%i_W%s_L%s'%(Ntest,
                                              str(lambda_regularizer),
                                              n_kernels_numerator,
                                              str(kernel_width_numerator),
                                              str(lambda_L2_numerator))

out_dir = folder_path+'/LRT_test/'+test_id_string
if args.calibration: out_dir+='/calibration/'
else: out_dir+='/test/'

os.makedirs(out_dir, exist_ok=True)
with open(folder_path+'config.json', 'r') as jsonfile:
    config_json = json.load(jsonfile)
n_kernels = config_json["number_centroids"]
n_layers = len(n_kernels)
model_type = config_json["model"]
split_indices = np.cumsum(n_kernels)[:-1]

n_wifi_components =args.nensemble
centroids_init, coefficients_init, widths_init = [], [], []
for i in range(n_wifi_components):
    tmp = np.load(folder_path+'/seed%i/widths_history.npy'%(i))
    count=-1
    for j in range(tmp.shape[0]):
        if tmp[j][0].sum(): count+=1
        else: 
            print(count)
            break
    centroids_init_i = np.load(folder_path+'/seed%i/centroids_history.npy'%(i))[count]
    coefficients_init_i = np.load(folder_path+'/seed%i/coeffs_history.npy'%(i))[count]
    widths_init_i = np.load(folder_path+'/seed%i/widths_history.npy'%(i))[count]
    
    centroids_init.append(centroids_init_i)
    coefficients_init.append(coefficients_init_i)
    widths_init.append(widths_init_i)

centroids_init  = np.stack(centroids_init, axis=0)
coefficients_init = np.stack(coefficients_init, axis=0)
coefficients_init = coefficients_init/np.sum(coefficients_init, axis=1, keepdims=True)
widths_init = np.stack(widths_init, axis=0)
print(centroids_init.shape, coefficients_init.shape, widths_init.shape)

weights_centralv = np.load(folder_path+'/final_weights.npy')
weights_cov_init = np.load(folder_path+'/covariance_final_weights.npy')

if args.calibration:
    # generate data from the ensemble
    folder_data = folder_path+'/generated_samples'
    data_all = np.concatenate([np.load(f) for f in glob.glob("%s/*.npy"%(folder_data))], axis=0)
else:
    # generate data from ground truth
    if 'two-moons' in folder_path:
        data_all=ens.generate_two_moons(N_train_tot=100_000, seed=1, noise=0.1)
    elif '2GMMskew' in folder_path:
        data_all=ens.generate_2GMMskew()
print('number of available data points:', data_all.shape[0])
np.random.seed(seed)
idx = np.random.choice(len(data_all), Ntest, replace=False)
bootstrap_sample = data_all[idx]

#evaluate the ensemble at the test points
model_probs = ens.pdf_components_from_ensemble(bootstrap_sample, 
                                           centroids_init, coefficients_init, widths_init)
model_probs = torch.from_numpy(np.stack(model_probs, axis=1)).float()


x_data = torch.from_numpy(bootstrap_sample).float().to(device)
noise_scale_init =np.abs(weights_cov_init.diagonal()).mean()
w_init = torch.from_numpy(weights_centralv+np.random.normal(scale=noise_scale_init,
                                                            size=weights_centralv.shape)).float()
w_centralv = torch.from_numpy(weights_centralv).float()
w_cov = torch.from_numpy(weights_cov_init).float()


# ------------------ TAU models (float32 everywhere) ------------------
# Denominator: no extra kernels
model_den = lrt.TAU(
    (None, 2),
    ensemble_probs=model_probs.to(device),
    weights_init=w_init.to(device),
    weights_cov=w_cov.to(device),
    weights_mean=w_centralv.to(device),
    gaussian_center=[],
    gaussian_coeffs=[],
    gaussian_sigma=None,
    lambda_regularizer=lambda_regularizer,
    train_net=False).to(device)

den_epochs, den_losses = lrt.train_loop(x_data, model_den, "DEN", epochs=int(epochs/1), lr=0.00001)

# Save loss curve
fig, ax = plt.subplots()
ax.plot(den_epochs, den_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Denominator loss")
fig.savefig(os.path.join(out_dir, "seed%i_denominator_loss.png"%(seed)), dpi=180, bbox_inches="tight")
plt.close(fig)
denominator = model_den.loglik(x_data).detach().cpu().numpy()

# Numerator
centers = x_data[:n_kernels_numerator].clone()  # already on device
coeffs  = torch.ones((n_kernels_numerator,), dtype=torch.float32, device=device) / n_kernels_numerator

model_num = lrt.TAU(
    (None, 2),
    ensemble_probs=model_probs.to(device),
    weights_init=w_init.to(device),
    weights_cov=w_cov.to(device),
    weights_mean=w_centralv.to(device),
    gaussian_center=centers.to(device),
    gaussian_coeffs=coeffs.to(device),
    gaussian_sigma=kernel_width_numerator,
    lambda_regularizer=lambda_regularizer,
    lambda_net=lambda_L2_numerator,
    train_net=True).to(device)

num_epochs, num_losses = lrt.train_loop(x_data, model_num, "NUM", epochs=epochs, lr=0.0001)

# Save loss curve (linear y-scale)
fig, ax = plt.subplots()
ax.plot(num_epochs, num_losses)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Numerator loss")
fig.savefig(os.path.join(out_dir, "seed%i_numerator_loss.png"%(seed)), dpi=180, bbox_inches="tight")
plt.close(fig)
numerator = model_num.loglik(x_data).detach().cpu().numpy()

test = numerator - denominator
print('numerator:', numerator,
      #model_num.normalization_constraint_term(),
      #model_num.network.coefficients, model_num.network.coefficients.sum(),
      #model_num.weights, model_num.weights.sum()
)
print('denominator:', denominator,
      #model_den.normalization_constraint_term(),
      #model_den.weights, model_den.weights.sum()
)
print('test: ', test)
# save test statistic                                                             
t_file=open(out_dir+'seed%i_t.txt'%(seed), 'w')
t_file.write("%f,%f,%f\n"%(test, numerator, denominator))
t_file.close()
