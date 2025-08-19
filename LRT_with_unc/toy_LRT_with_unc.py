import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch, traceback
from torchmin import minimize
from torch import nn
from torch.autograd import Variable
import argparse
from utils_LRT import TAU
import os 
from utils_flows import make_flow
import gc 
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #sets device to GPU if available 

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--toys', type=int, required=True, help="Number of toys")
parser.add_argument('-c', "--calibration", type=str, default="True",
                    help="Enable calibration mode (True/False)")
parser.add_argument('--w_path', type=str, help="Path to fitted weights .npy")
parser.add_argument('--w_cov_path', type=str, help="Path to weights covariance .npy")
parser.add_argument('--hit_or_miss_data', type=str,
                    help="Path to hit-or-miss MC samples (needed if calibration=True)")
parser.add_argument('--out_dir', type=str, required=True, help="Base output directory")
parser.add_argument('--ensemble_dir', type=str, required=True, help="Directory containing the ensemble models")
args = parser.parse_args()

# Convert string to bool
calibration = args.calibration.lower() == "true"

# Load reference model 

# Load reference model weights and covariance of the weights 
# Optional weights
w_init = np.load(args.w_path).reshape(-1) if args.w_path else None
w_cov = np.load(args.w_cov_path)     if args.w_cov_path else None
if (w_init is None) ^ (w_cov is None):
    # If only one is given, ignore both to avoid half-configured state
    w_init, w_cov = None, None

# Out dir decided by mode
out_dir = os.path.join(args.out_dir, "calibration" if calibration else "comparison")
os.makedirs(out_dir, exist_ok=True)
print(f"Output directory set to: {out_dir}", flush=True)

N_events = 1000

def generate_target_data(n_points, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean_feat1, std_feat1 = -0.5, 0.25
    mean_feat2, std_feat2 = 0.6, 0.4
    feat1 = np.random.normal(mean_feat1, std_feat1, n_points)
    feat2 = np.random.normal(mean_feat2, std_feat2, n_points)
    return np.column_stack((feat1, feat2)).astype(np.float32)

# Ground truth data
if calibration:
    if args.hit_or_miss_data:
        data = np.load(args.hit_or_miss_data)[:N_events].astype(np.float32)
    else:
        raise ValueError("Calibration mode requires --hit_or_miss_data")
else:
    data = generate_target_data(N_events, seed=args.toys)

if w_init is not None and w_cov is not None:
    print(f"Loaded w shape: {w.shape}, w_cov shape: {w_cov.shape}")
print(f"Calibration mode: {calibration}")
print(f"Data shape: {data.shape}")

x_data = torch.from_numpy(data).float()


# Load ensemble 
f_i_file = os.path.join(args.ensemble_dir, "f_i.pth")
cfg_file = os.path.join(args.ensemble_dir, "architecture_config.json")

# Load on CPU directly to avoid any accidental GPU spikes
f_i_statedicts = torch.load(f_i_file, map_location="cpu")
with open(cfg_file) as f:
    config = json.load(f) 

# ------------------
# Evaluate f_i(x) -> model_probs
# ------------------
model_probs_list = []

for state_dict in f_i_statedicts:
    flow = make_flow(
        num_layers=config["num_layers"],
        hidden_features=config["hidden_features"],
        num_bins=config["num_bins"],
        num_blocks=config["num_blocks"],
        num_features=config["num_features"]
    )

    flow.load_state_dict(state_dict)
    flow = flow.to("cpu")  # keep model on CPU

    flow.eval()
    batch_size = 5000 #3200
    flow_probs = []

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            x_batch = x_data[i:i+batch_size].to("cpu")  # keep data on CPU too
            logp_batch = flow.log_prob(x_batch)
            flow_probs.append(torch.exp(logp_batch))

    flow_probs_tensor = torch.cat(flow_probs, dim=0).detach()

    model_probs_list.append(flow_probs_tensor)

    # Cleanup
    del flow_probs
    del flow
    torch.cuda.empty_cache()
    gc.collect()

model_probs = torch.stack(model_probs_list, dim=1).to("cpu").requires_grad_()

# fit the weights 

def probs(weights, model_probs):
    # weights: (M,), model_probs: (N, M)
    return (model_probs * weights).sum(dim=1)  # returns (N,)

def aux(weights, weights_0, weights_cov):
    '''
    auxiliary likelihood term.
    it's the likelihood of the nuisance parameters. 
    It's assumed Multivariate Gaussian with Covariance matrix given by the error covariance matrix 
    obtained fitting the {w} to the ensemble  
    '''
    d = torch.distributions.MultivariateNormal(weights_0, covariance_matrix=weights_cov)
    return d.log_prob(weights) 
    
def nll_aux(weights, weights_0, weights_cov):
    '''
    negative log-likelihood (nll) with auxiliary (aux) term included
    '''
    p = probs(weights, model_probs)
    if not torch.all(p > 0):
        # Use same device and dtype
        return weights.sum() * float("inf")
    loss = -torch.log(p + 1e-8).sum() - aux(weights, weights_0, weights_cov).sum()
    return loss

# fit model with extra kernel layer and uncertainties 
epochs=100000

# TEST DENIMONINATOR of the likelihood rato test 
model = TAU((None, 2), ensemble_probs=model_probs, weights_init=w_init, weights_cov=w_cov, weights_mean=w, 
            gaussian_center=[], gaussian_coeffs=[], gaussian_sigma=None,
            lambda_regularizer=1e6, train_net=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_hist_den = []
epoch_hist_den = []
# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()          # Reset gradients
    loss = model.loss(torch.from_numpy(data).float())       # Forward pass
    loss.backward()                # Backpropagation
    optimizer.step()                # Update weights

    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
        loss_hist_den.append(loss.item())
        epoch_hist_den.append(epoch)

plt.plot(epoch_hist_den, loss_hist_den)

denominator = model.loglik(data).detach().numpy()
print('Denomiantor:', denominator)

# TEST NUMERATOR of the likelihood ratio test
# additional degrees of freedom for the hypothesis at the numerator
n_kernels=20
centers = torch.from_numpy(data[:n_kernels])
coeffs = torch.ones((n_kernels,))/n_kernels

model = TAU((None, 2), ensemble_probs=model_probs, weights_init=w_init, weights_cov=w_cov, weights_mean=w, 
            gaussian_center=centers, gaussian_coeffs=coeffs, gaussian_sigma=0.3,
            lambda_regularizer=1e6, train_net=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_hist_num = []
epoch_hist_num = []
# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()          # Reset gradients
    loss = model.loss(torch.from_numpy(data).float()) # Forward pass
    loss.backward()                # Backpropagation
    optimizer.step()                # Update weights

    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
        loss_hist_num.append(loss.item())
        epoch_hist_num.append(epoch)

plt.plot(loss_hist_num[0:])
plt.yscale('log')

numerator = model.loglik(torch.from_numpy(data).float()).detach().numpy()
print('Numerator:', numerator)

print('Mixture of ensemble and extra DOF. Coefficients:', model.get_coeffs().detach().numpy())

print('Ensemble weights after training: ', model.weights.detach().numpy())







def plot_marginal(weights, model_probs, x_limits, ground_truth_generator, ground_truth_mean, ground_truth_cov, 
                  N_models, N_events, N_bumps, bumps_width, bumps_amplitude, axis=0, nn=None, nn_coeff=None):
    x = np.linspace(x_limits[0], x_limits[1], 10000)
    y = np.zeros_like(x)
    if axis==0:
        z = np.stack([x, y], axis=1)
    elif axis==1:
        z = np.stack([y, x], axis=1)
    
    # ground truth
    x_gt = ground_truth_generator.pdf(z)#*N_events
    
    # fitting output
    model_probs = []
    for n in range(N_models):
        np.random.seed(n)
        probs_n = noised_model(z, ground_truth_mean=ground_truth_mean, 
                             ground_truth_cov=ground_truth_cov, 
                             N_events=N_events, n_noise_bumps=N_bumps, noise_width=bumps_width, epsilon=bumps_amplitude)
        model_probs.append(probs_n)
    model_probs = torch.from_numpy(np.stack(model_probs, axis=1))
    print(model_probs.shape)
    
    plt.plot(x, x_gt, label='Ground truth')
    plt.plot(x, probs(weights, model_probs), label='Ensemble')
    if nn!=None:
        net_out=nn.forward(torch.from_numpy(z).float()).detach().numpy().squeeze()
        #p = 0.5*(1+nn_coeff[0])*probs(weights, model_probs) + 0.5*nn_coeff[1]*net_out 
        p = nn_coeff[0]*probs(weights, model_probs) + nn_coeff[1]*net_out 
        plt.plot(x, p, label='ensemble + extra DOF')
        plt.plot(x, coeffs[1]*net_out, label='extra DOF alone')
        plt.plot(x,nn_coeff[0]*probs(weights, model_probs), label='ensemble alone')
    plt.xlabel('Axis %i'%(axis))
    plt.legend()
    plt.show()
    return

x_limits=[-10, 10]

plot_marginal(w_init, model_probs, x_limits, ground_truth_generator, ground_truth_mean, ground_truth_cov, 
                  N_models, N_events, N_bumps, bumps_width, bumps_amplitude, axis=0, nn=model.network, nn_coeff=model.get_coeffs().detach().numpy())

#training data distribution
plt.hist(data[:, 0], bins=np.linspace(-10, 10, 200))
plt.show()

for k in model.network.parameters():
    print(k)

test= numerator - denominator
print('test: ', test)
