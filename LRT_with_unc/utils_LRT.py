import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch, traceback
from torchmin import minimize
from torch import nn
from torch.autograd import Variable
import math 

import math, torch
from torch import nn

class GaussianKernelLayer(nn.Module):
    def __init__(self, centers_init, coefficients_init, sigma, train_centers=False):
        super().__init__()
        self.centers = nn.Parameter(centers_init.float(), requires_grad=train_centers)
        self.register_buffer('sigma', torch.tensor(float(sigma), dtype=torch.float32))
        self.coefficients = nn.Parameter(coefficients_init.float(), requires_grad=True)

        # --- auto mode detection from init ---
        with torch.no_grad():
            s = coefficients_init.float().sum()
            L1 = coefficients_init.float().abs().sum()
        tol = 1e-7
        if L1 < tol:                 # all zeros -> zero-sum correction
            self._mode = "zero_sum"
        elif abs(float(s) - 1.0) < 1e-6:
            self._mode = "sum1"      # sums to 1 -> mixture weights
        else:
            self._mode = "raw"       # leave as-is

        d = centers_init.shape[1]
        self.norm_const = (1.0 / ((2*math.pi)**(d/2) * (self.sigma**d)))

    def get_coefficients(self):
        a = self.coefficients
        '''
        if self._mode == "zero_sum":
            # keep sum(a)=0 at all times
            return a - a.mean()
        elif self._mode == "sum1":
            # keep sum(a)=1 despite updates (allow negatives)
            return a / (a.sum() + 1e-12)
        else:
        '''
        return a

    def forward(self, x):
        x = x.float()
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        kern = self.norm_const * torch.exp(-0.5 * dist_sq / (self.sigma ** 2))  # (N, m)
        # remove the per-sample DC component so gradients aren’t washed out
        kern = kern - kern.mean(dim=1, keepdim=True)
        return torch.einsum("m,Nm->N", self.get_coefficients(), kern)

class TAU(nn.Module):
    """ 
    N= input_shape[0]
    m= number of elements in the ensemble
    """
    def __init__(self, input_shape, ensemble_probs, weights_init, weights_cov, weights_mean, 
                 gaussian_center, gaussian_coeffs,
                 lambda_regularizer=1e5, lambda_net=1e-6, gaussian_sigma=0.1, train_centers=False,
                 train_weights=True, train_net=True, 
                 model='TAU', name=None, **kwargs):
        super(TAU, self).__init__()
        self.lambda_net = float(lambda_net)  
        self.ensemble_probs = ensemble_probs.float()   # [N, m] -> float32
        self.x_dim = input_shape[1]
        print("problem dimensionality:", self.x_dim)
        self.n_ensemble = self.ensemble_probs.shape[1]

        # store priors as float32 (if provided as tensors)
        self.weights_cov  = weights_cov.float()  if isinstance(weights_cov,  torch.Tensor) else weights_cov
        self.weights_mean = weights_mean.float() if isinstance(weights_mean, torch.Tensor) else weights_mean
        self.train_weights = train_weights

        self.lambda_regularizer = lambda_regularizer
        if (self.weights_mean is not None) and (self.weights_cov is not None):
            #self.aux_model = torch.distributions.MultivariateNormal(self.weights_mean, covariance_matrix=self.weights_cov)
            # ensure dtype/device
            dtype  = self.ensemble_probs.dtype
            device = self.ensemble_probs.device
            self.weights_mean = self.weights_mean.to(dtype=dtype, device=device)
            self.weights_cov  = self.weights_cov.to(dtype=dtype, device=device)

            # symmetrize
            cov = 0.5 * (self.weights_cov + self.weights_cov.T)

            # try Cholesky with increasing jitter
            eye = torch.eye(cov.shape[0], dtype=dtype, device=device)
            jitter = 1e-6 * (cov.diagonal().abs().mean() + 1.0)
            L = None
            for _ in range(7):
                try:
                    L = torch.linalg.cholesky(cov + jitter * eye)
                    break
                except RuntimeError:
                    jitter *= 10.0

            if L is None:
                # eigenvalue clip fallback
                evals, evecs = torch.linalg.eigh(cov)
                evals = torch.clamp(evals, min=1e-8)
                cov = (evecs * evals) @ evecs.T
                L = torch.linalg.cholesky(cov)

            # build MVN using scale_tril so torch doesn't re-factorize
            self.aux_model = torch.distributions.MultivariateNormal(self.weights_mean, scale_tril=L)

        else:
            self.aux_model = None

        # trainable weights (float32)
        self.weights = nn.Parameter(weights_init.reshape((self.n_ensemble)).float(),
                                    requires_grad=train_weights)  # [M,]

        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.train_net = train_net

        if self.train_net:
            '''
            self.coeffs = nn.Parameter(torch.tensor([1, 0.], dtype=torch.float32), requires_grad=True)
            '''
            self.network = GaussianKernelLayer(gaussian_center, gaussian_coeffs, gaussian_sigma, train_centers=train_centers)
            # self.network = nn.Sequential(nn.Linear(self.x_dim, 1000), nn.Sigmoid(),
            #                              nn.Linear(1000, 1), nn.Sigmoid())  # flow

    def call(self, x):
        x = x.float()
        ensemble = torch.einsum("ij,jk->ik", self.ensemble_probs, self.weights.unsqueeze(1))  # (N,1)
        # print(ensemble.shape)
        if self.train_net:
            net_out = self.network(x)
            return ensemble, net_out
        else:
            return ensemble

    def get_coeffs(self):
        return self.softmax(self.coeffs)

    def net_coeffs_L2(self):
        return torch.sum(self.network.get_coefficients()**2)
        
    def weights_constraint_term(self):
        return (torch.sum(self.weights) - 1.0)**2/self.weights.shape[0]

    def amplitudes_constraint_term(self):
        return (torch.sum(self.network.get_coefficients()) - 1.0)**2/self.network.get_coefficients().shape[0]
        
    def net_constraint_term(self):
        squared_sum = 0.0
        for param in self.network.parameters():
            if param.requires_grad:
                squared_sum += torch.sum(param ** 2)
        return squared_sum
    
    def log_auxiliary_term(self):
        return self.aux_model.log_prob(self.weights) 
    
    def normalization_constraint_term(self):
        # somma pesi w_i
        sum_w = torch.sum(self.weights)
        # somma ampiezze a_j se la rete esiste, altrimenti 0
        if self.train_net:
            sum_a = torch.sum(self.network.get_coefficients())
            if self.train_weights:
                denom = self.weights.shape[0] + self.network.get_coefficients().shape[0]
            else:
                denom = self.network.get_coefficients().shape[0] 
        else:
            sum_a = torch.tensor(0.0, dtype=self.weights.dtype, device=self.weights.device)
            denom = self.weights.shape[0]
        denom = max(1, int(denom))
        print("denom=", denom)
        total = sum_w + sum_a

        # <<< simple debug print >>>
        print(f"[norm] sum_w={sum_w.item():.6f}, sum_a={sum_a.item():.6f}, total={total.item():.6f}", flush=True)

        return ((total - 1.0) ** 2) / denom

    def loglik(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)          # ensemble: (N,1) ; net_out: (N,)
            p = ensemble[:, 0] + net_out
            return torch.log(p).sum() + aux.sum()
        else:
            p = self.call(x)                           # (N,1)
        out = torch.log(p).sum() 
        if self.train_weights:
            out = out + aux.sum()
        return out 

    def loss(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)
            p = self.relu(ensemble[:, 0] + net_out)
            out = -torch.log(p).sum() + self.lambda_regularizer * self.normalization_constraint_term() \
                + self.lambda_net * self.net_constraint_term()
            if self.train_weights:
                out = out - aux.sum()
            return out 
        else:
            p = self.relu(self.call(x))
            out = -torch.log(p).sum() + self.lambda_regularizer * self.normalization_constraint_term()
            if self.train_weights:
                out = out - aux.sum() 
            return out 


        