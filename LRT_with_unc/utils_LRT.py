import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch, traceback
from torchmin import minimize
from torch import nn
from torch.autograd import Variable
import math 

class GaussianKernelLayer(nn.Module):
    def __init__(self, centers_init, coefficients_init, sigma, train_centers=False):
        """
        centers: torch.Tensor of shape (m, d)  -> Gaussian centers
        sigma: float or torch scalar           -> fixed std deviation
        """
        super().__init__()
        self.centers = nn.Parameter(centers_init.float(), requires_grad=train_centers)   # trainable centers
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32)) # non-trainable sigma (float32)
        self.coefficients = nn.Parameter(coefficients_init.float(), requires_grad=True) # trainable coefficients
        d = centers_init.shape[1]
        self.softmax = nn.Softmax(dim=0)
        self.norm_const = (1.0 / ((2 * math.pi) ** (d / 2) * (self.sigma ** d)))

    def forward(self, x):
        """
        x: shape (N, d) -> batch of N points
        returns: shape (N, m) -> Gaussian kernel activations
        """
        x = x.float()
        # (N, 1, d) - (1, m, d) -> (N, m, d)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        coeffs = self.softmax(self.coefficients)  # / self.coefficients.sum()
        # Gaussian kernel with proper normalization
        return torch.einsum("a, ba -> b", coeffs, self.norm_const * torch.exp(-0.5 * dist_sq / (self.sigma ** 2)))

class TAU(nn.Module):
    """ 
    N= input_shape[0]
    m= number of elements in the ensemble
    """
    def __init__(self, input_shape, ensemble_probs, weights_init, weights_cov, weights_mean, 
                 gaussian_center, gaussian_coeffs,
                 lambda_regularizer=1e5, gaussian_sigma=0.1, train_centers=False,
                 train_weights=True, train_net=True, 
                 model='TAU', name=None, **kwargs):
        super(TAU, self).__init__()
        self.ensemble_probs = ensemble_probs.float()   # [N, m] -> float32
        self.x_dim = input_shape[1]
        print("problem dimensionality:", self.x_dim)
        self.n_ensemble = self.ensemble_probs.shape[1]

        # store priors as float32 (if provided as tensors)
        self.weights_cov  = weights_cov.float()  if isinstance(weights_cov,  torch.Tensor) else weights_cov
        self.weights_mean = weights_mean.float() if isinstance(weights_mean, torch.Tensor) else weights_mean

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
            self.coeffs = nn.Parameter(torch.tensor([1, 0.], dtype=torch.float32), requires_grad=True)
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

    def weights_constraint_term(self):
        return (torch.sum(self.weights) - 1.0) ** 2 
    
    def net_constraint_term(self):
        squared_sum = 0.0
        for param in self.network.parameters():
            if param.requires_grad:
                squared_sum += torch.sum(param ** 2)
        return squared_sum
    
    def log_auxiliary_term(self):
        if self.aux_model is None:
            return torch.tensor(0.0, dtype=self.weights.dtype, device=self.weights.device)
        return self.aux_model.log_prob(self.weights) 

    def loglik(self, x):
        x = x.float()
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)
            coeffs = self.softmax(self.coeffs)
            # p = 0.5*(1+coeffs[0])*ensemble[:, 0] + 0.5*coeffs[1]*net_out
            p = coeffs[0] * ensemble[:, 0] + coeffs[1] * net_out
            return torch.log(p).sum() + aux.sum()
        else:
            p = self.call(x)
            return torch.log(p).sum() + aux.sum()
            
    def loss(self, x):
        x = x.float()
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)
            coeffs = self.softmax(self.coeffs)
            # p = 0.5*(1+coeffs[0])*ensemble[:, 0] + 0.5*coeffs[1]*net_out 
            p = self.relu(coeffs[0] * ensemble[:, 0] + coeffs[1] * net_out )
            lambda_krl = 10  # tune in [1e-4, 1e-2]
            k = self.network.softmax(self.network.coefficients)      # softmaxed kernel weights
            pen_krl = lambda_krl * k.pow(2).mean()
            return -torch.log(p).sum() - aux.sum() + self.lambda_regularizer * self.weights_constraint_term()+ pen_krl
        else:
            p = self.relu(self.call(x))
            return -torch.log(p).sum() - aux.sum() + self.lambda_regularizer * self.weights_constraint_term()