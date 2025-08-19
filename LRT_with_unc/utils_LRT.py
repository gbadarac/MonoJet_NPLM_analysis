import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch, traceback
from torchmin import minimize
from torch import nn
from torch.autograd import Variable
import math 

class GaussianKernelLayer(nn.Module):
    def __init__(self, centers_init, coefficients_init, sigma):
        """
        centers: torch.Tensor of shape (m, d)  -> Gaussian centers
        sigma: float or torch scalar           -> fixed std deviation
        """
        super().__init__()
        self.centers = nn.Parameter(centers_init.float(), requires_grad=True)  # trainable centers
        self.register_buffer('sigma', torch.tensor(sigma))        # non-trainable sigma
        self.coefficients = nn.Parameter(coefficients_init.float(), requires_grad=True) # trainable coefficients
        d = centers_init.shape[1]
        self.softmax = nn.Softmax()
        self.norm_const = (1.0 / ((2 * math.pi) ** (d / 2) * (self.sigma ** d)))

    def forward(self, x):
        """
        x: shape (N, d) -> batch of N points
        returns: shape (N, m) -> Gaussian kernel activations
        """
        # (N, 1, d) - (1, m, d) -> (N, m, d)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        coeffs = self.softmax(self.coefficients)#/self.coefficients.sum()
        # Gaussian kernel with proper normalization
        return torch.einsum("a, ba -> b", coeffs, self.norm_const * torch.exp(-0.5 * dist_sq / (self.sigma ** 2)))

class TAU(nn.Module):
    """ 
    N= input_shape[0]
    m= number of elements in the ensemble
    
    """
    def __init__(self, input_shape, ensemble_probs, weights_init, weights_cov, weights_mean, 
                 gaussian_center, gaussian_coeffs,
                 lambda_regularizer=1e5, gaussian_sigma=0.1,
                 train_weights=True, train_net=True, 
                 model='TAU',name=None, **kwargs):
        super(TAU, self).__init__()
        self.ensemble_probs = ensemble_probs # [N, m]
        self.x_dim = input_shape[1]
        print("problem dimensionality:", self.x_dim)
        self.n_ensemble= ensemble_probs.shape[1]
        self.weights_cov= weights_cov # [m,m]
        self.weights_mean=weights_mean # [m,]
        self.lambda_regularizer = lambda_regularizer
        self.aux_model = torch.distributions.MultivariateNormal(weights_mean, covariance_matrix=weights_cov)
        self.weights = nn.Parameter(weights_init.reshape((self.n_ensemble)).type(torch.double), 
                               requires_grad=train_weights) # [M, 1]
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.train_net=train_net
        if self.train_net:
            self.coeffs = nn.Parameter(torch.tensor([1, 0.]).float(), requires_grad=True)
            self.network = GaussianKernelLayer(gaussian_center, gaussian_coeffs, gaussian_sigma)
            #self.network = nn.Sequential( nn.Linear(self.x_dim, 1000), nn.Sigmoid(), nn.Linear(1000, 1), nn.Sigmoid()) #flow
            
            
    def call(self, x):
        ensemble = torch.einsum("ij, jk -> ik", model_probs, torch.unsqueeze(self.weights, dim=1))
        #print(ensemble.shape)
        if self.train_net:
            net_out = self.network(x)
            return ensemble, net_out
        else:
            return ensemble

    def get_coeffs(self):
        return self.softmax(self.coeffs)

    def weights_constraint_term(self):
        return (torch.sum(self.weights) - 1.0)**2 
    
    def net_constraint_term(self):
        squared_sum = 0.0
        for param in self.network.parameters():
            if param.requires_grad:
                squared_sum += torch.sum(param ** 2)
        return squared_sum
    
    def log_auxiliary_term(self):
        return self.aux_model.log_prob(self.weights) 

    def loglik(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)
            coeffs = self.softmax(self.coeffs)

            #p = 0.5*(1+coeffs[0])*ensemble[:, 0] + 0.5*coeffs[1]*net_out
            p = coeffs[0]*ensemble[:, 0] + coeffs[1]*net_out
            return torch.log(p).sum() + aux.sum()
        else:
            p = self.call(x)
            return torch.log(p).sum() + aux.sum()
            
    def loss(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)
            coeffs = self.softmax(self.coeffs)

            #p = 0.5*(1+coeffs[0])*ensemble[:, 0] + 0.5*coeffs[1]*net_out 
            p = self.relu(coeffs[0]*ensemble[:, 0] + coeffs[1]*net_out )
            
            return -torch.log(p).sum() - aux.sum() + self.lambda_regularizer*self.weights_constraint_term()#+self.lambda_regularizer*torch.sum(self.relu(-p)*torch.exp(-p))
        else:
            p = self.relu(self.call(x))
            return -torch.log(p).sum() - aux.sum() + self.lambda_regularizer*self.weights_constraint_term()#+self.lambda_regularizer*torch.sum(self.relu(-p)*torch.exp(-p))

        