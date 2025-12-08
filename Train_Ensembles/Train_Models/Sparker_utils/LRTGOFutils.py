import torch, time, math
import numpy as np
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

        d = centers_init.shape[1]
        self.norm_const = (1.0 / ((2*math.pi)**(d/2) * (self.sigma**d)))

    def get_coefficients(self):
        a = self.coefficients
        return a

    def forward(self, x):
        x = x.float()
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        kern = self.norm_const * torch.exp(-0.5 * dist_sq / (self.sigma ** 2))  # (N, m)

        # remove the per-sample DC component so gradients aren't washed out
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
        dtype  = self.ensemble_probs.dtype
        device = self.ensemble_probs.device
        # store priors as float32 (if provided as tensors)
        self.weights_cov  = weights_cov
        self.weights_mean = weights_mean
        self.train_weights = train_weights
        self.weights_mean = self.weights_mean.to(dtype=dtype, device=device)
        self.weights_cov  = self.weights_cov.to(dtype=dtype, device=device)
        
        self.lambda_regularizer = lambda_regularizer
        if (self.weights_mean is not None) and (self.weights_cov is not None):

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
            #L = torch.linalg.cholesky(cov)
            # build MVN using scale_tril so torch doesn't re-factorize
            self.aux_model = torch.distributions.MultivariateNormal(self.weights_mean, scale_tril=L)

        else:
            self.aux_model = None

        # trainable weights (float32)
        self.weights = nn.Parameter(weights_init.clone().reshape((self.n_ensemble)).float(),
                                    requires_grad=train_weights)  # [M,]

        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.train_net = train_net

        if self.train_net:
            '''
            self.coeffs = nn.Parameter(torch.tensor([1, 0.], dtype=torch.float32), requires_grad=True)
            '''
            self.network = GaussianKernelLayer(gaussian_center, gaussian_coeffs, gaussian_sigma, train_centers=train_centers)

    def call(self, x):
        x = x.float()
        ensemble = torch.einsum("ij,jk->ik", self.ensemble_probs, self.weights.unsqueeze(1))  # (N,1)
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
        
        sum_w = torch.sum(self.weights)
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
        #print("denom=", denom)
        total = sum_w + sum_a
        return (total - 1.0)**2 #/ denom
        
    def loglik(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)       
            p = (ensemble[:, 0] + net_out)/(torch.sum(self.weights)+torch.sum(self.network.get_coefficients()))
            #return torch.log(p).sum() + aux.sum()
        else:
            p = self.call(x)/torch.sum(self.weights)
        out = torch.log(p).sum() 
        if self.train_weights:
            out = out + aux.sum()
        return out 

    def loss(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)
            p = ensemble[:, 0] + net_out
            out = -torch.log(p).mean() + self.lambda_regularizer * self.normalization_constraint_term() \
                + self.lambda_net * self.net_constraint_term()
            if self.train_weights:
                out = out - aux.mean()
            return out 
        else:
            p = self.call(x)
            out = -torch.log(p).mean() + self.lambda_regularizer * self.normalization_constraint_term()
            if self.train_weights:
                out = out - aux.mean() 
            return out 


# ------------------ Train Function ------------------
def train_loop(x_data, model, name='model', lr=1e-4, tol=100, epochs=20000,patience=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist, epoch_hist = [], []
    best = float("inf")
    bad = 0

    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x_data)   # runs on device
        loss.backward()
        opt.step()
        # log + early stop checks every patience iters
        if (epoch % patience) == 0:
            cur = float(loss.detach().item())
            print(f"[{name}] epoch {epoch} loss {cur:.6f}", flush=True)
            loss_hist.append(cur)
            epoch_hist.append(epoch)
            #if cur + tol < best:
            #    best = cur
            #    bad = 0
            #else:
            #    bad += 1
            #    if bad >= tol:
            #        print(f"[{name}] early stopping after {tol} at epoch {epoch} best {best:.6f}", flush=True)
            #        break
        # time budget: stop cleanly before SLURM wall time
        #if (time.time() - t0) / 60.0 > max_time_min:
        #    print(f"[{name}] stopping due to time budget at epoch {epoch}", flush=True)
        #    break
    return np.array(epoch_hist, np.int32), np.array(loss_hist, np.float32)
