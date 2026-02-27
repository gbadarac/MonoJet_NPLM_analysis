import torch, time, math, os 
import numpy as np
from torch import nn
from scipy.stats import chi2 as chi2_scipy


class GaussianKernelLayer(nn.Module):
    def __init__(self, centers_init, coefficients_init, sigma, train_centers=False, clip_coeffs=None):
        super().__init__()
        self.centers = nn.Parameter(centers_init.double(), requires_grad=train_centers)
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.coefficients = nn.Parameter(coefficients_init.double(), requires_grad=True)

        # --- auto mode detection from init ---
        with torch.no_grad():
            s = coefficients_init.double().sum()
            L1 = coefficients_init.double().abs().sum()

        d = centers_init.shape[1]
        self.norm_const = (1.0 / ((2*math.pi)**(d/2) * (self.sigma**d)))
        self.softmax = nn.Softmax(dim=0)
        self.clip = clip_coeffs
        self.train_centers=train_centers
        
    def get_coefficients(self):
        a = self.coefficients- self.coefficients.mean()
        return a
    
    def get_kernels(self, x):
        x = x.double()
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        kern = self.norm_const * torch.exp(-0.5 * dist_sq / (self.sigma ** 2))  # (N, m)
        return kern
    
    def clip_coefficients(self):
        self.coefficients.data = self.coefficients.data.clamp(-1*self.clip, self.clip)

    def forward_coeffs_only(self, kern):
        w = self.coefficients - self.coefficients.mean()                             
        return torch.einsum("m,Nm->N", w, kern)
        
    def forward(self, x):
        kern = self.get_kernels(x)
        w = self.coefficients - self.coefficients.mean()
        return torch.einsum("m,Nm->N", w, kern)

class TAU(nn.Module):
    """ 
    N= input_shape[0]
    m= number of elements in the ensemble
    """
    def __init__(self, input_shape, ensemble_probs, ensemble_norm_probs,
                 weights_init, weights_cov, weights_mean, 
                 gaussian_center, gaussian_coeffs,
                 lambda_net=1e-6, gaussian_sigma=0.1,
                 train_centers=False,
                 train_weights=True,
                 train_net=True,
                 clip_net_coeffs=None,
                 model='TAU', name=None, **kwargs):
        super(TAU, self).__init__()
        self.lambda_net = float(lambda_net)  
        self.ensemble_probs = ensemble_probs.double()   # [N, m] -> float32
        self.ensemble_norm_probs = ensemble_norm_probs.double()   # [N, m] -> float32 
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

        # Just verify that:
        assert self.weights_mean.shape[0] == self.n_ensemble, \
            f"weights_mean shape {self.weights_mean.shape} doesn't match n_ensemble {self.n_ensemble}"
        assert self.weights_cov.shape == (self.n_ensemble, self.n_ensemble), \
            f"weights_cov shape {self.weights_cov.shape} doesn't match expected ({self.n_ensemble}, {self.n_ensemble})"

        if (self.weights_mean is not None) and (self.weights_cov is not None):
            print(f"weights_mean shape: {self.weights_mean.shape}")
            print(f"weights_cov shape: {self.weights_cov.shape}")
            print(f"n_ensemble: {self.n_ensemble}")
            # symmetrize
            cov = 0.5 * (self.weights_cov + self.weights_cov.T)
            cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            # try Cholesky with increasing jitter
            eye = torch.eye(cov.shape[0], dtype=dtype, device=device)
            eps = 1e-6* torch.trace(cov) / cov.shape[0]
            jitter = eps* (cov.diagonal().abs().mean() + 1.0)
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
        self.weights = nn.Parameter(weights_init.clone().reshape((self.n_ensemble)).double(),
                                    requires_grad=train_weights)  # [M,]

        self.eps = 1e-10
        self.train_net = train_net
        self.train_centers=train_centers
        if self.train_net:
            self.network = GaussianKernelLayer(gaussian_center, gaussian_coeffs, gaussian_sigma,
                                               train_centers=train_centers, clip_coeffs=clip_net_coeffs)

    def call(self, x):
        x = x.double()
        w = self.weights.unsqueeze(1)
        w_norm = 1 - torch.sum(w, dim=0, keepdim=True)

        ensemble = torch.einsum("ij,jk->ik", self.ensemble_probs, w)  # (N,1)
        ensemble += torch.einsum("ij,jk->ik", self.ensemble_norm_probs, w_norm)

        if self.train_net and self.train_centers:
            # full mode: x is raw (N,d)
            net_out = self.network(x)
            return ensemble, net_out

        elif self.train_net and (not self.train_centers):
            # cheap mode:
            # x can be either raw (N,d) OR precomputed kernels (N,M)
            if x.dim() == 2 and x.shape[1] == self.x_dim:
                kern = self.network.get_kernels(x)      # (N,M)
            else:
                kern = x                                # assume already (N,M)
            net_out = self.network.forward_coeffs_only(kern)
            return ensemble, net_out

        else:
            return ensemble
        
    #--------------------------------------    
    # DIAGNOSTIC BIT
    #--------------------------------------
    @torch.no_grad()
    def p_raw_stats(self, x_input):
        """
        Check whether the *raw* model density p(x) (before clamp/eps) goes non-positive.

        IMPORTANT: x_input must be the same object you pass into loss():
          - if cheap mode (train_net=True, train_centers=False): x_input is kernels (N,M)
          - else: x_input is raw data (N,d)
        Returns: (p_min, n_le0, n_nonfinite)
        """
        if self.train_net:
            ens, net_out = self.call(x_input)     # ens: (N,1), net_out: (N,)
            p_raw = ens[:, 0] + net_out
        else:
            ens = self.call(x_input)              # ens: (N,1)
            p_raw = ens[:, 0]

        p_min = float(p_raw.min().item())
        n_le0 = int((p_raw <= 0).sum().item())
        n_nonfinite = int((~torch.isfinite(p_raw)).sum().item())
        return p_min, n_le0, n_nonfinite

    #--------------------------------------
    
    def get_coeffs(self):
        return self.weights.unsqueeze(1)

    def net_coeffs_L2(self):
        return torch.sum(self.network.get_coefficients()**2)
                    
    def log_auxiliary_term(self):
        return self.aux_model.log_prob(self.weights) 
        
    def loglik(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net and (not self.train_centers):
            print('Cheap mode: ON')
            x_input = self.network.get_kernels(x) #[N, M]                                                                      
        else:
            x_input = x #[N, d]
        if self.train_net:
            ensemble, net_out = self.call(x_input)       
            p = (ensemble[:, 0] + net_out)
        else:
            p = self.call(x_input)
        p = torch.clamp(p, min=self.eps)
        out = torch.log(p).sum() 
        if self.train_weights:
            out = out + aux.sum()
        return out
    
    def monitor(self, x):
        if self.train_net:
            ensemble, net_out = self.call(x)
            print(torch.min(ensemble[:, 0]), torch.max(ensemble[:, 0]),
                  torch.min(net_out), torch.max(net_out))
        else:
            ensemble = self.call(x)
            print(torch.min(ensemble[:, 0]), torch.max(ensemble[:, 0]))
        return 

    def loss(self, x):
        aux = self.log_auxiliary_term()
        if self.train_net:
            ensemble, net_out = self.call(x)
            p = ensemble[:, 0] + net_out + self.eps
            out = -torch.log(p).sum()
            out+= self.lambda_net * self.net_coeffs_L2()
            if self.train_weights:
                out = out - aux.sum()
            return out 
        else:
            p = self.call(x) +self.eps
            out = -torch.log(p).sum()
            if self.train_weights:
                out = out - aux.sum()
            return out 
'''
def train_loop(x_data, model, name='model', lr=1e-4, epochs=20000, patience=1000, monitor=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist, epoch_hist = [], []
    best = float("inf")
    bad = 0
    if model.train_net and (not model.train_centers):
        print('Cheap mode: ON')
        x_input = model.network.get_kernels(x_data) #[N, M]
    else:
        x_input = x_data #[N, d]
        
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x_input)   # runs on device
        loss.backward()
        opt.step()
                
        if model.train_net:
            if model.network.clip!=None:
                model.network.clip_coefficients()
    
        if (epoch % patience) == 0:
            cur = float(loss.detach().item())
            print(f"[{name}] epoch {epoch} loss {cur:.6f}", flush=True)
            if monitor:
                with torch.no_grad():
                    model.monitor(x_data)
            loss_hist.append(cur)
            epoch_hist.append(epoch)
        
    return np.array(epoch_hist, np.int32), np.array(loss_hist, np.float32)
'''
def train_loop(
    x_data,
    model,
    name='model',
    lr=1e-4,
    epochs=20000,
    patience=1000,
    monitor=False,
    save_every=1000,   # NEW: save cadence in epochs (e.g. 1000)
    out_dir=None,      # NEW: where to write .npy files
    seed=None          # NEW: for filenames
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist, epoch_hist = [], []

    # Keep your "cheap mode" logic
    if model.train_net and (not model.train_centers):
        print('Cheap mode: ON')
        x_input = model.network.get_kernels(x_data)  # [N, M]
    else:
        x_input = x_data  # [N, d]

    # NEW: history buffers
    w_hist = []
    wsum_hist = []
    ep_hist = []
    wnorm_hist = []
    
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x_input)   # runs on device
        loss.backward()
        opt.step()

        # Keep your coeff clipping
        if model.train_net:
            if model.network.clip is not None:
                model.network.clip_coefficients()

        # NEW: record weights every save_every epochs
        if (save_every is not None) and (out_dir is not None) and (epoch % save_every == 0):
            with torch.no_grad():
                w = model.weights.detach().cpu().numpy().copy()  # (n_ensemble,)
                s = float(w.sum())
                w_hist.append(w)
                wsum_hist.append(s)
                wnorm_hist.append(1.0 - s)
                ep_hist.append(epoch)
            
        # Keep your loss print logic
        if (epoch % patience) == 0:
            cur = float(loss.detach().item())

            # optional extra monitor
            if monitor:
                with torch.no_grad():
                    model.monitor(x_data)

            # loss history
            loss_hist.append(cur)
            epoch_hist.append(epoch)

            with torch.no_grad():
                pmin, nle0, nnan = model.p_raw_stats(x_input)

            print(
                f"[{name}] ep {epoch:6d}  loss {cur: .3e}  "
                f"pmin {pmin: .2e}  n<=0 {nle0:6d}  nan {nnan:6d}  ",
                flush=True
            )

    # NEW: dump the histories once at the end (cheap + safe)
    if (save_every is not None) and (out_dir is not None) and (len(ep_hist) > 0):
        tag = f"seed{seed}_" if seed is not None else ""
        np.save(os.path.join(out_dir, f"{tag}{name}_weights_hist.npy"),
                np.stack(w_hist, axis=0))  # (n_saves, n_ensemble)
        np.save(os.path.join(out_dir, f"{tag}{name}_weights_sum_hist.npy"),
                np.array(wsum_hist, dtype=np.float64))  # (n_saves,)
        np.save(os.path.join(out_dir, f"{tag}{name}_weights_epoch_hist.npy"),
                np.array(ep_hist, dtype=np.int32))  # (n_saves,)
        np.save(os.path.join(out_dir, f"{tag}{name}_weights_wnorm_hist.npy"),
                np.array(wnorm_hist, dtype=np.float64))  # (n_saves,

    return np.array(epoch_hist, np.int32), np.array(loss_hist, np.float32)
