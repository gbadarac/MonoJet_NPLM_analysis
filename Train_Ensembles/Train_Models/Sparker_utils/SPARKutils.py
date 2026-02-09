import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
import math 

def pairwise_dist(X, P):
    X2 = (X ** 2).sum(dim=1, keepdim=True) # (n x 1)                                                            
    P2 = (P ** 2).sum(dim=1, keepdim=True) # (n' x 1)                                                            
    XP = X @ P.T # (n x n')                                                                                      
    return X2 + P2.T - 2 * XP # (n x n')  

def Annealing_Linear(t, ini, fin, t_fin):
    if t < t_fin:
        return ini + (fin - ini) * (t + 1) / t_fin
    else:
        return fin

def Annealing(t, ini, fin, t_fin):
    if t<t_fin: return ini*((fin/ini)**((t+1)/t_fin))
    else: return fin

class Hierarchical(nn.Module):
    """                              
    Mixture of experts: sum of models (default=KernelMethod_SoftMax_2) with different sigma. 
    Trained sequentially in decreasing order of sigma, with annealing.     
    return                            
    - out: sum_j(KernelMethod_SoftMax_2_j) with sigma_j in sigma_list 
    """
    def __init__(self, input_shape, centroids_list, widths_list, coeffs_list,
                 resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False, probability_coeffs=False, model='Soft-SparKer2',
                 name=None, **kwargs):
        super(Hierarchical, self).__init__()

        self.n_layers = len(widths_list)
        self.epsilon=1e-10
        if positive_coeffs:
            self.cmin=0
            self.cmax=coeffs_clip
        else:
            self.cmin=-coeffs_clip
            self.cmax=coeffs_clip
        if model=='Soft-SparKer2':
            self.layers = nn.ModuleList([KernelMethod_SoftMax_2(input_shape, centroids_list[i], widths_list[i], coeffs_list[i],
                                                            resolution_const, resolution_scale, coeffs_clip,
                                                            train_centroids=train_centroids,
                                                            train_widths=train_widths,
                                                            train_coeffs=train_coeffs,
                                                            positive_coeffs=positive_coeffs,
                                                            probability_coeffs=False
                                                               )
                                         for i in range(self.n_layers)])
        elif model=='Soft-SparKer':
            self.layers = nn.ModuleList([KernelMethod_SoftMax(input_shape, centroids_list[i], widths_list[i], coeffs_list[i],
                                                            resolution_const, resolution_scale, coeffs_clip,
                                                            train_centroids=train_centroids,
                                                            train_widths=train_widths,
                                                            train_coeffs=train_coeffs,
                                                            positive_coeffs=positive_coeffs,
                                                            probability_coeffs=False
                                                               )
                                         for i in range(self.n_layers)])
        elif model=='SparKer':
            self.layers = nn.ModuleList([KernelMethod(input_shape, centroids_list[i], widths_list[i], coeffs_list[i],
                                                            resolution_const, resolution_scale, coeffs_clip,
                                                            train_centroids=train_centroids,
                                                            train_widths=train_widths,
                                                            train_coeffs=train_coeffs,
                                                            positive_coeffs=positive_coeffs,
                                                            probability_coeffs=False
                                                               )
                                         for i in range(self.n_layers)])
        self.probability_coeffs=probability_coeffs
    
    def call(self, x):
        out = [self.layers[i].call(x) for i in range(self.n_layers)]
        out = torch.stack(out, axis=0) # [n_layers, n, 1]    
        out = torch.cumsum(out, axis=0) # [n_layers, n, 1]   
        if self.probability_coeffs:
            # Create divisor [1, 2, 3, ..., n_layers]
            divisors = self.get_norm()#torch.arange(1, self.n_layers + 1, device=out.device).view(-1, 1, 1)
            out = out / divisors
        return out
        
    def call_cumsum_j(self, x, j):
        out = [self.layers[i].call(x) for i in range(j+1)]
        out = torch.stack(out, axis=0) # [j, n, 1] 
        out = torch.sum(out, axis=0) # [n]
        if self.probability_coeffs:
            divisors = self.get_norm()[j]
            out = out / divisors
        return out# [n, 1]
        
    def call_j(self, x, j):
        return self.layers[j].call(x) # [n, 1]                                                            

    def get_norm(self):
        coeffs = torch.stack([self.layers[i].get_coeffs().sum() for i in range(self.n_layers)]) # [n_layers,]
        cumnorm = torch.abs(torch.cumsum(coeffs, axis=0)+1e-10) # [n_layers,]
        return cumnorm
        
    def get_widths(self):
        out = [self.layers[i].get_widths() for i in range(self.n_layers)]
        return torch.cat(out, dim=0)

    def get_coeffs(self):
        out = [self.layers[i].get_coeffs() for i in range(self.n_layers)]
        return torch.cat(out, dim=0)

    def get_centroids(self):
        out = [self.layers[i].get_centroids() for i in range(self.n_layers)]
        return torch.cat(out, dim=0)

    def get_widths_j(self, j):
        return self.layers[j].get_widths()

    def get_coeffs_j(self, j):
        return self.layers[j].get_coeffs()

    def get_centroids_j(self, j):
        return self.layers[j].get_centroids()

    def set_width_j(self, width, j):
        self.layers[j].set_width(width)
        return

    def set_coeffs_grad(self, require_grad, j):
        self.get_coeffs_j(j=j).requires_grad = require_grad

    def set_width_grad(self, require_grad, j):
        self.get_widths_j(j=j).requires_grad = require_grad

    def set_centroids_grad(self, require_grad, j):
        self.get_centroids_j(j=j).requires_grad = require_grad

    def clip_coeffs(self):
        for j in range(self.n_layers):
            self.layers[j].clip_coeffs()

class KernelMethod(nn.Module):
    '''    
    return: coeff * exp(-0.5(x-mu)**2/scale**2)  
    '''
    def __init__(self, input_shape, centroids, widths, coeffs, resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False, probability_coeffs=False,
                 name=None, **kwargs):
        super(KernelMethod, self).__init__()
        self.positive_coeffs=positive_coeffs
        if self.positive_coeffs:
            self.cmin=0
            self.cmax=coeffs_clip
        else:
            self.cmin=-coeffs_clip
            self.cmax=coeffs_clip
        # keep coeffs dtype consistent with inputs
        if isinstance(coeffs, torch.Tensor):
            dtype = coeffs.dtype
            device = coeffs.device
        else:
            dtype = torch.float64
            device = torch.device("cpu")

        self.coeffs = Variable(coeffs.reshape((-1, 1)).to(device=device, dtype=dtype),
                            requires_grad=train_coeffs)
                                                                               
        self.kernel_layer = KernelLayer(input_shape=input_shape, centroids=centroids, widths=widths,
                                        train_centroids=train_centroids, train_widths=train_widths,
                                        resolution_const=resolution_const, resolution_scale=resolution_scale,
                                        name='kernel_layer')
        self.softmax = torch.softmax
        self.probability_coeffs=probability_coeffs


    def call(self, x):
        K_x, _ = self.kernel_layer.call(x)  # [n, M]

        W_x = self.coeffs  # [M, 1]
        if self.probability_coeffs == True and W_x.sum():
            W_x = self.softmax(W_x, 0)

        out = torch.tensordot(K_x, W_x, dims=([1], [0]))
        return out

    def get_centroids_entropy(self):
        return self.kernel_layer.get_centroids_entropy()

    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_widths(self):
        return self.kernel_layer.get_widths()

    def set_width(self, width):
        self.kernel_layer.set_width(width)
        return

    def get_widths_tilde(self):
        return self.kernel_layer.get_widths_tilde()

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()
        return

    def clip_coeffs(self):
        self.coeffs.data = self.coeffs.data.clamp(self.cmin,self.cmax)

class KernelMethod_SoftMax_2(nn.Module):
    '''                              
    return exp(-0.5(x-mu_i)**2/scale**2) * exp( -0.5(x-mu_i)**2/scale**2 )/sum_j[exp( -0.5(x-mu_j)**2/scale**2 )] 
    '''
    def __init__(self, input_shape, centroids, widths, coeffs, resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False, probability_coeffs=False,
                 name=None, **kwargs):
        super(KernelMethod_SoftMax_2, self).__init__()
        self.train_coeffs=train_coeffs
        self.coeffs=coeffs
        self.epsilon=1e-10
        if positive_coeffs:
            self.cmin=0
            self.cmax=coeffs_clip
        else:
            self.cmin=-coeffs_clip
            self.cmax=coeffs_clip
        self.coeffs = Variable(self.coeffs.reshape((-1, 1)).type(torch.double),
                               requires_grad=train_coeffs) # [M, 1]                                                                                  
        self.kernel_layer = KernelLayer(input_shape=input_shape, centroids=centroids, widths=widths,
                                        train_centroids=train_centroids, train_widths=train_widths,
                                        resolution_const=resolution_const, resolution_scale=resolution_scale,
                                        name='kernel_layer')
        self.softmax = torch.softmax
        self.probability_coeffs=probability_coeffs

    def call(self, x):
        K_x, _ = self.kernel_layer.call(x)  # [n, M]

        Z = torch.sum(K_x, dim=1, keepdim=True) + self.epsilon  # [n, 1]

        # ---- local copy of coeffs with correct dtype/device (DO NOT overwrite self.coeffs) ----
        W_x = self.coeffs
        if W_x.dtype != K_x.dtype or W_x.device != K_x.device:
            W_x = W_x.to(dtype=K_x.dtype, device=K_x.device)
        # ---------------------------------------------------------------------------------------

        if self.probability_coeffs:
            W_x = self.softmax(W_x, 0)

        out = torch.tensordot(K_x * K_x, W_x, dims=([1], [0]))  # [n, 1]
        out = out / Z  # [n, 1]
        return out

    def get_softmax(self, x):
        K_x, _ = self.kernel_layer.call(x) # [n, M]    
        Z = torch.sum(K_x, dim=1, keepdim=True) +self.epsilon # [n]  
        return K_x/Z # [n, M] 
        
    def get_kernel(self, x):
        K_x, _ = self.kernel_layer.call(x) # [n, M]  
        return K_x

    
    def clip_coeffs(self):
        self.coeffs.data = self.coeffs.data.clamp(self.cmin,self.cmax)
        return

    def get_centroids_entropy(self):
        return self.kernel_layer.get_centroids_entropy()

    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_widths(self):
        return self.kernel_layer.get_widths()

    def get_widths_tilde(self):
        return self.kernel_layer.get_widths_tilde()

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()

    def set_widths(self, widths):
        self.kernel_layer.set_widths(widths)
        return

    def set_width(self, width):
        self.kernel_layer.set_width(width)
        return

class KernelLayer(nn.Module):
    def __init__(self, centroids, widths, resolution_const=0, resolution_scale=1,
                 beta=None,
                 cmin=None, cmax=None, train_centroids=True, train_widths=False,
                 name=None, **kwargs):
        super(KernelLayer, self).__init__()

        self.resolution_const = resolution_const
        self.resolution_scale = resolution_scale
        self.cmin = cmin
        self.cmax = cmax
        self.beta = beta

        self.M = centroids.shape[0]
        self.d = centroids.shape[1]

        # Decide device from centroids if they are a tensor, otherwise CPU
        if isinstance(centroids, torch.Tensor):
            device = centroids.device
        else:
            device = torch.device("cpu")

        # keep dtype consistent with inputs
        if isinstance(centroids, torch.Tensor):
            dtype = centroids.dtype
        elif isinstance(widths, torch.Tensor):
            dtype = widths.dtype
        else:
            dtype = torch.float64  # safe default

        centroids = torch.as_tensor(centroids, dtype=dtype, device=device)
        widths    = torch.as_tensor(widths,    dtype=dtype, device=device)

        # width is a tensor (scalar or [d]-vector) on the right device
        self.width = widths[0]              # shape [d] or scalar
        self.centroids = Variable(
            centroids,
            requires_grad=train_centroids
        )                                   # [M, d]

        self.cov_diag = self.width ** 2

    def call(self, x):
        if self.beta==None:
            out, arg = self.Kernel(x)
            return out, arg
        else:
            out, arg, out2, arg2 = self.Kernel(x)
            return out, arg, out2, arg

    def transform_widths(self):
        # transform width variable to account for resolution boundaries (quadrature sum)        
        widths = torch.add(self.widths**2, self.resolution_const**2) # [M, d]  
        widths+= torch.multiply(self.centroids, self.resolution_scale)**2 # [M, d]
        widths = torch.sqrt(widths) # [M, d] 
        return widths

    def get_widths(self):
        return self.width * torch.ones(
            (self.M, self.d),
            device=self.width.device,
            dtype=self.width.dtype,
        )


    def set_widths(self, widths):
        self.widths.data = widths
        self.compute_cov_diag()
        return

    def set_width(self, width):
        """
        Ensure self.width is always a torch.Tensor on the same
        device/dtype as the existing width.
        """
        # If width is not a tensor, wrap it
        if not isinstance(width, torch.Tensor):
            width = torch.tensor(
                width,
                device=self.width.device if isinstance(self.width, torch.Tensor) else self.centroids.device,
                dtype=self.width.dtype if isinstance(self.width, torch.Tensor) else self.centroids.dtype,
            )

        self.width = width
        self.compute_cov_diag()
        return

    def get_centroids(self):
        return self.centroids #[M, d]

    def clip_centroids(self):
        if (not self.cmin==None) and (not self.cmax==None):
            self.centroids.data = self.centroids.data.clamp(self.cmin,self.cmax)
        return

    def get_centroids_entropy(self):
        """  
        sum_j(sum_i(K_i(mu_j))*log(sum_i(K_i(mu_j))))  
        return: scalar 
        """
        K_mu, _ = self.call(self.centroids) #[M, M]    
        K_mu = torch.mean(K_mu, axis=1) # [M,]  
        entropy = torch.sum(torch.multiply(K_mu, torch.log(K_mu)))
        return entropy

    def compute_cov_diag(self):
        self.cov_diag = self.width**2
        return

    def gauss_const(self):
        # self.width is a tensor on the correct device
        det_sigma_sqrt = self.width.pow(self.d)

        two_pi = torch.tensor(
            2 * math.pi,
            device=self.width.device,
            dtype=self.width.dtype,
        )

        const = 1.0 / (det_sigma_sqrt * two_pi.pow(0.5 * self.d))
        const = torch.unique(const)  # optional, to mimic old behavior
        return const

    def Kernel(self,x):
        """  
        # x.shape = [N, d] 
        # widths.shape = [M, d] 
        # centroids.shape = [M, d] 
        Returns exp(-0.5*(x-mu)^2/scale^2)
        # return.shape = [N,M]   
        """
        dist_sq  = torch.subtract(x[:, None, :], self.centroids[None, :, :])**2 # [N, M, d]  
        arg = -0.5*torch.sum(dist_sq/self.cov_diag,axis=2) # [N, M]                                                      
        kernel = self.gauss_const()*torch.exp(arg)
        if self.beta!=None:
            arg2 = -1*self.beta*torch.sum(dist_sq, axis=2) #[N, M] 
            kernel2 = torch.exp(arg2)
            return kernel, arg, kernel2, arg2
        else:
            return kernel, arg # [N, M] 

