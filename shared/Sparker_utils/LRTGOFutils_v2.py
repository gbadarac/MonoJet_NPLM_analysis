import torch, time, math, os 
import numpy as np
from torch import nn

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
        if self.weights_mean is not None:
            self.weights_mean = self.weights_mean.to(dtype=dtype, device=device)
        if self.weights_cov is not None:
            self.weights_cov  = self.weights_cov.to(dtype=dtype, device=device)

        # Verify shapes only when prior is provided:
        if self.weights_mean is not None:
            assert self.weights_mean.shape[0] == self.n_ensemble, \
                f"weights_mean shape {self.weights_mean.shape} doesn't match n_ensemble {self.n_ensemble}"
        if self.weights_cov is not None:
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
        if self.aux_model is None:
            return torch.zeros(1, dtype=self.ensemble_probs.dtype, device=self.ensemble_probs.device)
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
            p_raw = ensemble[:, 0] + net_out
            p = torch.clamp(p_raw, min=self.eps)
            out = -torch.log(p).sum()
            out += self.lambda_net * self.net_coeffs_L2()
            if self.train_weights:
                out = out - aux.sum()
        else:
            p_raw = self.call(x).squeeze(-1)
            p = torch.clamp(p_raw, min=self.eps)
            out = -torch.log(p).sum()
            if self.train_weights:
                out = out - aux.sum()
        if not torch.isfinite(out):
            raise RuntimeError(
                f"loss() is not finite ({float(out):.3e}) even after density clamping. "
                f"p_raw: min={float(p_raw.min()):.3e}, max={float(p_raw.max()):.3e}. "
                f"Check aux term or L2 regularisation."
            )
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
    save_param_history=False,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist, epoch_hist = [], []
    weight_hist, coeff_hist = [], []

    if model.train_net and (not model.train_centers):
        print('Cheap mode: ON')
        x_input = model.network.get_kernels(x_data)  # [N, M]
    else:
        x_input = x_data  # [N, d]

    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x_input)
        loss.backward()
        opt.step()

        if model.train_net:
            if model.network.clip is not None:
                model.network.clip_coefficients()

        if (epoch % patience) == 0:
            cur = float(loss.detach().item())

            if monitor:
                with torch.no_grad():
                    model.monitor(x_data)

            loss_hist.append(cur)
            epoch_hist.append(epoch)

            with torch.no_grad():
                pmin, nle0, nnan = model.p_raw_stats(x_input)

                if save_param_history:
                    weight_hist.append(model.weights.cpu().numpy().copy())
                    if model.train_net:
                        coeff_hist.append(model.network.coefficients.cpu().numpy().copy())

            print(
                f"[{name}] ep {epoch:6d}  loss {cur: .3e}  "
                f"pmin {pmin: .2e}  n<=0 {nle0:6d}  nan {nnan:6d}  ",
                flush=True
            )

    param_hist = None
    if save_param_history:
        param_hist = {'weights': np.array(weight_hist)}   # (n_checkpoints, n_ensemble)
        if coeff_hist:
            param_hist['coeffs'] = np.array(coeff_hist)   # (n_checkpoints, M)

    return np.array(epoch_hist, np.int32), np.array(loss_hist, np.float32), param_hist


# ======================================================================
# Exact profiling fit — analogue of wifi_better_basis/classifier_gof.py's
# fit_classifier, adapted to the density-space LRT.
#
#   minimize over (w, c):
#     L(w, c) = - sum_i log f_i
#               + 0.5 (w - w_mean)^T (Sigma_w + ridge)^{-1} (w - w_mean)   [if Sigma_w]
#               + lam_pert * || c - mean(c) ||^2                           [if Kmat]
#     f_i = a_i + Phi_i . w + sum_j (c_j - mean(c)) K_ij ,   |c_j| <= clip
#
# f is linear in (w, c), so -sum log f is convex on the feasible set
# {f > 0}; with the convex penalties and box the optimum is unique.
# Solved by active-set projected damped Newton with closed-form gradient
# and Hessian. The line search only accepts steps with f > 0 at every
# data point — the log is its own barrier, there is NO density clamp.
# This replaces the Adam + eps-clamp path, whose clamp severed the
# barrier and enabled the signed-weight runaway.
#
# Conventions match TAU exactly: the last ensemble column is the derived
# weight 1 - sum(w) (callers pass Phi = probs[:, :-1] - probs[:, -1:] and
# a = probs[:, -1]); kernel coefficients are mean-centred in the density
# and in the L2 penalty (lam_pert * sum(c_eff^2), no 1/2 factor); the box
# applies to the raw coefficients.
# ======================================================================

GRAD_NORM_OK = 1e-4   # same "well-converged" threshold as classifier_gof.py


def gaussian_kernel_matrix(x, centers, sigma, chunk=20000):
    """K[i, j] = normalized isotropic Gaussian kernel j at x_i, shape (N, M)."""
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    d = x.shape[1]
    norm_const = (2.0 * np.pi) ** (-d / 2.0) * float(sigma) ** (-d)
    out = np.empty((x.shape[0], centers.shape[0]), dtype=np.float64)
    for i0 in range(0, x.shape[0], chunk):
        diff = x[i0:i0 + chunk, None, :] - centers[None, :, :]
        out[i0:i0 + chunk] = norm_const * np.exp(
            -0.5 * (diff ** 2).sum(axis=2) / float(sigma) ** 2)
    return out


def fit_lrt_exact(Phi, a, w_init, Kmat=None, clip=None, lam_pert=0.0,
                  Sigma_w=None, w_mean=None, frozen_weights=False,
                  max_iter=500, gtol=1e-6, dec_tol=1e-7, ridge_rel=1e-8,
                  verbose=False, name="fit"):
    """
    Returns dict:
        loss         : final penalized objective
        loglik       : sum_i log f_i at the optimum (no penalties)
        aux          : -0.5 (w-w_mean)^T Sinv (w-w_mean) (0 if no prior);
                       MVN normalization constants cancel in T differences
        f            : (N,) fitted density at the data points (all > 0)
        w, c, c_eff  : final parameters (c raw, c_eff mean-centred; None if absent)
        grad_norm    : projected-gradient norm at the final point
        n_iter, converged, hit_max_iter, fmin, n_at_clip, loss_hist
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    a = np.ascontiguousarray(a, dtype=np.float64)
    N, K1 = Phi.shape
    train_w = not frozen_weights
    has_pert = Kmat is not None

    w = np.array(w_init, dtype=np.float64).ravel().copy()
    if has_pert:
        Kt = np.ascontiguousarray(Kmat, dtype=np.float64)
        Kt = Kt - Kt.mean(axis=1, keepdims=True)   # mean-centring: zero-mass perturbation
        M = Kt.shape[1]
        c = np.zeros(M, dtype=np.float64)
        box = np.inf if clip is None else float(clip)
        # tiny ridge along the constant direction (f and the penalty are
        # invariant under c -> c + const, which would make H singular)
        mu = 1e-10 * max(1.0, lam_pert)
        C_pen = np.eye(M) - np.ones((M, M)) / M
    else:
        M = 0
        c = None

    use_prior = (Sigma_w is not None) and train_w
    if use_prior:
        S = np.asarray(Sigma_w, dtype=np.float64)
        eps = ridge_rel * (np.trace(S) / S.shape[0])
        Sinv = np.linalg.inv(S + eps * np.eye(S.shape[0]))
        wm = np.asarray(w_mean, dtype=np.float64).ravel()

    def density(w_, c_):
        f = a + Phi @ w_
        if has_pert:
            f = f + Kt @ c_
        return f

    def objective(w_, c_):
        f = density(w_, c_)
        if f.min() <= 0.0:
            return np.inf, f
        L = -np.log(f).sum()
        if use_prior:
            dw = w_ - wm
            L += 0.5 * float(dw @ Sinv @ dw)
        if has_pert and lam_pert > 0:
            ce = c_ - c_.mean()
            L += lam_pert * float(ce @ ce)
        return L, f

    def result(L, f, grad_norm, n_iter, converged, hit_max, loss_hist):
        aux = 0.0
        if use_prior:
            dw = w - wm
            aux = -0.5 * float(dw @ Sinv @ dw)
        out = {
            "loss": float(L),
            "loglik": float(np.log(f).sum()),
            "aux": aux,
            "f": f.copy(),
            "w": w.copy(),
            "c": (c.copy() if has_pert else None),
            "c_eff": ((c - c.mean()).copy() if has_pert else None),
            "grad_norm": float(grad_norm),
            "newton_dec": float(newton_dec),
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "hit_max_iter": bool(hit_max),
            "fmin": float(f.min()),
            "n_at_clip": (int(np.sum(np.abs(c) >= box * 0.999)) if has_pert else 0),
            "loss_hist": np.array(loss_hist, dtype=np.float64),
        }
        return out

    L, f = objective(w, c)
    if not np.isfinite(L):
        raise RuntimeError(
            f"[{name}] infeasible start: min f = {density(w, c).min():.3e} <= 0")

    newton_dec = 0.0

    # Trivial case (classifier_gof parity): frozen weights, no perturbation.
    if frozen_weights and not has_pert:
        return result(L, f, 0.0, 0, True, False, [L])

    P = (K1 if train_w else 0) + M
    loss_hist = [L]
    grad_norm = np.inf
    newton_dec = np.inf
    hit_max = True
    lam_lm = 0.0      # Levenberg damping — adapted; handles the near-singular
                      # Hessian from strongly-overlapping (near-collinear) components
    stall = 0

    for it in range(1, max_iter + 1):
        r = 1.0 / f

        # --- gradient (closed form) ---
        g = np.empty(P)
        g_c = None
        if train_w:
            g_w = -(Phi.T @ r)
            if use_prior:
                g_w = g_w + Sinv @ (w - wm)
            g[:K1] = g_w
        if has_pert:
            g_c = -(Kt.T @ r) + 2.0 * lam_pert * (c - c.mean())
            g[P - M:] = g_c

        # --- active set on the box, projected gradient ---
        free = np.ones(P, dtype=bool)
        if has_pert:
            at_hi = c >= box * (1 - 1e-14)
            at_lo = c <= -box * (1 - 1e-14)
            blocked = (at_hi & (g_c <= 0)) | (at_lo & (g_c >= 0))
            free[P - M:] = ~blocked
        pg = np.where(free, g, 0.0)
        grad_norm = float(np.linalg.norm(pg))
        if grad_norm <= gtol:
            hit_max = False
            break

        # --- Hessian (closed form) ---
        H = np.zeros((P, P))
        R = Phi * r[:, None]
        if train_w:
            H[:K1, :K1] = R.T @ R
            if use_prior:
                H[:K1, :K1] += Sinv
        if has_pert:
            Q = Kt * r[:, None]
            H[P - M:, P - M:] = Q.T @ Q + 2.0 * lam_pert * C_pen + mu * np.eye(M)
            if train_w:
                H[:K1, P - M:] = R.T @ Q
                H[P - M:, :K1] = H[:K1, P - M:].T

        idx = np.where(free)[0]
        H_ff = H[np.ix_(idx, idx)]
        g_f = g[idx]
        Dscale = np.maximum(np.diag(H_ff), 1e-300)   # Marquardt scaling

        # --- Levenberg-damped Newton step, fraction-to-boundary line search ---
        accepted = False
        for _attempt in range(15):
            try:
                d_f = np.linalg.solve(H_ff + lam_lm * np.diag(Dscale), -g_f)
            except np.linalg.LinAlgError:
                lam_lm = max(lam_lm * 10.0, 1e-12)
                continue
            gd = float(g_f @ d_f)
            if not np.isfinite(gd) or gd >= 0:
                lam_lm = max(lam_lm * 10.0, 1e-12)
                continue

            d = np.zeros(P)
            d[idx] = d_f
            dw = d[:K1] if train_w else None
            dc = d[P - M:] if has_pert else None

            # fraction-to-boundary: largest step keeping f > 0 everywhere,
            # then never start the backtracking beyond 99.5% of it
            df = np.zeros(N)
            if train_w:
                df += Phi @ dw
            if has_pert:
                df += Kt @ dc
            negm = df < 0
            t_feas = float((-f[negm] / df[negm]).min()) if negm.any() else np.inf
            t0 = min(1.0, 0.995 * t_feas)

            t = t0
            for _ in range(40):
                w_t = w + t * dw if train_w else w
                if has_pert:
                    c_lin = c + t * dc
                    c_t = np.clip(c_lin, -box, box)
                    clipped = not np.array_equal(c_t, c_lin)
                else:
                    c_t, clipped = None, False
                if clipped:
                    L_t, f_t = objective(w_t, c_t)      # projection bent the step
                else:
                    f_t = f + t * df                     # exact for linear f
                    if f_t.min() <= 0.0:
                        L_t = np.inf
                    else:
                        L_t = -np.log(f_t).sum()
                        if use_prior:
                            dwp = w_t - wm
                            L_t += 0.5 * float(dwp @ Sinv @ dwp)
                        if has_pert and lam_pert > 0:
                            ce = c_t - c_t.mean()
                            L_t += lam_pert * float(ce @ ce)
                if np.isfinite(L_t) and L_t <= L + 1e-4 * t * gd:
                    accepted = True
                    break
                t *= 0.5
            if accepted:
                # near-full step -> relax damping; truncated step -> stiffen
                lam_lm = lam_lm * 0.25 if t >= 0.5 * t0 else min(lam_lm * 4.0 + 1e-12, 1e8)
                if lam_lm < 1e-14:
                    lam_lm = 0.0
                break
            lam_lm = max(lam_lm * 10.0, 1e-12)
        if not accepted:
            hit_max = False              # cannot improve further
            break

        newton_dec = -gd
        dL = L - L_t
        if train_w:
            w = w_t
        if has_pert:
            c = c_t
        L, f = L_t, f_t
        loss_hist.append(L)

        if verbose:
            print(f"  [{name}] it {it:3d}  L {L:.6f}  |pg| {grad_norm:.3e}  "
                  f"dec {newton_dec:.3e}  t {t:.2e}  lm {lam_lm:.1e}  "
                  f"fmin {f.min():.3e}", flush=True)

        # Newton decrement: for this barrier-type (self-concordant) objective,
        # -g.d of the UNDAMPED step bounds the remaining suboptimality of the
        # penalized log-likelihood — the quantity T actually depends on.
        if newton_dec < dec_tol and lam_lm == 0.0:
            hit_max = False
            break
        # loss-stall fallback: flat-valley crawl where meaningless (near-null)
        # weight directions never settle but the likelihood no longer moves
        stall = stall + 1 if dL < 1e-9 * (1.0 + abs(L)) else 0
        if stall >= 3:
            hit_max = False
            break

    # final projected-gradient norm for reporting
    r = 1.0 / f
    g = np.empty(P)
    if train_w:
        g_w = -(Phi.T @ r)
        if use_prior:
            g_w = g_w + Sinv @ (w - wm)
        g[:K1] = g_w
    if has_pert:
        g_c = -(Kt.T @ r) + 2.0 * lam_pert * (c - c.mean())
        at_hi = c >= box * (1 - 1e-14)
        at_lo = c <= -box * (1 - 1e-14)
        blocked = (at_hi & (g_c <= 0)) | (at_lo & (g_c >= 0))
        g[P - M:] = np.where(blocked, 0.0, g_c)
    if train_w and not has_pert:
        pg = g[:K1]
    else:
        pg = g
    grad_norm = float(np.linalg.norm(pg))

    converged = (grad_norm < GRAD_NORM_OK) or (newton_dec < dec_tol)
    flag = "OK" if converged else "WARN"
    print(f"[{name}] exact fit [{flag}]  loss={L:.6f}  ||pg||={grad_norm:.2e}  "
          f"dec={newton_dec:.2e}  n_iter={len(loss_hist) - 1}  fmin={f.min():.3e}",
          flush=True)

    return result(L, f, grad_norm, len(loss_hist) - 1, converged,
                  hit_max, loss_hist)


# ======================================================================
# Simplex-constrained profiling fit (single-model LRT control).
#
# The single SParKer model is trained as a POSITIVE, sum-normalized mixture
# (EstimationKernels.py: positive_coeffs=True -> clip_coeffs clamps to
# [0, coeffs_clip] every step; probability_coeffs=True -> density / |sum c|).
# So its natural weight domain is the probability simplex, NOT free signed
# reals. Free signed-weight profiling (the old `_free_gmm_weights` mode) is
# ill-posed: the signed-mixture log-likelihood is unbounded/degenerate along
# recession/near-null directions (weights ran to 1e4 in 2D, 1e12 in 4D).
# Constraining the weights to the simplex makes the feasible set COMPACT, so
# the problem is bounded, the MIXTURE density stays positive (convex combo of
# positive Gaussians), and the profiling stays inside the family the model was
# actually trained in.
#
# Alignment with wifi_better_basis/classifier_gof.py: SAME goal (positive
# density) and SAME solver philosophy (scipy + closed-form Hessian, à la
# fit_classifier), but a DIFFERENT positivity mechanism. The classifier is
# log-space (ratio = exp(w·F + b·G) > 0 for any w) so its weights are free-
# sign + Gaussian prior; the kernels are density-space (f = Σ w_k g_k) so
# positivity must come from w on the simplex. One residual gap the classifier
# doesn't have: the numerator kernel perturbation Σ c_eff_j G_j is added in
# DENSITY space, so it can in principle push f<0 (Sean's negative-density
# mechanism). The simplex only guarantees the MIXTURE part is positive; the
# caller must hard-check f>0 at the numerator optimum (there is NO silent
# density clamp here — the 1e-300 in the log is an iteration-time NaN guard
# only, and the compact feasible set means it cannot enable a runaway).
#
#   maximize   sum_i log f_i                       (over w, and c_eff if kernels)
#   f_i      = a_i + Phi_i . w + Kmat_i . c_eff
#   subject to w_k >= 0,  sum_k w_k <= 1           (=> derived weight 1-sum w >= 0)
#              |c_eff_j| <= clip,  sum_j c_eff = 0  (zero-mass, normalization-preserving)
#   objective  -sum_i log f_i + lam_pert * ||c_eff||^2     (convex; +convex penalty)
#
# Conventions match TAU / the density-space LRT: callers pass
#   Phi = comps[:, :-1] - comps[:, -1:],  a = comps[:, -1]
# and c_eff are the mean-centred (sum=0) kernel coefficients that
# GaussianKernelLayer.get_coefficients() would return.
# ======================================================================

def fit_lrt_simplex(Phi, a, w_init, Kmat=None, clip=None, lam_pert=0.0,
                    max_iter=1000, gtol=1e-8, xtol=1e-12, verbose=False, name="fit"):
    """
    Kernel-coefficient handling MATCHES TAU exactly (this is deliberate — an
    earlier version added an explicit `sum c_eff = 0` equality constraint and a
    density-space box on c_eff, which trust-constr could not satisfy on the 4D
    numerator: 400 iters, KKT 3e3, cviol 3e-2). TAU instead keeps the RAW
    coefficients c as the free parameter, box-clips them, and uses the
    mean-centred c_eff = c - mean(c) inside the density and the L2 penalty
    (GaussianKernelLayer.get_coefficients / net_coeffs_L2). Baking the
    mean-centring into a row-mean-centred kernel matrix Kt_c = Kt - rowmean
    means c_eff enters via Kt_c @ c with NO equality constraint — the only hard
    constraints left are the simplex on w (bounds + sum w <= 1) and the box on c.
    A tiny ridge mu*||c||^2 pins the otherwise-flat constant direction of c
    (c -> c + alpha leaves c_eff, the density and the L2 unchanged) so the
    Hessian is PD and the solution is unique.

    Returns dict (keys parallel to fit_lrt_exact where they apply):
        loss       : final penalized objective  (-sum log f + lam ||c_eff||^2)
        loglik     : sum_i log f_i at the optimum (no penalty)
        f          : (N,) fitted density at data points
        w          : (K-1,) profiled free mixture weights (on the simplex)
        c, c_eff   : (M,) mean-centred kernel coefficients (== get_coefficients()); None if no kernels
        grad_norm  : scaled projected-gradient norm at the solution
        n_iter, converged, hit_max_iter, fmin, n_at_clip

    SOLVER: projected damped-Newton with a FEASIBILITY line search. A general
    constrained optimiser (scipy trust-constr) fails the numerator in BOTH 2D and
    4D: it does not know the density must stay positive, so it steps into f<0
    (KKT ~ 1e10, fmin < 0). Here the log is its own barrier — the backtracking
    line search NEVER accepts a step with f<=0, so the iterate stays in the
    positive-density region by construction (no floor/clamp band-aid). The
    weights are projected onto the simplex {w>=0, sum w<=1} and the kernels onto
    the box each step; the objective is convex so any feasible start reaches the
    same optimum. Closed-form gradient/Hessian; converges in ~tens of iterations
    (trust-constr needed hundreds and ~10 min/fit).
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    a = np.ascontiguousarray(a, dtype=np.float64)
    N, K1 = Phi.shape
    has_pert = Kmat is not None
    if has_pert:
        Kt = np.ascontiguousarray(Kmat, dtype=np.float64)
        Ktc = Kt - Kt.mean(axis=1, keepdims=True)   # row-mean-centred: Ktc @ c = Kt @ (c - mean c)
        M = Kt.shape[1]
        box = np.inf if clip is None else float(clip)
        mu_ridge = 1e-6 * max(lam_pert, 1.0)         # pins the constant direction of c
    else:
        M = 0
    P = K1 + M

    # ---- projections onto the feasible set ----
    def proj_w(v):
        """Euclidean projection onto {w >= 0, sum w <= 1}."""
        w = np.maximum(v, 0.0)
        if w.sum() <= 1.0:
            return w
        # else project onto the probability simplex {w >= 0, sum w = 1} (Duchi et al.)
        u = np.sort(v)[::-1]
        css = np.cumsum(u)
        k = np.arange(1, v.size + 1)
        cond = u - (css - 1.0) / k > 0
        rho = np.nonzero(cond)[0][-1]
        theta = (css[rho] - 1.0) / (rho + 1)
        return np.maximum(v - theta, 0.0)

    def proj(x):
        if has_pert:
            return np.concatenate([proj_w(x[:K1]), np.clip(x[K1:], -box, box)])
        return proj_w(x)

    def dens(x):
        f = a + Phi @ x[:K1]
        if has_pert:
            f = f + Ktc @ x[K1:]
        return f

    def objective(f, x):
        if f.min() <= 0.0:
            return np.inf
        L = -np.log(f).sum()
        if has_pert:
            c = x[K1:]; ce = c - c.mean()
            L += lam_pert * float(ce @ ce) + mu_ridge * float(c @ c)
        return float(L)

    def grad(f, x):
        r = 1.0 / f
        g = np.empty(P)
        g[:K1] = -(Phi.T @ r)
        if has_pert:
            c = x[K1:]
            g[K1:] = -(Ktc.T @ r) + 2.0 * lam_pert * (c - c.mean()) + 2.0 * mu_ridge * c
        return g

    def hess(f):
        D = 1.0 / f ** 2
        H = np.zeros((P, P))
        H[:K1, :K1] = Phi.T @ (Phi * D[:, None])
        if has_pert:
            KtcD = Ktc * D[:, None]
            Pmat = np.eye(M) - 1.0 / M               # I - 11^T/M (mean-centring projector)
            H[K1:, K1:] = Ktc.T @ KtcD + 2.0 * lam_pert * Pmat + 2.0 * mu_ridge * np.eye(M)
            H[:K1, K1:] = Phi.T @ KtcD
            H[K1:, :K1] = H[:K1, K1:].T
        return H

    # ---- feasible start: trained weights projected onto the simplex, kernels 0 ----
    w0 = proj_w(np.array(w_init, dtype=np.float64).ravel())
    x = np.concatenate([w0, np.zeros(M)]) if has_pert else w0
    f = dens(x)
    if f.min() <= 0.0:
        raise RuntimeError(f"[{name}] infeasible start: fmin = {f.min():.3e} <= 0")
    L = objective(f, x)

    lam_lm = 1e-6          # Levenberg damping (adapted)
    gnorm = np.inf
    stall = 0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g = grad(f, x)
        H = hess(f)
        dH = np.maximum(np.diag(H), 1e-30)          # diagonal scaling for the KKT measure
        gnorm = float(np.linalg.norm(x - proj(x - g / dH)))
        if gnorm < gtol:
            converged = True
            break

        # Levenberg-damped Newton direction
        d = None
        for _ in range(30):
            try:
                d = np.linalg.solve(H + lam_lm * np.diag(dH), -g)
                break
            except np.linalg.LinAlgError:
                lam_lm = max(lam_lm * 10.0, 1e-12)
        if d is None:
            d = -g / dH

        # feasibility + Armijo backtracking along the PROJECTED arc (never accept f<=0)
        t = 1.0
        accepted = False
        for _ in range(60):
            xt = proj(x + t * d)
            ft = dens(xt)
            if ft.min() > 0.0:
                Lt = objective(ft, xt)
                if Lt <= L + 1e-4 * float(g @ (xt - x)):
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            lam_lm = min(lam_lm * 10.0 + 1e-12, 1e12)
            stall += 1
            if stall >= 3:
                break
            continue
        stall = 0
        dL = L - Lt
        x, f, L = xt, ft, Lt
        lam_lm = max(lam_lm * 0.5, 1e-12)
        if dL < 1e-12 * (1.0 + abs(L)):             # objective plateau
            converged = True
            break

    # final KKT measure
    g = grad(f, x)
    dH = np.maximum(np.diag(hess(f)), 1e-30)
    gnorm = float(np.linalg.norm(x - proj(x - g / dH)))

    w = x[:K1].copy()
    c_raw = x[K1:].copy() if has_pert else None
    c_eff = (c_raw - c_raw.mean()) if has_pert else None
    converged = bool(converged and f.min() > 0.0)

    flag = "OK" if converged else "WARN"
    print(f"[{name}] simplex fit [{flag}]  loss={L:.6f}  kkt={gnorm:.2e}  "
          f"n_iter={it}  fmin={float(f.min()):.3e}  sum(w)={float(w.sum()):.4f}", flush=True)

    return {
        "loss": float(L),
        "loglik": float(np.log(f).sum()),
        "f": f,
        "w": w,
        "c": c_eff,
        "c_eff": c_eff,
        "grad_norm": gnorm,
        "n_iter": int(it),
        "converged": converged,
        "hit_max_iter": bool(it >= max_iter and not converged),
        "fmin": float(f.min()),
        "n_at_clip": (int(np.sum(np.abs(c_raw) >= box * 0.999)) if has_pert else 0),
        "constr_violation": 0.0,
    }


# ======================================================================
# EM fit of mixture weights on the probability simplex — the solver for the
# one-model GoF numerator in "Sean Option 2" (2026-07-20): keep the kernels
# ADDITIVE but force ALL weights positive and summing to 1, over ONE simplex
# that spans the model components AND the kernels. That makes the numerator a
# plain positive mixture:
#     f(x) = sum_c theta_c C_c(x),   theta >= 0,  sum theta = 1,   C_c >= 0
# so the density is positive by construction (no f<0), the feasible set is
# compact (bounded), and — for the LRT — the denominator (a mixture over the
# model components only) is NESTED in the numerator (set the kernel weights to
# 0), so T >= 0. Maximum-likelihood over the simplex with fixed component
# densities is exactly the classic mixture-weight problem, which EM solves:
#     E/M update:  theta_c <- theta_c * (1/N) * sum_i C[i,c] / (C theta)_i
# This is guaranteed monotonic, keeps theta on the simplex and positive, has no
# line search / Hessian / step size, and cannot blow up (C theta > 0 as long as
# any component explains each point — true here, since the model components do).
# Concave in theta -> unique global optimum -> start-independent.
#
# DENOMINATOR:  C = the K model-component densities (comps).            -> theta = w
# NUMERATOR:    C = [comps | kernels]  (K + M columns).                 -> theta = [w | v]
# No clip, no L2, no mean-centring (Sean's formulation): the simplex is the only
# constraint and it does all the regularising. (An optional L2/entropy on the
# kernel block could be added later; start without, per Sean.)
# ======================================================================

def fit_simplex_em(C, theta_init=None, max_iter=20000, tol=1e-11, name="fit", accelerate=True):
    """
    Maximize sum_i log( (C @ theta)_i ) over {theta >= 0, sum theta = 1} by EM.

    C : (N, P) array of per-component densities at the data (all >= 0).
    Returns dict:
        theta     : (P,) MLE mixture weights on the simplex
        loglik    : sum_i log (C @ theta) at the optimum
        f         : (N,) fitted density
        n_iter, converged, hit_max_iter, fmin, n_active
    """
    C = np.ascontiguousarray(C, dtype=np.float64)
    N, P = C.shape

    if theta_init is None:
        theta = np.full(P, 1.0 / P)
    else:
        theta = np.maximum(np.asarray(theta_init, dtype=np.float64).ravel(), 0.0)
        s = theta.sum()
        theta = theta / s if s > 0 else np.full(P, 1.0 / P)

    f = C @ theta
    if f.min() <= 0.0:
        raise RuntimeError(f"[{name}] EM start infeasible: fmin = {f.min():.3e} <= 0 "
                           "(some data point has zero density under every component).")

    def em_step(th):
        """One EM map: th_c <- th_c * (1/N) sum_i C[i,c]/(C th)_i. Stays on the simplex."""
        return th * (C.T @ (1.0 / (C @ th))) / N

    def loglik(th):
        return float(np.log(C @ th).sum())

    ll = loglik(theta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if not accelerate:
            theta = em_step(theta)
            ll_new = loglik(theta)
        else:
            # SQUAREM (Varadhan & Roland): extrapolate two EM steps, then stabilise
            # with one more EM map. Monotonicity safeguard: never do worse than the
            # plain double-EM point th2 (loglik(th2) >= loglik(theta) always).
            th1 = em_step(theta)
            th2 = em_step(th1)
            r = th1 - theta
            v = (th2 - th1) - r
            vn = float(np.linalg.norm(v))
            if vn < 1e-300:
                theta, ll_new = th2, loglik(th2)
            else:
                alpha = -float(np.linalg.norm(r)) / vn      # SQUAREM-3 steplength (<= -1)
                th_new = theta - 2.0 * alpha * r + (alpha ** 2) * v
                k = 0
                while th_new.min() < 0.0 and k < 40:        # back off toward alpha=-1 (=> th2)
                    alpha = (alpha - 1.0) / 2.0
                    th_new = theta - 2.0 * alpha * r + (alpha ** 2) * v
                    k += 1
                if th_new.min() < 0.0:
                    th_new = th2
                th_new = np.maximum(th_new, 0.0)
                th_new = th_new / th_new.sum()
                th_stab = em_step(th_new)
                ll_stab = loglik(th_stab)
                ll_th2 = loglik(th2)
                if ll_stab >= ll_th2:                        # keep the accelerated step
                    theta, ll_new = th_stab, ll_stab
                else:                                        # safeguard: fall back to plain EM
                    theta, ll_new = th2, ll_th2
        if ll_new - ll < tol * (1.0 + abs(ll)):
            ll = ll_new
            converged = True
            break
        ll = ll_new

    f = C @ theta
    print(f"[{name}] EM{'+sqrm' if accelerate else ''}  loglik={ll:.6f}  n_iter={it}  "
          f"conv={converged}  fmin={float(f.min()):.3e}  "
          f"n_active={int((theta > 1e-8).sum())}/{P}  sum(theta)={float(theta.sum()):.6f}",
          flush=True)

    return {
        "theta": theta,
        "loglik": ll,
        "f": f,
        "n_iter": int(it),
        "converged": converged,
        "hit_max_iter": bool(it >= max_iter and not converged),
        "fmin": float(f.min()),
        "n_active": int((theta > 1e-8).sum()),
    }


# ======================================================================
# NPLM exponential-tilt fit — the solver for the one-model GoF numerator in
# "Sean Option 1" (multiplicative, 2026-07-21). Instead of ADDING kernels to the
# density (Option 2), MULTIPLY the DEN-optimal model by an exponential tilt:
#     f_num(x) ∝ f_mix(x; w_den) · exp( tau(x) ),   tau(x) = sum_j b_j G_j(x).
# This is the standard NPLM form and exactly what wifi_better_basis/classifier_gof.py
# does (there the tilt is added in logit space -> multiplicative on the ratio).
#
# NORMALIZATION. The proper density needs Z(b) = E_{f_mix}[exp tau], which has no
# closed form (GMM × exp-of-Gaussians). We estimate it with a REFERENCE SAMPLE
# drawn from the DEN-optimal model (the caller samples the GMM at w_den). Because
# the reference is drawn from exactly the DEN model, the shared sum_i log f_mix(x_i)
# term CANCELS between numerator and denominator, leaving the pure NPLM statistic
#     T = 2 · max_b [ sum_data tau(x_i) - N · log( (1/R) sum_ref exp tau(y_r) ) ].
#
# CONVEXITY. With the mixture weights FROZEN at w_den (only b is fit), the objective
#     L(b) = - s_data·b + N·( logsumexp(K_ref b) - log R )   [ + 0.5 lam ||b||^2 ]
# (s_data = sum_i K_data[i]) is linear minus N·log-sum-exp, hence CONVEX: the
# Hessian N·Cov_softmax(K_ref) [+ lam I] is PSD, so the optimum is unique and any
# Newton solver converges. Density is positive by construction (exp-tilt > 0) and
# T >= 0 (b=0 recovers the denominator, L(0)=0). This mirrors fit_classifier in
# classifier_gof.py (scipy trust-exact + closed-form gradient/Hessian).
#
# CO-PROFILING w with b is possible (keeps w coupled to b via an importance-
# weighted Z) but makes L NON-convex — it reintroduces the non-convergence Option 2
# escaped — so it is deliberately NOT done here; w is profiled in the denominator
# only. See [[project-lrt-onemodel-signed-weight-runaway]].
# ======================================================================

def build_eval_grid(data, grid_points, pad=0.2, cover_lo=None, cover_hi=None,
                    max_points=4_000_000):
    """Uniform tensor-product grid for deterministic quadrature of the tilt
    normalization Z in low dimension (the grid-Z / one-sample path).

    data        : (N, d) array; the grid spans [min - pad*span, max + pad*span]
                  per dimension.
    grid_points : points per dimension (total = grid_points**d).
    cover_lo/hi : optional (d,) arrays; the extent is unioned with them so the
                  model's own support is covered even where there are few data.
    Returns (grid, step): grid (grid_points**d, d) float64, step (d,) spacing.
    Raises if grid_points**d > max_points (grid quadrature is for low d only;
    use the sample path in high d).
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    d = data.shape[1]
    G_total = grid_points ** d
    if G_total > max_points:
        raise ValueError(f"grid too large: {grid_points}**{d} = {G_total} > {max_points}. "
                         "Grid-Z is for low d (2D); use the sample path in high d.")
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    span = hi - lo
    lo = lo - pad * span
    hi = hi + pad * span
    if cover_lo is not None:
        lo = np.minimum(lo, np.asarray(cover_lo, dtype=np.float64))
    if cover_hi is not None:
        hi = np.maximum(hi, np.asarray(cover_hi, dtype=np.float64))
    axes = [np.linspace(lo[k], hi[k], grid_points) for k in range(d)]
    mesh = np.meshgrid(*axes, indexing='ij')
    grid = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float64)
    step = (hi - lo) / (grid_points - 1)
    return grid, step


def fit_nplm_tilt(K_data, K_ref, lam_pert=0.0, clip=None, log_w_ref=None,
                  max_iter=500, tol=1e-9, verbose=False, name="NUM"):
    """
    Maximize  ll(b) = sum_data tau(x_i) - N * log( Z(b) )
    over the tilt coefficients b, with tau(x) = sum_j b_j G_j(x), where
        log Z(b) = logsumexp_r ( a_r + tau(y_r) ),   sum_r exp(a_r) = 1,
    estimates the normalization  Z = E_{f0}[exp tau]  over the reference points.
    The optimum ll(b*) IS T/2 (the shared log f0 term has cancelled; see header).

    K_data : (N, M) kernel matrix G_j(x_i) at the data points.
    K_ref  : (R, M) kernel matrix G_j(y_r) at the reference points y_r.
    lam_pert : optional L2 ridge 0.5*lam*||b||^2 on the tilt coeffs (= classifier
               lam_pert; default 0 => pure MLE).
    clip   : optional symmetric box |b_j| <= clip (switches solver to L-BFGS-B).
    log_w_ref : optional (R,) reference-measure log-weights a_r (self-normalized
               internally so sum_r exp(a_r)=1, giving Z(0)=1 exactly).
               - None  => SAMPLE mode: y_r ~ f0, uniform a_r = -log R  (two-sample-
                 looking; the finite-R MC noise broadens the null).
               - given => GRID/quadrature mode: y_r on a fixed grid with
                 a_r = log( f0(y_r) ) (+ log cell-volume; the constant cancels).
                 This makes Z deterministic -> a genuine one-sample test.

    Returns dict:
        loglik    : ll(b*) = T/2
        T         : 2 * ll(b*)  (>= 0 by construction)
        b         : (M,) fitted tilt coefficients (signed)
        tau_data  : (N,) tau(x_i) at the optimum
        logZ      : log of the normalization estimate at the optimum
        grad_norm, n_iter, converged, hit_max_iter, loss_hist
    """
    from scipy.optimize import minimize as _scipy_minimize
    from scipy.special import logsumexp as _logsumexp

    K_data = np.ascontiguousarray(K_data, dtype=np.float64)
    K_ref  = np.ascontiguousarray(K_ref,  dtype=np.float64)
    N, M = K_data.shape
    R = K_ref.shape[0]
    s_data = K_data.sum(axis=0)          # (M,)  sum_i G_j(x_i)
    # Reference-measure log-weights a_r with sum_r exp(a_r) = 1, so that
    #   Z(b) = sum_r exp(a_r + tau(y_r))  estimates E_{f0}[exp tau] and Z(0) = 1.
    # SAMPLE mode (log_w_ref=None): y_r ~ f0, uniform a_r = -log R.
    # GRID   mode (log_w_ref given): y_r on a grid, a_r = log f0(y_r) (self-normalized).
    if log_w_ref is None:
        a = np.full(R, -math.log(R))
    else:
        a = np.ascontiguousarray(log_w_ref, dtype=np.float64).ravel()
        if a.shape != (R,):
            raise ValueError(f"log_w_ref shape {a.shape} != ({R},)")
        a = a - _logsumexp(a)            # self-normalize -> sum exp(a) = 1 -> Z(0)=1
    loss_hist = []

    def _softmax_w(z):
        za = z + a
        za = za - za.max()
        e = np.exp(za)
        return e / e.sum()

    def fun(b):
        z = K_ref @ b                    # (R,)
        L = -(s_data @ b) + N * _logsumexp(z + a)
        if lam_pert > 0:
            L += 0.5 * lam_pert * float(b @ b)
        loss_hist.append(L)
        return L

    def jac(b):
        z = K_ref @ b
        p = _softmax_w(z)                # (R,)
        g = -s_data + N * (K_ref.T @ p)  # -sum_data G + N * measure-weighted ref mean
        if lam_pert > 0:
            g = g + lam_pert * b
        return g

    def hess(b):
        z = K_ref @ b
        p = _softmax_w(z)                # (R,)
        m = K_ref.T @ p                  # (M,)
        H = N * (K_ref.T @ (K_ref * p[:, None]) - np.outer(m, m))   # N * Cov_softmax(K_ref)
        if lam_pert > 0:
            H = H + lam_pert * np.eye(M)
        return H

    b0 = np.zeros(M, dtype=np.float64)
    if clip is None:
        res = _scipy_minimize(fun, b0, jac=jac, hess=hess, method="trust-exact",
                              options={"maxiter": max_iter, "gtol": tol})
    else:
        box = float(clip)
        res = _scipy_minimize(fun, b0, jac=jac, method="L-BFGS-B",
                              bounds=[(-box, box)] * M,
                              options={"maxiter": max_iter, "ftol": 1e-14, "gtol": tol})

    b_final = res.x
    n_iter = int(getattr(res, "nit", 0))
    ll = -float(res.fun)                 # ll(b*) = -L(b*) >= 0
    grad_norm = float(np.linalg.norm(jac(b_final)))
    max_b = float(np.abs(b_final).max())
    hit_max_iter = n_iter >= max_iter
    # A small gradient at a HUGE b is the separation signature: with no (or too
    # little) regularisation the tilt runs to infinity along kernels that have
    # data support but little reference support, and the log-sum-exp gradient
    # goes flat there. So require a small gradient AND that we did not exhaust the
    # iteration budget; flag a runaway on max|b|.
    runaway = max_b > 1e3
    converged = (grad_norm < GRAD_NORM_OK) and (not hit_max_iter) and (not runaway)
    T = 2.0 * ll

    tau_data = K_data @ b_final          # (N,)
    logZ = float(_logsumexp(K_ref @ b_final + a))

    flag = "OK" if converged else "WARN"
    print(f"[{name}] nplm-tilt [{flag}]  ll={ll:.6f}  T={T:.4f}  ||g||={grad_norm:.2e}  "
          f"n_iter={n_iter}  logZ={logZ:.4f}  max|b|={max_b:.3e}", flush=True)
    if runaway or hit_max_iter:
        print(f"[{name}] WARNING: max|b|={max_b:.3e}, n_iter={n_iter}/{max_iter} — "
              f"likely tilt runaway (separation). Increase lam_pert (now {lam_pert}) "
              f"or set clip.", flush=True)

    return {
        "loglik": ll,
        "T": T,
        "b": b_final,
        "tau_data": tau_data,
        "logZ": logZ,
        "grad_norm": grad_norm,
        "max_b": max_b,
        "n_iter": n_iter,
        "converged": converged,
        "hit_max_iter": bool(hit_max_iter),
        "runaway": bool(runaway),
        "loss_hist": np.array(loss_hist, dtype=np.float64),
    }


def fit_nplm_tilt_constrained(P_data, w_hat, Sigma_w,
                              K_data=None, P_ref=None, K_ref=None, q_ref=None,
                              fit_w=True, lam_pert=0.0, ridge_rel=1e-8,
                              u_init=None, b_init=None,
                              max_iter=500, tol=1e-9, name="NUM"):
    """
    CONSTRAINED multiplicative NPLM tilt in DENSITY space: jointly profile the
    ensemble weights w (Gaussian prior N(w_hat, Sigma_w) on the M-1 free weights u)
    and the exp-tilt coeffs b. Density twin of classifier_gof.fit_classifier.

        f_model(x) = f_ens(x; w) * exp(tau(x; b)) / Z(w, b),   tau = sum_j b_j G_j
        f_ens(x; w) = P(x) . w,   w = [u, 1 - sum(u)]         (sum-to-one; u free)
        Z(w, b)     = E_{f_ens(w)}[exp tau]  ~  sum_r m_r(u) exp(tau(y_r)),
                      m_r(u) proportional to f_ens(y_r; u)/q(y_r)   (importance from q)

    Objective MINIMIZED (MAP; the b-ridge only shapes the fit, it is NOT in the
    reported statistic — see the T assembly in LRT.py):
        L(u,b) = -sum_i log f_ens(x_i; u) - sum_i tau(x_i) + N log Z(u,b)
                 + 1/2 (u-û)^T Sigma_w^{-1} (u-û) + 1/2 lam_pert ||b||^2

    Legs:
      DEN  : K_data=None            -> b absent; Z=1; L(u) = -sum log f_ens + prior.
      NUM  : K_data given           -> joint (u, b).
      fit_w=False fixes u=û: then the b-fit is IDENTICAL to fit_nplm_tilt (the frozen
             path) — used as a reduction/validation check.

    NON-CONVEX in u (log Z couples u and b); locally convex at the null and the
    Sigma_w prior + start at (û, 0) stabilize. Solved with L-BFGS-B on the analytic
    gradient (no Hessian; robust to indefiniteness). Note dlogZ/du = 0 at b=0, so at
    the null the w-fit of NUM coincides with the DEN w-fit.

    Returns dict:
        u, w, b            : fitted free weights, full weight vector, tilt (b None for DEN)
        ll                 : sum_i log f_model(x_i)  — the DATA loglik (prior- & ridge-free);
                             the ingredient for T = 2[(ll_num+logprior_num)-(ll_den+logprior_den)]
        logprior           : -1/2 (u-û)^T Sigma_w^{-1} (u-û)   (0 if fit_w=False)
        log_model_data     : (N,) per-event log f_model(x_i)
        logZ, grad_norm, max_b, n_iter, converged, hit_max_iter
    """
    from scipy.optimize import minimize as _scipy_minimize
    from scipy.special import logsumexp as _logsumexp

    P_data = np.ascontiguousarray(P_data, dtype=np.float64)      # (N, M)
    N, M = P_data.shape
    w_hat = np.asarray(w_hat, dtype=np.float64).ravel()          # (M,)
    u_hat = w_hat[:-1].copy()                                    # (M-1,)
    Pd_data = P_data[:, :-1] - P_data[:, -1:]                    # (N, M-1) = df/du
    pl_data = P_data[:, -1]                                      # (N,)

    has_b = K_data is not None
    if has_b:
        K_data = np.ascontiguousarray(K_data, dtype=np.float64)  # (N, Mk)
        K_ref  = np.ascontiguousarray(K_ref,  dtype=np.float64)  # (R, Mk)
        P_ref  = np.ascontiguousarray(P_ref,  dtype=np.float64)  # (R, M)
        q_ref  = np.ascontiguousarray(q_ref,  dtype=np.float64).ravel()  # (R,)
        Mk = K_data.shape[1]
        s_data = K_data.sum(axis=0)                              # (Mk,)
        Pd_ref = P_ref[:, :-1] - P_ref[:, -1:]                   # (R, M-1)
        pl_ref = P_ref[:, -1]                                    # (R,)
        log_q  = np.log(np.maximum(q_ref, 1e-300))               # (R,)

    if fit_w:
        Sig = np.ascontiguousarray(Sigma_w, dtype=np.float64)
        K1 = Sig.shape[0]
        eps = ridge_rel * (np.trace(Sig) / max(K1, 1))
        Sw_inv = np.linalg.inv(Sig + eps * np.eye(K1))

    def _fval(u, Pd, pl):
        return np.maximum(Pd @ u + pl, 1e-300)

    def _unpack(x):
        if fit_w and has_b:
            return x[:M - 1], x[M - 1:]
        if fit_w:
            return x, None
        if has_b:
            return u_hat, x
        return u_hat, None

    loss_hist = []

    def fun(x):
        u, b = _unpack(x)
        f_d = _fval(u, Pd_data, pl_data)
        L = -np.log(f_d).sum()
        if has_b:
            f_r = _fval(u, Pd_ref, pl_ref)
            atil = np.log(f_r) - log_q
            atil = atil - _logsumexp(atil)                       # log m (self-normalized)
            logZ = _logsumexp(atil + (K_ref @ b))
            L += -(s_data @ b) + N * logZ
            if lam_pert > 0:
                L += 0.5 * lam_pert * float(b @ b)
        if fit_w:
            du = u - u_hat
            L += 0.5 * float(du @ Sw_inv @ du)
        loss_hist.append(L)
        return L

    def jac(x):
        u, b = _unpack(x)
        f_d = _fval(u, Pd_data, pl_data)
        g_u = None
        if fit_w:
            g_u = -(Pd_data.T @ (1.0 / f_d)) + Sw_inv @ (u - u_hat)
        if has_b:
            f_r = _fval(u, Pd_ref, pl_ref)
            atil = np.log(f_r) - log_q
            atil = atil - _logsumexp(atil)
            m = np.exp(atil)                                     # base IS weights (sum 1)
            zz = atil + (K_ref @ b)
            zz = zz - zz.max()
            p = np.exp(zz); p = p / p.sum()                      # tilted weights (sum 1)
            g_b = -s_data + N * (K_ref.T @ p)
            if lam_pert > 0:
                g_b = g_b + lam_pert * b
            if fit_w:
                inv_fr = 1.0 / f_r
                g_u = g_u + N * (Pd_ref.T @ (p * inv_fr) - Pd_ref.T @ (m * inv_fr))
        if fit_w and has_b:
            return np.concatenate([g_u, g_b])
        return g_u if fit_w else g_b

    def hess(x):
        # Analytic Hessian. Blocks (v_r = Pd_ref_r / f_ref_r):
        #   H_uu = Pd_d^T diag(1/f_d^2) Pd_d + Sigma^{-1} + N (v̄_m v̄_m^T - v̄_p v̄_p^T)
        #   H_bb = N [K^T diag(p) K - K̄_p K̄_p^T] + lam I
        #   H_ub = N [V^T diag(p) K - v̄_p K̄_p^T]
        # The logZ u-block is rank-2 (0 at b=0). Generally indefinite -> trust-exact.
        u, b = _unpack(x)
        f_d = _fval(u, Pd_data, pl_data)
        H_uu = None
        if fit_w:
            H_uu = (Pd_data.T @ (Pd_data * (1.0 / (f_d * f_d))[:, None])) + Sw_inv
        if has_b:
            f_r = _fval(u, Pd_ref, pl_ref)
            atil = np.log(f_r) - log_q
            atil = atil - _logsumexp(atil)
            m = np.exp(atil)
            zz = atil + (K_ref @ b)
            zz = zz - zz.max()
            p = np.exp(zz); p = p / p.sum()
            Kp = K_ref * p[:, None]
            Kbar = K_ref.T @ p
            H_bb = N * (K_ref.T @ Kp - np.outer(Kbar, Kbar))
            if lam_pert > 0:
                H_bb = H_bb + lam_pert * np.eye(Mk)
            if fit_w:
                V = Pd_ref / f_r[:, None]                # (R, M-1),  v_r
                vbar_p = V.T @ p
                vbar_m = V.T @ m
                H_ub = N * (V.T @ Kp - np.outer(vbar_p, Kbar))
                H_uu = H_uu + N * (np.outer(vbar_m, vbar_m) - np.outer(vbar_p, vbar_p))
        if fit_w and has_b:
            n = (M - 1) + Mk
            H = np.empty((n, n))
            H[:M - 1, :M - 1] = H_uu
            H[M - 1:, M - 1:] = H_bb
            H[:M - 1, M - 1:] = H_ub
            H[M - 1:, :M - 1] = H_ub.T
            return H
        return H_uu if fit_w else H_bb

    # Warm-start (crucial at production scale): u_init near u_den + b_init near the
    # frozen tilt optimum lands the joint fit next to its optimum, so trust-exact
    # converges in a few iters instead of stalling in the far-from-optimum region.
    u0 = u_hat.copy() if u_init is None else np.asarray(u_init, dtype=np.float64).ravel()
    b0 = None if not has_b else (np.zeros(Mk) if b_init is None
                                 else np.asarray(b_init, dtype=np.float64).ravel())
    if fit_w and has_b:
        x0 = np.concatenate([u0, b0])
    elif fit_w:
        x0 = u0
    elif has_b:
        x0 = b0
    else:
        x0 = np.zeros(0)

    if x0.size == 0:                        # frozen w, no b: nothing to optimize
        res_x = x0; n_iter = 0; grad_norm = 0.0; hit_max_iter = False
    else:
        # trust-exact: Newton trust-region with the exact (possibly indefinite)
        # Hessian — robust to the non-convexity in u (L-BFGS-B stalls here).
        res = _scipy_minimize(fun, x0, jac=jac, hess=hess, method="trust-exact",
                              options={"maxiter": max_iter, "gtol": tol})
        res_x = res.x
        n_iter = int(getattr(res, "nit", 0))
        grad_norm = float(np.linalg.norm(jac(res_x)))
        hit_max_iter = n_iter >= max_iter

    u_fin, b_fin = _unpack(res_x)
    w_fin = np.append(u_fin, 1.0 - u_fin.sum())
    log_model_data = np.log(_fval(u_fin, Pd_data, pl_data))      # (N,) log f_ens(x;u)
    logZ = 0.0
    if has_b:
        f_r = _fval(u_fin, Pd_ref, pl_ref)
        atil = np.log(f_r) - log_q
        atil = atil - _logsumexp(atil)
        logZ = float(_logsumexp(atil + (K_ref @ b_fin)))
        log_model_data = log_model_data + (K_data @ b_fin) - logZ
    ll = float(log_model_data.sum())
    logprior = -0.5 * float((u_fin - u_hat) @ Sw_inv @ (u_fin - u_hat)) if fit_w else 0.0
    max_b = float(np.abs(b_fin).max()) if has_b else 0.0
    # Convergence on the RELATIVE gradient: the objective and gradient scale with N
    # (g_b = -s_data + N*mean(...), both ~N), so an ABSOLUTE threshold is meaningless at
    # production N. grad_rel = ||g|| / N; a rel gradient < 1e-6 means the fit sits at the
    # optimum to ~1e-6 relative -> T accurate to many digits (verified: T stable to 4
    # digits across ||g|| spanning 2.5 -> 1e-2). Real-target (calib=0) fits can stall the
    # last relative 1e-7 due to near-zero f_ens stiffening the u-Hessian; T is unaffected.
    grad_rel = grad_norm / max(1.0, float(N))
    converged = (grad_rel < 1e-6) and (not hit_max_iter)

    flag = "OK" if converged else "WARN"
    print(f"[{name}] nplm-tilt-constrained [{flag}]  ll={ll:.4f}  logprior={logprior:.4f}  "
          f"||g||={grad_norm:.2e}  ||g||/N={grad_rel:.2e}  n_iter={n_iter}  logZ={logZ:.4f}  "
          f"max|b|={max_b:.3e}  fit_w={fit_w}  has_b={has_b}", flush=True)

    return {
        "u": u_fin, "w": w_fin, "b": (b_fin if has_b else None),
        "ll": ll, "logprior": logprior, "log_model_data": log_model_data,
        "logZ": logZ, "grad_norm": grad_norm, "grad_rel": grad_rel, "max_b": max_b,
        "n_iter": n_iter, "converged": converged, "hit_max_iter": bool(hit_max_iter),
        "loss_hist": np.array(loss_hist, dtype=np.float64),
    }
