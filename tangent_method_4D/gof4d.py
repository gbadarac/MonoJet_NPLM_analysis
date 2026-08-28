"""Core library for the 4D embedding GOF experiment (tangent-space composite test).

Same math as gof4d_embeddings_tangent.ipynb, plus ANALYTIC GMM scores (closed-form
per-sample gradient of log p w.r.t. all parameters) replacing finite differences:
~P x faster (P = 15K-1 at d=4). Validated against finite differences in prep.py.
"""
import glob, hashlib
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import config as C

# ---------------- seeding ----------------------------------------------------
def rng_for(key: str) -> np.random.Generator:
    h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(np.random.SeedSequence(C.BASE_SEED, spawn_key=(h,)))

# ---------------- data pool ---------------------------------------------------
_POOL_CACHE = {}

def load_partition():
    """Load, standardize, shuffle. Partition v2 (ORDER-FREE):
      - wfit: first N_WFIT events of the shuffled pool (fixed);
      - training set for a given N_fit: drawn from the remainder by an RNG keyed on
        N ITSELF (not on the position in n_fit_list) -> N_fit values can be added in
        any order, any size, without touching other models' data;
      - observed sets for model N: drawn from the remainder EXCLUDING model N's own
        training events (training sets of different N may overlap each other, which
        is harmless: no model is ever tested on its own training events)."""
    if "part" in _POOL_CACHE:
        return _POOL_CACHE["part"]
    files = sorted(glob.glob(C.DATA_GLOB))
    assert files, f"no files match {C.DATA_GLOB}"
    pool = np.concatenate([np.load(f)[C.DATA_KEY] for f in files]).astype(np.float64)
    if C.STANDARDIZE:
        pool = (pool - pool.mean(axis=0)) / pool.std(axis=0)
    pool = pool[np.random.default_rng(C.BASE_SEED).permutation(len(pool))]
    part = dict(X_wfit=pool[:C.N_WFIT], REST=pool[C.N_WFIT:], dim=pool.shape[1])
    _POOL_CACHE["part"] = part
    return part

_TRAIN_IDX = {}

def train_idx(N):
    if N not in _TRAIN_IDX:
        rest = load_partition()["REST"]
        assert N < len(rest), f"N_fit={N} exceeds the available pool"
        _TRAIN_IDX[N] = rng_for(f"dens:{N}").choice(len(rest), size=N, replace=False)
    return _TRAIN_IDX[N]

def get_train(N):
    return load_partition()["REST"][train_idx(N)]

_TEST_IDX = {}

def sample_data(nt, rng, N_excl, eps=0.0, coord=0):
    """Observed sample from the empirical truth, excluding model N_excl's training
    events; eps scales `coord` about its mean."""
    rest = load_partition()["REST"]
    if N_excl not in _TEST_IDX:
        mask = np.ones(len(rest), dtype=bool)
        mask[train_idx(N_excl)] = False
        _TEST_IDX[N_excl] = np.flatnonzero(mask)
    avail = _TEST_IDX[N_excl]
    assert nt <= len(avail), f"N_test={nt} exceeds pool minus training block"
    X = rest[avail[rng.choice(len(avail), size=int(nt), replace=False)]].copy()
    if eps:
        mu = rest[:, coord].mean()
        X[:, coord] = mu + (1.0 + eps) * (X[:, coord] - mu)
    return X

# ---------------- general-d GMM ------------------------------------------------
def n_params(K, d): return K - 1 + K*d + K*(d*(d+1))//2

def pack_params(w, means, covs):
    K = len(w); d = means.shape[1]
    il, jl = np.tril_indices(d)
    beta = np.log(w[1:]/w[0]); ch = []
    for S in covs:
        L = np.linalg.cholesky(S)
        v = L[il, jl].copy()
        v[np.where(il == jl)[0]] = np.log(np.diag(L))
        ch.append(v)
    return np.concatenate([beta, np.asarray(means).ravel(), np.concatenate(ch)])

def unpack_params(theta, K, d):
    il, jl = np.tril_indices(d); nch = len(il)
    beta = np.concatenate([[0.0], theta[:K-1]])
    w = np.exp(beta - beta.max()); w /= w.sum()
    means = theta[K-1:K-1+K*d].reshape(K, d)
    rest = theta[K-1+K*d:].reshape(K, nch)
    Ls = np.zeros((K, d, d)); diag_pos = np.where(il == jl)[0]
    for k in range(K):
        v = rest[k].copy(); v[diag_pos] = np.exp(v[diag_pos])
        Ls[k][il, jl] = v
    return w, means, Ls

def gmm_logpdf(X, theta, K):
    # lean: no per-component arrays retained (matters at large K)
    d = X.shape[1]
    w, means, Ls = unpack_params(theta, K, d)
    lps = np.empty((len(X), K))
    for k in range(K):
        Z = solve_triangular(Ls[k], (X - means[k]).T, lower=True)
        lps[:, k] = (-0.5*(Z**2).sum(axis=0) - np.log(np.diag(Ls[k])).sum()
                     - 0.5*d*np.log(2*np.pi) + np.log(w[k]))
    m = lps.max(axis=1)
    return m + np.log(np.exp(lps - m[:, None]).sum(axis=1))

def gmm_rvs(theta, K, n, rng, d=None):
    d = d if d is not None else load_partition()["dim"]
    w, means, Ls = unpack_params(theta, K, d)
    comp = rng.choice(K, size=n, p=w); z = rng.standard_normal((n, d))
    out = np.empty((n, d))
    for k in range(K):
        m = comp == k; out[m] = means[k] + z[m] @ Ls[k].T
    return out

def fit_gmm(X, K, seed=0, n_init=3):
    gm = GaussianMixture(n_components=K, covariance_type="full", n_init=n_init,
                         reg_covar=1e-6, max_iter=500, random_state=seed).fit(X)
    return pack_params(gm.weights_, gm.means_, gm.covariances_)

# ---------------- ANALYTIC scores ----------------------------------------------
def scores(X, theta, K):
    """Per-sample d log p / d theta, closed form, (N, P).
    Memory-lean: one transient (d, N) block per component; peak ~ S + lps."""
    d = X.shape[1]; N = len(X)
    il, jl = np.tril_indices(d); nch = len(il)
    diag_pos = np.where(il == jl)[0]
    w, means, Ls = unpack_params(theta, K, d)
    P = n_params(K, d)
    S = np.empty((N, P))
    lps = np.empty((N, K))
    col_m, col_c = K - 1, K - 1 + K*d
    for k in range(K):
        Z = solve_triangular(Ls[k], (X - means[k]).T, lower=True)
        Lit = solve_triangular(Ls[k], np.eye(d), lower=True).T
        lps[:, k] = (-0.5*(Z**2).sum(axis=0) - np.log(np.diag(Ls[k])).sum()
                     - 0.5*d*np.log(2*np.pi) + np.log(w[k]))
        Q = Lit @ Z
        S[:, col_m + k*d: col_m + (k+1)*d] = Q.T
        G = Q[il, :]*Z[jl, :] - Lit[il, jl][:, None]
        G[diag_pos, :] *= np.diag(Ls[k])[:, None]
        S[:, col_c + k*nch: col_c + (k+1)*nch] = G.T
    m = lps.max(axis=1)
    lp = m + np.log(np.exp(lps - m[:, None]).sum(axis=1))
    R = np.exp(lps - lp[:, None])
    S[:, :K-1] = R[:, 1:] - w[1:][None, :]
    for k in range(K):
        S[:, col_m + k*d: col_m + (k+1)*d] *= R[:, k][:, None]
        S[:, col_c + k*nch: col_c + (k+1)*nch] *= R[:, k][:, None]
    return S

def scores_fd(X, theta, K, eps=1e-5):
    """Finite-difference scores (validation only)."""
    P = len(theta); out = np.empty((len(X), P))
    for p in range(P):
        e = eps*max(1.0, abs(theta[p]))
        tp, tm = theta.copy(), theta.copy(); tp[p] += e; tm[p] -= e
        out[:, p] = (gmm_logpdf(X, tp, K) - gmm_logpdf(X, tm, K))/(2*e)
    return out

# ---------------- tangent machinery ---------------------------------------------
def feat_multi(X, centers, sigs):
    cols = []
    Jh = len(centers)//len(sigs)
    for s_i, sig in enumerate(sigs):
        cs = centers[s_i*Jh:(s_i+1)*Jh]
        d2 = ((X[:, None, :]-cs[None, :, :])**2).sum(-1)
        cols.append(np.exp(-0.5*d2/sig**2))
    return np.hstack(cols)

def logZhat(s_R):
    m = s_R.max(); u = np.exp(s_R - m)
    return m + np.log(u.mean()), u

def build_tangent(theta, K, N_fit, X_dens, r0_seed=777):
    """Fisher/sandwich covariance, drop-capped whitening, fixed dictionary.
    R0 is regenerated deterministically from r0_seed (not stored on disk)."""
    d = X_dens.shape[1]
    R0 = gmm_rvs(theta, K, C.S_REF, np.random.default_rng(r0_seed), d=d)
    P = n_params(K, d); BLK = 20_000
    def opg(X):                     # blockwise sum psi psi^T (peak: one score block)
        M_ = np.zeros((P, P))
        for i0 in range(0, len(X), BLK):
            Ps = scores(X[i0:i0+BLK], theta, K)
            M_ += Ps.T @ Ps
        return M_ / len(X)
    F1 = opg(R0)
    jit = 1e-10*np.trace(F1)*np.eye(P)
    if C.COV_MODE == "sandwich":
        Bmat = opg(X_dens)
        Ainv = np.linalg.pinv(F1 + jit)
        cov = Ainv @ Bmat @ Ainv / N_fit
        del Bmat, Ainv
    else:
        cov = np.linalg.pinv(N_fit*(F1 + jit))
    lam_all, V_all = np.linalg.eigh(cov)
    order = np.argsort(lam_all)[::-1]
    lam_desc = np.maximum(lam_all[order], 0.0); V_desc = V_all[:, order]
    keep = (lam_desc > 0) & (lam_desc <= C.LAM_MAX)
    n_dropped = int((lam_desc > C.LAM_MAX).sum())
    lam = lam_desc[keep]; V = V_desc[:, keep]
    if C.M_EIG not in (None, "all"):
        lam = lam[:C.M_EIG]; V = V[:, :C.M_EIG]
    W = V*np.sqrt(np.maximum(lam, 1e-30))[None, :]
    Phi_R = np.empty((len(R0), W.shape[1]))
    for i0 in range(0, len(R0), BLK):        # second pass over cheap analytic scores
        Phi_R[i0:i0+BLK] = scores(R0[i0:i0+BLK], theta, K) @ W
    # fixed multi-scale dictionary from the model
    rng0 = np.random.default_rng(r0_seed)
    sub = R0[:min(len(R0), 20_000)]
    dsub = sub[rng0.choice(len(sub), size=1000, replace=False)]
    q50 = np.percentile(np.sqrt(((dsub[:, None, :]-dsub[None, :, :])**2).sum(-1))[
        np.triu_indices(len(dsub), 1)], 50)
    sigs = [f*q50 for f in C.SCALE_FRACS]
    centers = KMeans(n_clusters=C.J_CENTERS, n_init=10, random_state=0).fit(sub).cluster_centers_
    Kr = feat_multi(R0, centers, sigs)
    mu_k, sd_k = Kr.mean(axis=0), Kr.std(axis=0) + 1e-12
    Kstd_R = (Kr - mu_k)/sd_k
    sup = np.abs(Kstd_R).max(axis=0)
    bnd = C.ALPHA_CLIP/sup if C.ALPHA_CLIP else np.full(Kr.shape[1], np.inf)
    return dict(theta=theta, K=K, N_fit=N_fit, r0_seed=r0_seed, W=W, lam=lam,
                lam_all=lam_desc, cov=cov, n_dropped=n_dropped,
                centers=centers, sigs=np.array(sigs), mu_k=mu_k, sd_k=sd_k, bnd=bnd,
                R0=R0, Phi_R=Phi_R, Kstd_R=Kstd_R)

def rehydrate_tangent(art_entry, n_test=None):
    """Rebuild the heavy per-model arrays (R0, Phi_R, Kstd_R) from a slim artifact.
    The Z-hat evaluation bank size adapts to the working point: S_eval = S_EVAL_FACTOR
    x n_test (floored at 1000, capped at S_EVAL_MAX), pinning the Z-noise contribution
    to the statistic at ~ 1/S_EVAL_FACTOR dof for every N_test. The Fisher matrix and
    the dictionary (centers, standardization, bounds) come from prep and are NOT
    affected: they define the statistic and stay identical across working points."""
    t = dict(art_entry)
    S_eval = C.S_REF if n_test is None else int(
        min(max(C.S_EVAL_FACTOR * int(n_test), 1_000), C.S_EVAL_MAX))
    d = load_partition()["dim"]
    R0 = gmm_rvs(t["theta"], t["K"], S_eval if n_test is not None else C.S_REF, np.random.default_rng(int(t["r0_seed"])), d=d)
    t["R0"] = R0
    W = t["W"]; BLK = 20_000
    Phi_R = np.empty((len(R0), W.shape[1]))
    for i0 in range(0, len(R0), BLK):
        Phi_R[i0:i0+BLK] = scores(R0[i0:i0+BLK], t["theta"], t["K"]) @ W
    t["Phi_R"] = Phi_R
    t["Kstd_R"] = (feat_multi(R0, t["centers"], t["sigs"]) - t["mu_k"])/t["sd_k"]
    return t

def tan_features(X, tan, blk=20_000):
    W = tan["W"]
    Phi_X = np.empty((len(X), W.shape[1]))
    for i0 in range(0, len(X), blk):         # transient (blk x P) score blocks only
        Phi_X[i0:i0+blk] = scores(X[i0:i0+blk], tan["theta"], tan["K"]) @ W
    K_X = (feat_multi(X, tan["centers"], tan["sigs"]) - tan["mu_k"])/tan["sd_k"]
    return Phi_X, K_X

def sample_theta_tan(tan, rng):
    return tan["theta"] + tan["W"]@rng.standard_normal(tan["W"].shape[1])

# ---------------- convex tests ----------------------------------------------------
def t_point_tan(X, tan):
    _, K_X = tan_features(X, tan)
    K_R = tan["Kstd_R"]; Nt = len(X)
    def negf(a):
        lZ, u = logZhat(K_R@a)
        return (-(K_X@a).sum() + Nt*lZ + C.RIDGE_A*(a@a),
                -K_X.sum(axis=0) + Nt*(K_R.T@u)/u.sum() + 2*C.RIDGE_A*a)
    r = minimize(negf, np.zeros(K_X.shape[1]), jac=True, method="L-BFGS-B",
                 bounds=[(-b, b) for b in tan["bnd"]], options=dict(maxiter=300))
    return max(-2*r.fun, 0.0)

def t_comp_tan(X, tan):
    Phi_X, K_X = tan_features(X, tan)
    Phi_R, K_R = tan["Phi_R"], tan["Kstd_R"]
    Nt = len(X); M = Phi_X.shape[1]; J = K_X.shape[1]
    def negden(v):
        lZ, u = logZhat(Phi_R@v)
        return (-(Phi_X@v).sum() + Nt*lZ + 0.5*(v@v),
                -Phi_X.sum(axis=0) + Nt*(Phi_R.T@u)/u.sum() + v)
    rd = minimize(negden, np.zeros(M), jac=True, method="L-BFGS-B",
                  options=dict(maxiter=300))
    B_X, B_R = np.hstack([Phi_X, K_X]), np.hstack([Phi_R, K_R])
    pen = np.concatenate([np.full(M, 0.5), np.full(J, C.RIDGE_A)])
    def negnum(p):
        lZ, u = logZhat(B_R@p)
        return (-(B_X@p).sum() + Nt*lZ + (pen*p*p).sum(),
                -B_X.sum(axis=0) + Nt*(B_R.T@u)/u.sum() + 2*pen*p)
    rn = minimize(negnum, np.concatenate([rd.x, np.zeros(J)]), jac=True,
                  method="L-BFGS-B",
                  bounds=[(None, None)]*M + [(-b, b) for b in tan["bnd"]],
                  options=dict(maxiter=300))
    return max(2*(-rn.fun - (-rd.fun)), 0.0)
