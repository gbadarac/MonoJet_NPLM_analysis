"""Prep job (run ONCE before workers): truth samples, GMM fits (all K),
tangent construction, analytic-score validation. Saves a SLIM artifacts file
(R0 and feature banks are regenerated deterministically by each worker)."""
import os, time
import numpy as np
import gof2d as G
import config as C

os.makedirs(C.ART_DIR, exist_ok=True)
X_wfit = G.sample_true(C.N_WFIT, G.rng_for("wfit"))
X_dens_all = {N: G.sample_true(N, G.rng_for(f"dens:{N}")) for N in C.N_FIT_LIST}
print(f"2D toy: wfit={len(X_wfit)}  dens={[len(v) for v in X_dens_all.values()]}")

# Shared fit cache ACROSS run directories: the EM fit depends only on
# (data, base_seed, n_wfit, N, K) - not on any test parameter (LAM_MAX, RIDGE_A,
# ALPHA_CLIP, cov_mode, ...). Variant runs therefore reuse bit-identical models.
FIT_CACHE = "fits_cache"; os.makedirs(FIT_CACHE, exist_ok=True)
def _fit_key(N, K):
    import hashlib
    h = hashlib.sha256(f"2dtoy|{C.BASE_SEED}|{C.N_WFIT}|em{C.EM_TOL:g}-{C.EM_MAX_ITER}-{C.EM_N_INIT}".encode()).hexdigest()[:10]
    return os.path.join(FIT_CACHE, f"fit_{h}_N{N}_K{K}.npz")

models, val_nll = {}, {}
for N in C.N_FIT_LIST:
    for K in C.K_LIST:
        t0 = time.time(); fk = _fit_key(N, K)
        if os.path.exists(fk):
            z = np.load(fk)
            models[(N, K)] = z["theta"]; val_nll[(N, K)] = float(z["val_nll"])
            tag = "(cached)"
        else:
            th = G.fit_gmm(X_dens_all[N], K, seed=K)
            models[(N, K)] = th
            val_nll[(N, K)] = float(-G.gmm_logpdf(X_wfit, th, K).mean())
            np.savez(fk, theta=th, val_nll=val_nll[(N, K)])
            tag = ""
        print(f"fit N={N:>7d} K={K:>2d}  val NLL/pt {val_nll[(N,K)]:.4f}"
              f"  ({time.time()-t0:.0f}s) {tag}", flush=True)
K_BEST = {N: min(C.K_LIST, key=lambda K: val_nll[(N, K)]) for N in C.N_FIT_LIST}
print("K_best (INFORMATIONAL only - the pipeline runs all comp_k_list):", K_BEST)

# validate analytic scores against finite differences (once, on a small sample)
N0 = C.N_FIT_LIST[0]; K0 = C.K_LIST[0]
Xs = X_dens_all[N0][:200]
sa = G.scores(Xs, models[(N0, K0)], K0)
sf = G.scores_fd(Xs, models[(N0, K0)], K0)
err = np.abs(sa - sf).max() / max(np.abs(sf).max(), 1e-12)
print(f"analytic-vs-FD score check: max rel err = {err:.2e}")
assert err < 1e-4, "analytic scores disagree with finite differences!"

tans_slim = {}
for (N, K), th in models.items():
    t0 = time.time()
    tan = G.build_tangent(th, K, N, X_dens_all[N], r0_seed=777)
    tans_slim[(N, K)] = {k: tan[k] for k in
                         ["theta", "K", "N_fit", "r0_seed", "W", "lam", "lam_all",
                          "cov", "n_dropped", "centers", "sigs", "mu_k", "sd_k", "bnd"]}
    print(f"tangent N={N:>7d} K={K:>2d}: P={G.n_params(K, 2)}"
          f"  M_kept={tan['W'].shape[1]}  dropped={tan['n_dropped']}"
          f"  sqrt(lam) top3: " + " ".join(f"{np.sqrt(l):.3f}" for l in tan["lam"][:3])
          + f"  ({time.time()-t0:.0f}s)", flush=True)

np.savez(C.ARTIFACTS,
         models=np.array(models, dtype=object),
         val_nll=np.array(val_nll, dtype=object),
         K_best=np.array(K_BEST, dtype=object),
         tans=np.array(tans_slim, dtype=object))
print("wrote", C.ARTIFACTS)
