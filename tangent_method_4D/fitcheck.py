"""Model fit checks + tangent-budget diagnostics (figures only; no worker results
needed - runs any time after prep). One figure pair per (N_fit, K_best); use
--all-k for every fitted (N_fit, K).

Outputs (in <run>/figs/):
  fit_result_N<N>_K<K>.pdf     analytic model marginals + pairwise contours vs
                               held-out data, with the kernel dictionary overlaid
  fit_diagnostic_N<N>_K<K>.pdf budget maps on held-out points (sd[log p] used,
                               variance fraction covered, linearization ratio)
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import gof4d as G
import config as C
from paperstyle import INK, GRID, MUTED, BLUE, ORANGE, style_ax, save_fig

ap = argparse.ArgumentParser()
ap.add_argument("--n-eval", type=int, default=6000)
ap.add_argument("--n-draws", type=int, default=30)
args = ap.parse_args()

os.makedirs(C.FIG_DIR, exist_ok=True)
art = np.load(C.ARTIFACTS, allow_pickle=True)
K_BEST = art["K_best"].item()
tans_slim = art["tans"].item()
import glob as _glob
if not _glob.glob(C.DATA_GLOB):
    raise SystemExit(
        f"[fitcheck] SKIPPED: no files match DATA_GLOB = {C.DATA_GLOB}\n"
        "  The raw event files are not reachable from this node (fitcheck is the\n"
        "  only analyze step that needs them - merge/plots/diagnostics are fine).\n"
        "  Likely cause on clusters: this storage tier is not mounted on login\n"
        "  nodes. Run fitcheck from an interactive/compute job, e.g.\n"
        "    salloc -c 4 --mem 16G -t 1:00:00\n"
        "    GOF4D_PARAMS=<run>/params.json python fitcheck.py")
part = G.load_partition()
DIM = part["dim"]
PAIRS = [(a, b) for a in range(DIM) for b in range(a+1, DIM)][:4]

def gmm_marginal_1d(theta, K, i, grid):
    w, means, Ls = G.unpack_params(theta, K, DIM)
    covs = np.einsum("kij,klj->kil", Ls, Ls)
    return sum(w[k]*stats.norm.pdf(grid, means[k, i], np.sqrt(covs[k, i, i]))
               for k in range(K))

def gmm_pair_density(theta, K, pair, G1, G2):
    a, b = pair
    w, means, Ls = G.unpack_params(theta, K, DIM)
    covs = np.einsum("kij,klj->kil", Ls, Ls)
    P2 = np.dstack([G1, G2]).reshape(-1, 2)
    Z = np.zeros(len(P2))
    for k in range(K):
        Z += w[k]*stats.multivariate_normal(
            means[k, [a, b]], covs[k][np.ix_([a, b], [a, b])]).pdf(P2)
    return Z.reshape(G1.shape)

def fit_check(N, K, tan):
    th = tan["theta"]
    Xd = part["TEST_POOL"][:200_000]
    ncol = max(len(PAIRS), DIM)
    fig, ax = plt.subplots(2, ncol, figsize=(3.4*ncol, 6.4))
    Jh = len(tan["centers"]) // len(tan["sigs"])
    for j, (a, b) in enumerate(PAIRS):
        lo1, hi1 = np.percentile(Xd[:, a], [0.2, 99.8])
        lo2, hi2 = np.percentile(Xd[:, b], [0.2, 99.8])
        G1, G2 = np.meshgrid(np.linspace(lo1, hi1, 110), np.linspace(lo2, hi2, 110))
        ax[0][j].contourf(G1, G2, gmm_pair_density(th, K, (a, b), G1, G2), 30)
        ax[0][j].scatter(Xd[::400, a], Xd[::400, b], s=2, alpha=.25, color="w")
        for s_i, (sig, mk) in enumerate(zip(tan["sigs"], ["+", "x", "o", "s"])):
            cs = tan["centers"][s_i*Jh:(s_i+1)*Jh]
            ax[0][j].scatter(cs[:, a], cs[:, b], s=40, c="red", marker=mk,
                             label=f"sig={sig:.3f}" if j == 0 else None)
        ax[0][j].set(xlabel=f"x{a}", ylabel=f"x{b}", title=f"pair ({a},{b})")
    ax[0][0].legend(fontsize=7)
    for j in range(len(PAIRS), ncol): ax[0][j].axis("off")
    for i in range(DIM):
        lo, hi = np.percentile(Xd[:, i], [0.2, 99.8])
        grid = np.linspace(lo, hi, 300)
        ax[1][i].hist(Xd[:, i], bins=80, range=(lo, hi), density=True, alpha=.4,
                      color="gray", label="data (held out)")
        ax[1][i].plot(grid, gmm_marginal_1d(th, K, i, grid), "C3--", lw=1.6, label="model")
        ax[1][i].set(title=f"marginal x{i}", xlabel=f"x{i}"); ax[1][i].legend(fontsize=8)
    for i in range(DIM, ncol): ax[1][i].axis("off")
    fig.suptitle(f"fit check | N_fit={N}, K={K}", fontsize=12)
    plt.tight_layout()
    f = os.path.join(C.FIG_DIR, f"fit_result_N{N}_K{K}.pdf")
    plt.savefig(f); plt.close(fig); print(f)

def budget_diagnostic(N, K, tan):
    th = tan["theta"]
    rE = np.random.default_rng(500 + N)
    Xe = part["TEST_POOL"][rE.choice(len(part["TEST_POOL"]), args.n_eval, replace=False)]
    Psi_E = G.scores(Xe, th, K)
    sig2_full = np.einsum("ip,pq,iq->i", Psi_E, tan["cov"], Psi_E)
    sig2_used = ((Psi_E @ tan["W"])**2).sum(axis=1)
    lp0 = G.gmm_logpdf(Xe, th, K)
    draws = np.array([G.gmm_logpdf(Xe, G.sample_theta_tan(tan,
                        np.random.default_rng(1000*N + i)), K) - lp0
                      for i in range(args.n_draws)])
    sig2_exact = draws.var(axis=0)
    frac = np.where(sig2_full > 1e-12, sig2_used/np.maximum(sig2_full, 1e-12), 1.0)
    ratio = np.where(sig2_used > 1e-12, sig2_exact/np.maximum(sig2_used, 1e-12), 1.0)
    rows = [(np.sqrt(sig2_used), "sd[log p] budget used", None,
             (0, np.percentile(np.sqrt(sig2_used), 99))),
            (np.clip(frac, 0, 1), "variance fraction covered", "viridis", (0, 1)),
            (np.clip(ratio, 0, 2), "linearization: exact/tangent var", "coolwarm", (0, 2))]
    fig, ax = plt.subplots(3, len(PAIRS), figsize=(3.6*len(PAIRS), 9.6), squeeze=False)
    for r, (val, title, cmap, (v0, v1)) in enumerate(rows):
        for j, (a, b) in enumerate(PAIRS):
            sc = ax[r][j].scatter(Xe[:, a], Xe[:, b], c=val, s=4, alpha=.7,
                                  cmap=cmap, vmin=v0, vmax=v1)
            ax[r][j].set(xlabel=f"x{a}", ylabel=f"x{b}")
            if j == 0:
                ax[r][j].set_title(f"{title} (N={N}, K={K})", fontsize=9, loc="left")
        plt.colorbar(sc, ax=ax[r][-1])
    plt.tight_layout()
    f = os.path.join(C.FIG_DIR, f"fit_diagnostic_N{N}_K{K}.pdf")
    plt.savefig(f); plt.close(fig)
    M = tan["W"].shape[1]
    print(f"{f}  | covered median {np.median(frac):.1%} | lin ratio median "
          f"{np.median(ratio):.2f} | E[sig2_used]={sig2_used.mean():.4f} vs M/N={M/N:.4f}")

todo = sorted(tans_slim)          # no model selection: figures for every (N_fit, K)
for (N, K) in todo:
    tan = G.rehydrate_tangent(tans_slim[(N, K)])
    fit_check(N, K, tan)
    budget_diagnostic(N, K, tan)
