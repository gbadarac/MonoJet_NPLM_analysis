"""Model fit checks + tangent-budget diagnostics for the 2D toy (figures only;
runs any time after prep - the truth is analytic, no data files needed).

Outputs (in <run>/figs/): fit_result_N<N>_K<K>.{png,pdf}, fit_diagnostic_N<N>_K<K>.{png,pdf}
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gof2d as G
import config as C
from paperstyle import INK, GRID, MUTED, BLUE, ORANGE, style_ax, save_fig

ap = argparse.ArgumentParser()
ap.add_argument("--n-draws", type=int, default=30)
args = ap.parse_args()

os.makedirs(C.FIG_DIR, exist_ok=True)
art = np.load(C.ARTIFACTS, allow_pickle=True)
K_BEST = art["K_best"].item()
tans_slim = art["tans"].item()

g1 = np.linspace(-1.2, 0.4, 140); g2 = np.linspace(-0.6, 3.6, 140)
G1, G2 = np.meshgrid(g1, g2)
PT = np.column_stack([G1.ravel(), G2.ravel()])
d1, d2 = g1[1]-g1[0], g2[1]-g2[0]

def fit_check(N, K, tan):
    th = tan["theta"]
    Z = np.exp(G.gmm_logpdf(PT, th, K)).reshape(G1.shape)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].contourf(G1, G2, Z, 30)
    Jh = len(tan["centers"]) // len(tan["sigs"])
    for s_i, (sig, mk) in enumerate(zip(tan["sigs"], ["+", "x", "o", "s"])):
        cs = tan["centers"][s_i*Jh:(s_i+1)*Jh]
        ax[0].scatter(cs[:, 0], cs[:, 1], s=40, c="red", marker=mk,
                      label=f"kernels sig={sig:.3f}")
    ax[0].set(title=f"fitted GMM (K={K}, N_fit={N})", xlabel="x1", ylabel="x2")
    ax[0].legend(frameon=False, fontsize=7, labelcolor=INK)
    marg = {0: Z.sum(axis=0)*d2, 1: Z.sum(axis=1)*d1}
    for a, axis, name, grid in [(ax[1], 0, "x1", g1), (ax[2], 1, "x2", g2)]:
        a.plot(grid, G.true_marginal(axis, grid), color=INK, lw=1.6, label="truth")
        a.plot(grid, marg[axis], color=ORANGE, ls="--", lw=1.6, label="model")
        a.set(title=f"marginal {name}", xlabel=name)
        a.legend(frameon=False, fontsize=8, labelcolor=INK)
    for a in ax: style_ax(a)
    plt.tight_layout()
    save_fig(fig, C.FIG_DIR, f"fit_result_N{N}_K{K}")
    plt.close(fig)

def budget_diagnostic(N, K, tan):
    th = tan["theta"]
    Psi_G = G.scores(PT, th, K)
    sig2_full = np.einsum("ip,pq,iq->i", Psi_G, tan["cov"], Psi_G)
    sig2_used = ((Psi_G @ tan["W"])**2).sum(axis=1)
    lp0 = G.gmm_logpdf(PT, th, K)
    draws = np.array([G.gmm_logpdf(PT, G.sample_theta_tan(tan,
                        np.random.default_rng(1000*N + i)), K) - lp0
                      for i in range(args.n_draws)])
    sig2_exact = draws.var(axis=0)
    frac = np.where(sig2_full > 1e-12, sig2_used/np.maximum(sig2_full, 1e-12), 1.0)
    ratio = np.where(sig2_used > 1e-12, sig2_exact/np.maximum(sig2_used, 1e-12), 1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    im0 = ax[0].contourf(G1, G2, np.sqrt(sig2_used).reshape(G1.shape), 30)
    plt.colorbar(im0, ax=ax[0]); ax[0].set_title(f"sd[log p] budget used (N={N}, K={K})", fontsize=9)
    im1 = ax[1].contourf(G1, G2, np.clip(frac, 0, 1).reshape(G1.shape),
                         levels=np.linspace(0, 1, 31), cmap="viridis")
    plt.colorbar(im1, ax=ax[1]); ax[1].set_title("variance fraction covered", fontsize=9)
    im2 = ax[2].contourf(G1, G2, np.clip(ratio, 0, 2).reshape(G1.shape),
                         levels=np.linspace(0, 2, 31), cmap="coolwarm")
    plt.colorbar(im2, ax=ax[2]); ax[2].set_title("linearization: exact/tangent var", fontsize=9)
    for a in ax:
        a.set(xlabel="x1", ylabel="x2"); style_ax(a)
    plt.tight_layout()
    save_fig(fig, C.FIG_DIR, f"fit_diagnostic_N{N}_K{K}")
    plt.close(fig)
    M = tan["W"].shape[1]
    print(f"N={N} K={K}: covered median {np.median(frac):.1%} | "
          f"lin ratio median {np.median(ratio):.2f} | "
          f"E-ish[sig2_used] grid mean={sig2_used.mean():.4f} (M/N={M/N:.4f})")

todo = sorted(tans_slim)          # no model selection: figures for every (N_fit, K)
for (N, K) in todo:
    # slim artifact suffices: fitcheck never touches the Z-hat bank arrays.
    tan = dict(tans_slim[(N, K)])
    fit_check(N, K, tan)
    budget_diagnostic(N, K, tan)
