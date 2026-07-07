"""
Inspection plots for the new 4D JetClass embedding, QCD-only target.

Lives in the top-level <repo>/data/ (a sibling of shared/, not inside it),
because the raw embedding is pipeline-neutral -- read by BOTH the wifi_better_basis
classifier and the kernel pipeline. It sits next to the data it inspects:
data/4d_embedding_data_JetClass/.

Context: unlike the earlier 4d_embedding_Gaia run (which used the SUM of 4
processes as the target -- a class mixture whose artifacts we chased for a
while), the next milestone is a SINGLE clean class, QCD. In this dataset QCD is
the ZJetsToNuNu_*.npz files under data/4d_embedding_data_JetClass/. Each
file holds 100k jets, all label 0, and was NOT used in any training step -- it's
all test statistics, so we can pick how much to use per step.

npz layout (per file):
  embeddings : (100000, 4) float32   <- the 4D embedding, == table[:, 1:]
  table      : (100000, 5) float32   [label, z0, z1, z2, z3]
  labels     : (100000,)   int64     all 0 (QCD)
  columns    : ['label','z0','z1','z2','z3']

Produces, under data/4d_embedding_data_JetClass/inspection_plots/:
  - qcd_marginals_vs_gaussian.png : the 4 QCD marginals with the fitted Gaussian
       reference q overlaid (q is the base measure the wifi pipeline uses -- we
       want p != q but the same support).
  - qcd_pairwise.png              : 4x4 pairwise 2D histograms (structure/correlations).
  - qcd_corner.png                : corner plot (single color) -- QCD shape only.
Also prints summary statistics (mean/std/min/max, correlation matrix).
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))  # top-level data/ dir
DATA = os.path.join(HERE, "4d_embedding_data_JetClass")
OUT = os.path.join(DATA, "inspection_plots")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.RandomState(0)
N_PLOT = 300_000          # subsample for plotting speed
N_FILES = None            # how many ZJetsToNuNu files to load (None = all 20)
FEATS = [f"emb_{i}" for i in range(4)]


def load_qcd(n_files=None):
    """Concatenate the embeddings from the ZJetsToNuNu (QCD) files."""
    files = sorted(glob.glob(os.path.join(DATA, "ZJetsToNuNu_*.npz")))
    if n_files is not None:
        files = files[:n_files]
    embs = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert np.all(z["labels"] == 0), f"{f} is not pure QCD"
        embs.append(z["embeddings"])
    x = np.concatenate(embs, axis=0)
    print(f"Loaded {len(files)} QCD files -> {x.shape[0]} jets, dim {x.shape[1]}")
    return x


def subsample(a, n):
    if a.shape[0] <= n:
        return a
    idx = RNG.choice(a.shape[0], size=n, replace=False)
    return a[idx]


def main():
    x = load_qcd(N_FILES)
    d = x.shape[1]

    # ---- summary stats on the FULL QCD sample ----
    mu = x.mean(0)
    sd = x.std(0)
    print("=" * 70)
    print("Per-feature (full QCD sample):")
    for i in range(d):
        col = x[:, i]
        print(f"  emb_{i}: mean={mu[i]:+.4f} std={sd[i]:.4f} "
              f"min={col.min():+.4f} max={col.max():+.4f}")
    corr = np.corrcoef(subsample(x, N_PLOT), rowvar=False)
    print("\nCorrelation matrix (subsampled):")
    print(np.array2string(corr, precision=3, suppress_small=True))

    xp = subsample(x, N_PLOT)

    # ============ 1) marginals vs Gaussian reference q ============
    # q = N(mu, sd) fit on full QCD (matches reference.fit_gaussian_reference)
    fig, ax = plt.subplots(1, d, figsize=(4 * d, 3.4))
    for i in range(d):
        lo, hi = xp[:, i].min(), xp[:, i].max()
        bins = np.linspace(lo, hi, 120)
        ax[i].hist(xp[:, i], bins=bins, density=True, alpha=0.6,
                   color="C0", label="QCD (data)")
        xs = np.linspace(lo, hi, 400)
        g = np.exp(-0.5 * ((xs - mu[i]) / sd[i]) ** 2) / (sd[i] * np.sqrt(2 * np.pi))
        ax[i].plot(xs, g, "r-", lw=2, label="Gaussian ref q")
        ax[i].set_title(FEATS[i])
        ax[i].set_yscale("log")
        if i == 0:
            ax[i].legend(fontsize=8)
    fig.suptitle("QCD 4D embedding: marginals vs fitted Gaussian reference q (log-y)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qcd_marginals_vs_gaussian.png"), dpi=110)
    plt.close(fig)

    # ============ 2) pairwise 2D histograms ============
    fig, ax = plt.subplots(d, d, figsize=(3 * d, 3 * d))
    for i in range(d):
        for j in range(d):
            a = ax[i, j]
            if i == j:
                a.hist(xp[:, i], bins=100, color="C0")
                a.set_yscale("log")
            else:
                a.hist2d(xp[:, j], xp[:, i], bins=120, cmap="viridis",
                         norm=matplotlib.colors.LogNorm())
            if i == d - 1:
                a.set_xlabel(FEATS[j])
            if j == 0:
                a.set_ylabel(FEATS[i])
    fig.suptitle("QCD 4D embedding: pairwise 2D histograms (log color)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qcd_pairwise.png"), dpi=100)
    plt.close(fig)

    # ============ 3) corner plot (single color) ============
    per = min(40_000, xp.shape[0])
    sc = subsample(xp, per)
    fig, ax = plt.subplots(d, d, figsize=(3.2 * d, 3.2 * d))
    for i in range(d):
        for j in range(d):
            a = ax[i, j]
            if j > i:
                a.axis("off")
                continue
            if i == j:
                lo, hi = xp[:, i].min(), xp[:, i].max()
                a.hist(xp[:, i], bins=np.linspace(lo, hi, 80),
                       density=True, histtype="step", color="C0", lw=1.4)
            else:
                a.scatter(sc[:, j], sc[:, i], s=1, alpha=0.15,
                          color="C0", linewidths=0)
            if i == d - 1:
                a.set_xlabel(FEATS[j])
            if j == 0 and i != 0:
                a.set_ylabel(FEATS[i])
    fig.suptitle("QCD 4D embedding corner plot")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qcd_corner.png"), dpi=100)
    plt.close(fig)

    print(f"\nWrote plots to {OUT}")


if __name__ == "__main__":
    main()
