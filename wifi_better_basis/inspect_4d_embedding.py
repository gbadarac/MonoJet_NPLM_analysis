"""
Inspection plots for the 4D physics embedding shared by Gaia.

Purpose: before running the wifi_better_basis pipeline on the embedding, look at
what the data actually is and whether the pipeline's Gaussian reference q(x) is
a sane base measure for it.

Produces, under data/4d_embedding_Gaia/inspection_plots/:
  - marginals_bkg_vs_gaussian.png : the 4 embedding-feature marginals of the full
       background sample, with the fitted Gaussian reference q overlaid (q is what
       the pipeline uses as the base measure -- we want to see p != q but same support).
  - marginals_per_class.png       : the 4 marginals split by process label (0-3).
  - pairwise_bkg.png              : 4x4 pairwise 2D histograms (structure/correlations).
  - marginals_bkg_vs_signal.png   : background vs one signal (ato4l) marginals -- what a
       real discrepancy would look like.
Also prints summary statistics (mean/std/min/max, correlation matrix, %at bounds).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "4d_embedding_Gaia")
OUT = os.path.join(DATA, "inspection_plots")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.RandomState(0)
N_PLOT = 300_000          # subsample for speed
FEATS = [f"emb_{i}" for i in range(4)]


def subsample(a, n):
    if a.shape[0] <= n:
        return a
    idx = RNG.choice(a.shape[0], size=n, replace=False)
    return a[idx]


def load(name):
    return np.load(os.path.join(DATA, name))


def main():
    train = load("train_h.npy")
    test = load("test_h.npy")
    labels = load("train_labels.npy")
    d = train.shape[1]

    # ---- summary stats on the FULL train sample ----
    print("=" * 70)
    print(f"train_h: shape={train.shape} dtype={train.dtype}")
    print(f"test_h : shape={test.shape}")
    mu = train.mean(0)
    sd = train.std(0)
    print("\nPer-feature (full train background):")
    for i in range(d):
        col = train[:, i]
        frac_lo = np.mean(col <= -0.999)
        frac_hi = np.mean(col >= 0.999)
        print(f"  emb_{i}: mean={mu[i]:+.4f} std={sd[i]:.4f} "
              f"min={col.min():+.4f} max={col.max():+.4f} "
              f"frac@-1={frac_lo:.3%} frac@+1={frac_hi:.3%}")
    corr = np.corrcoef(subsample(train, N_PLOT), rowvar=False)
    print("\nCorrelation matrix (subsampled):")
    print(np.array2string(corr, precision=3, suppress_small=True))

    u, c = np.unique(labels, return_counts=True)
    print("\nProcess-label composition (train):",
          {int(k): f"{v} ({v/labels.shape[0]:.1%})" for k, v in zip(u, c)})

    # subsamples for plotting
    tr = subsample(train, N_PLOT)
    lab = labels[:train.shape[0]]
    lab_sub_idx = RNG.choice(train.shape[0], size=min(N_PLOT, train.shape[0]),
                             replace=False)
    tr_lab = train[lab_sub_idx]
    lab_sub = labels[lab_sub_idx]

    # ================= 1) marginals vs Gaussian reference q =================
    # q = N(mu, Sigma) fit on full train (matches reference.fit_gaussian_reference)
    fig, ax = plt.subplots(1, d, figsize=(4 * d, 3.4))
    for i in range(d):
        lo, hi = tr[:, i].min(), tr[:, i].max()
        bins = np.linspace(lo, hi, 120)
        ax[i].hist(tr[:, i], bins=bins, density=True, alpha=0.6,
                   color="C0", label="background (data)")
        xs = np.linspace(lo, hi, 400)
        g = np.exp(-0.5 * ((xs - mu[i]) / sd[i]) ** 2) / (sd[i] * np.sqrt(2 * np.pi))
        ax[i].plot(xs, g, "r-", lw=2, label="Gaussian ref q")
        ax[i].set_title(FEATS[i])
        ax[i].set_yscale("log")
        if i == 0:
            ax[i].legend(fontsize=8)
    fig.suptitle("4D embedding: background marginals vs fitted Gaussian reference q "
                 "(log-y)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "marginals_bkg_vs_gaussian.png"), dpi=110)
    plt.close(fig)

    # ================= 2) marginals per process label =================
    fig, ax = plt.subplots(1, d, figsize=(4 * d, 3.4))
    for i in range(d):
        lo, hi = tr[:, i].min(), tr[:, i].max()
        bins = np.linspace(lo, hi, 120)
        for cls in u:
            m = lab_sub == cls
            ax[i].hist(tr_lab[m, i], bins=bins, density=True, histtype="step",
                       lw=1.5, label=f"class {int(cls)}")
        ax[i].set_title(FEATS[i])
        ax[i].set_yscale("log")
        if i == 0:
            ax[i].legend(fontsize=8)
    fig.suptitle("4D embedding: marginals split by process label")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "marginals_per_class.png"), dpi=110)
    plt.close(fig)

    # ================= 3) pairwise 2D histograms =================
    fig, ax = plt.subplots(d, d, figsize=(3 * d, 3 * d))
    for i in range(d):
        for j in range(d):
            a = ax[i, j]
            if i == j:
                a.hist(tr[:, i], bins=100, color="C0")
                a.set_yscale("log")
            else:
                a.hist2d(tr[:, j], tr[:, i], bins=120, cmap="viridis",
                         norm=matplotlib.colors.LogNorm())
            if i == d - 1:
                a.set_xlabel(FEATS[j])
            if j == 0:
                a.set_ylabel(FEATS[i])
    fig.suptitle("4D embedding background: pairwise 2D histograms (log color)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pairwise_bkg.png"), dpi=100)
    plt.close(fig)

    # ================= 3b) CORNER PLOT colored by process label =================
    # Replicates Gaia's reference figure so we can compare SHAPES directly and
    # confirm the [-1,1] files are the same embedding as her [-20,20] picture.
    colors = ["C0", "C1", "C2", "C3"]
    per_cls = 40_000  # scatter points per class
    fig, ax = plt.subplots(d, d, figsize=(3.2 * d, 3.2 * d))
    for i in range(d):
        for j in range(d):
            a = ax[i, j]
            if j > i:
                a.axis("off")
                continue
            if i == j:
                for ci, cls in enumerate(u):
                    m = lab_sub == cls
                    lo, hi = tr_lab[:, i].min(), tr_lab[:, i].max()
                    a.hist(tr_lab[m, i], bins=np.linspace(lo, hi, 80),
                           density=True, histtype="step", color=colors[ci],
                           lw=1.4, label=f"{int(cls)}")
                if i == 0:
                    a.legend(fontsize=8, ncol=2, title="process")
            else:
                for ci, cls in enumerate(u):
                    m = np.where(lab_sub == cls)[0]
                    if m.shape[0] > per_cls:
                        m = RNG.choice(m, per_cls, replace=False)
                    a.scatter(tr_lab[m, j], tr_lab[m, i], s=1, alpha=0.15,
                              color=colors[ci], linewidths=0)
            if i == d - 1:
                a.set_xlabel(FEATS[j])
            if j == 0 and i != 0:
                a.set_ylabel(FEATS[i])
    fig.suptitle("4D embedding corner plot, colored by process label "
                 "(compare shape to Gaia's reference)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "corner_by_class.png"), dpi=100)
    plt.close(fig)

    # ================= 4) background vs a signal =================
    sig_name = "signal_ato4l_h.npy"
    if os.path.exists(os.path.join(DATA, sig_name)):
        sig = subsample(load(sig_name), N_PLOT)
        fig, ax = plt.subplots(1, d, figsize=(4 * d, 3.4))
        for i in range(d):
            lo = min(tr[:, i].min(), sig[:, i].min())
            hi = max(tr[:, i].max(), sig[:, i].max())
            bins = np.linspace(lo, hi, 120)
            ax[i].hist(tr[:, i], bins=bins, density=True, histtype="step",
                       lw=1.8, color="C0", label="background")
            ax[i].hist(sig[:, i], bins=bins, density=True, histtype="step",
                       lw=1.8, color="C3", label="signal ato4l")
            ax[i].set_title(FEATS[i])
            ax[i].set_yscale("log")
            if i == 0:
                ax[i].legend(fontsize=8)
        fig.suptitle("4D embedding: background vs signal (ato4l) marginals")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "marginals_bkg_vs_signal.png"), dpi=110)
        plt.close(fig)

    print(f"\nWrote plots to {OUT}")


if __name__ == "__main__":
    main()
