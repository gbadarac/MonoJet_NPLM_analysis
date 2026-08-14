"""
Inspection plots for a JetClass QCD embedding, sliced to its first N dimensions.

Lives under <repo>/data/4d_embeddings/ (a sibling of shared/, not inside it),
because the raw embedding is pipeline-neutral -- read by BOTH the wifi_better_basis
classifier and the kernel pipeline. It sits next to the data it inspects.

This script is dataset-agnostic: point it at any folder of ZJetsToNuNu_*.npz
files (all label 0 = QCD) with the layout below and pick how many leading
embedding dims to inspect. Two datasets currently use it (select with --data-dir):
  - 4d_embedding_data_JetClass          : the original 4D JetClass QCD embedding.
  - 4d_gaussian_embedding_data_JetClass : an 8D embedding with a LeCun transform +
       per-class SIGReg gaussianity (Christina). We focus on the FIRST 4 DIMS
       (--dims 4, the default) to keep it comparable to the 4D run.

Context: unlike the earlier 4d_embedding_Gaia run (which used the SUM of 4
processes as the target -- a class mixture whose artifacts we chased for a
while), the milestone is a SINGLE clean class, QCD (the ZJetsToNuNu_*.npz files).
Each file holds 100k jets, all label 0, and was NOT used in any training step --
it's all test statistics, so we can pick how much to use per step.

npz layout (per file):
  embeddings : (100000, D) float32   <- the D-dim embedding, == table[:, 1:]
  table      : (100000, D+1) float32 [label, z0, ..., z{D-1}]
  labels     : (100000,)   int64     all 0 (QCD)
  columns    : ['label','z0',...,'z{D-1}']

Produces, under <data-dir>/inspection_plots/:
  - qcd_pairwise.png : pairwise 2D histograms (structure/correlations).
  - qcd_corner.png   : corner plot (single color) -- QCD shape only.
Also prints summary statistics (mean/std/min/max, correlation matrix).
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))  # <repo>/data/4d_embeddings/

RNG = np.random.RandomState(0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="4d_gaussian_embedding_data_JetClass",
                   help="folder of ZJetsToNuNu_*.npz (relative to data/4d_embeddings/, "
                        "or absolute). Default: 4d_gaussian_embedding_data_JetClass.")
    p.add_argument("--dims", type=int, default=4,
                   help="use only the first N embedding dims (default 4).")
    p.add_argument("--n-files", type=int, default=None,
                   help="how many ZJetsToNuNu files to load (default: all).")
    p.add_argument("--n-plot", type=int, default=300_000,
                   help="subsample size for plotting speed (default 300k).")
    return p.parse_args()


def load_qcd(data_dir, dims, n_files=None):
    """Concatenate the first `dims` embedding columns from the QCD files."""
    files = sorted(glob.glob(os.path.join(data_dir, "ZJetsToNuNu_*.npz")))
    if not files:
        raise FileNotFoundError(f"No ZJetsToNuNu_*.npz under {data_dir}")
    if n_files is not None:
        files = files[:n_files]
    embs = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert np.all(z["labels"] == 0), f"{f} is not pure QCD"
        emb = z["embeddings"]
        assert emb.shape[1] >= dims, (
            f"{f} has {emb.shape[1]} dims < requested {dims}")
        embs.append(emb[:, :dims])
    x = np.concatenate(embs, axis=0)
    print(f"Loaded {len(files)} QCD files -> {x.shape[0]} jets, "
          f"using first {x.shape[1]} of {emb.shape[1]} dims")
    return x


def subsample(a, n):
    if a.shape[0] <= n:
        return a
    idx = RNG.choice(a.shape[0], size=n, replace=False)
    return a[idx]


def main():
    args = parse_args()
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(HERE, data_dir)
    out = os.path.join(data_dir, "inspection_plots")
    os.makedirs(out, exist_ok=True)

    x = load_qcd(data_dir, args.dims, args.n_files)
    d = x.shape[1]
    feats = [f"emb_{i}" for i in range(d)]
    n_plot = args.n_plot

    # ---- summary stats on the FULL QCD sample ----
    mu = x.mean(0)
    sd = x.std(0)
    print("=" * 70)
    print("Per-feature (full QCD sample):")
    for i in range(d):
        col = x[:, i]
        print(f"  emb_{i}: mean={mu[i]:+.4f} std={sd[i]:.4f} "
              f"min={col.min():+.4f} max={col.max():+.4f}")
    corr = np.corrcoef(subsample(x, n_plot), rowvar=False)
    print("\nCorrelation matrix (subsampled):")
    print(np.array2string(corr, precision=3, suppress_small=True))

    xp = subsample(x, n_plot)

    # ============ 1) pairwise 2D histograms ============
    fig, ax = plt.subplots(d, d, figsize=(3 * d, 3 * d))
    ax = np.atleast_2d(ax)
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
                a.set_xlabel(feats[j])
            if j == 0:
                a.set_ylabel(feats[i])
    fig.suptitle(f"QCD {d}D embedding: pairwise 2D histograms (log color)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "qcd_pairwise.png"), dpi=100)
    plt.close(fig)

    # ============ 2) corner plot (single color) ============
    per = min(40_000, xp.shape[0])
    sc = subsample(xp, per)
    fig, ax = plt.subplots(d, d, figsize=(3.2 * d, 3.2 * d))
    ax = np.atleast_2d(ax)
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
                a.set_xlabel(feats[j])
            if j == 0 and i != 0:
                a.set_ylabel(feats[i])
    fig.suptitle(f"QCD {d}D embedding corner plot")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "qcd_corner.png"), dpi=100)
    plt.close(fig)

    print(f"\nWrote plots to {out}")


if __name__ == "__main__":
    main()
