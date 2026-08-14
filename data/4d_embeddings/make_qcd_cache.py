"""
Prepare a QCD (ZJetsToNuNu) JetClass embedding as a train/test cache.

Lives under <repo>/data/4d_embeddings/ (a sibling of shared/), next to BOTH the raw
embedding folders it reads and the per-run caches it writes. The train/test split
is pipeline-neutral: the SPARKER KERNEL pipeline consumes it (each Sparker member
in Train_Ensembles/.../EstimationKernels.py bootstraps from data_train.npy via
--data_path, and the LRT GoF test reads data_test.npy), and the classifier
pipeline can read the same cache.

Single-class target (QCD only) -- the milestone. Two source embeddings currently
use this (select with --data-dir):
  - 4d_embedding_data_JetClass          : original 4D SimCLR QCD embedding (--dims 4).
  - 4d_gaussian_embedding_data_JetClass : 8D LeCun+SIGReg "more gaussian" embedding;
       use the FIRST 4 DIMS (--dims 4, default) to stay comparable to the 4D run.

Source layout: a POOL of 20 npz files x 100k jets = 2M jets under
<repo>/data/4d_embeddings/<data-dir>/ZJetsToNuNu_*.npz, all label 0, none used in
any training step. The embedding is the `embeddings` key (== table[:, 1:]).

Disjoint split: because it's one pool (not pre-split), we draw train and test from
a SINGLE shuffle and slice them disjointly, so the density-fit sample (data_train)
and the GoF observed sample (data_test) share no jets (independent subsampling
would risk overlap -> leakage).

Writes a sibling cache dir under this same data/4d_embeddings/ folder (d flows
through from the array shape -- no downstream code change needed):
    <repo>/data/4d_embeddings/<benchmark>_Ntrain<Ntr>_Ntest<Nte>_seed<seed>/
        data_train.npy   (fit the density model / kernel ensemble)
        data_test.npy    (held-out target sample the GoF/LRT is run against)
        data_config.json

Usage (new gaussian embedding, first 4 dims -- the current default):
    python make_qcd_cache.py --n_train 100000 --n_test 100000 \
        --benchmark 4d_gaussian_embedding_qcd --seed 42

Then point the kernel submit script (submit_SparkerKernels.sh) DATA_PATH at
    <repo>/data/4d_embeddings/4d_gaussian_embedding_qcd_Ntrain100000_Ntest100000_seed42/data_train.npy
"""

import os
import glob
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # top-level <repo>/data/
QCD_GLOB = "ZJetsToNuNu_*.npz"


def load_qcd_pool(qcd_dir, dims):
    """Concatenate the first `dims` embedding columns from all QCD npz files."""
    files = sorted(glob.glob(os.path.join(qcd_dir, QCD_GLOB)))
    if not files:
        raise FileNotFoundError(f"no {QCD_GLOB} under {qcd_dir}")
    embs = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert np.all(z["labels"] == 0), f"{f} is not pure QCD (label 0)"
        emb = z["embeddings"]
        assert emb.shape[1] >= dims, (
            f"{f} has {emb.shape[1]} dims < requested {dims}")
        embs.append(emb[:, :dims])
    pool = np.concatenate(embs, axis=0).astype(np.float64)
    print(f"loaded {len(files)} QCD files from {qcd_dir} "
          f"-> pool {pool.shape} (first {dims} of {emb.shape[1]} dims)")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_test", type=int, required=True)
    ap.add_argument("--benchmark", type=str, default="4d_gaussian_embedding_qcd")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", dest="data_dir", type=str,
                    default="4d_gaussian_embedding_data_JetClass",
                    help="raw embedding folder under this data/ dir (or an absolute "
                         "path). Default: 4d_gaussian_embedding_data_JetClass.")
    ap.add_argument("--dims", type=int, default=4,
                    help="use only the first N embedding dims (default 4).")
    args = ap.parse_args()

    qcd_dir = args.data_dir
    if not os.path.isabs(qcd_dir):
        qcd_dir = os.path.join(HERE, qcd_dir)

    pool = load_qcd_pool(qcd_dir, args.dims)
    need = args.n_train + args.n_test
    if need > pool.shape[0]:
        raise ValueError(
            f"requested n_train+n_test={need} > available {pool.shape[0]} QCD jets. "
            f"Load more files or lower the sizes."
        )

    # Single shuffle -> disjoint train / test slices (no shared jets).
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(pool.shape[0])
    X_train = pool[idx[:args.n_train]]
    X_test = pool[idx[args.n_train:args.n_train + args.n_test]]

    name = (f"{args.benchmark}_Ntrain{args.n_train}_Ntest{args.n_test}"
            f"_seed{args.seed}")
    out_dir = os.path.join(HERE, name)   # sibling cache dir under <repo>/data/
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "data_train.npy"), X_train)
    np.save(os.path.join(out_dir, "data_test.npy"), X_test)
    with open(os.path.join(out_dir, "data_config.json"), "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "N_train": args.n_train, "N_test": args.n_test,
            "seed": args.seed, "d": int(X_train.shape[1]),
            "source": f"{os.path.basename(qcd_dir.rstrip('/'))} QCD "
                      f"(ZJetsToNuNu_*.npz), first {args.dims} dims, "
                      "single-class, disjoint train/test split",
            "data_dir": out_dir,
        }, f, indent=2)

    print(f"wrote {out_dir}")
    print(f"  data_train {X_train.shape}  data_test {X_test.shape}")


if __name__ == "__main__":
    main()
