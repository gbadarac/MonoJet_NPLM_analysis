"""
Prepare the QCD (ZJetsToNuNu) 4D JetClass embedding as a pipeline data cache.

Single-class target (QCD only) -- the milestone. This differs from the earlier
make_4d_cache.py (old Gaia embedding: mixture background, with pre-split
train_h.npy / test_h.npy files) in two ways:

  1. Source layout. The JetClass QCD data is a POOL of 20 npz files x 100k jets
     = 2M jets under <repo>/data/4d_embedding_data_JetClass/
     ZJetsToNuNu_*.npz, all label 0, none used in any training step. The 4D
     embedding is the `embeddings` key (== table[:, 1:]). The raw data is shared
     with the kernel pipeline; only the per-run cache is written locally here.

  2. Disjoint split. Because it's one pool (not pre-split), we draw the train
     and test subsamples from a SINGLE shuffle and slice them disjointly, so the
     density-fit sample (data_train) and the GoF observed sample (data_test)
     share no jets. Independent subsampling would risk overlap -> leakage.

Writes, exactly as run.py/run_gof.py already expect (d=4 flows through from the
array shape -- no pipeline code change needed):
    data/<benchmark>_Ntrain<Ntr>_Ntest<Nte>_seed<seed>/
        data_train.npy   (fit the density model: basis + wifi weights)
        data_test.npy    (held-out target sample the GoF is run against)
        data_config.json

Usage:
    python make_qcd_cache.py --n_train 100000 --n_test 100000 \
        --benchmark 4d_embedding_qcd --seed 42

Then set in config.py: benchmark="4d_embedding_qcd", N_train, N_test, seed to
match, and run WITHOUT the coverage step (coverage needs an analytic truth real
data does not have):
    python submit_slurm.py --steps run gof plot_gof
"""

import os
import glob
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# Raw QCD embedding lives OUTSIDE this pipeline, in the top-level shared data dir
# (read by both the classifier and kernel pipelines):
#   <repo>/data/4d_embedding_data_JetClass/ZJetsToNuNu_*.npz
# Only the per-run cache (data_train/data_test) is written locally under HERE/data/.
QCD_DIR = os.path.join(REPO_ROOT, "data", "4d_embedding_data_JetClass")
QCD_GLOB = "ZJetsToNuNu_*.npz"


def load_qcd_pool():
    """Concatenate the 4D embeddings from all QCD (ZJetsToNuNu) npz files."""
    files = sorted(glob.glob(os.path.join(QCD_DIR, QCD_GLOB)))
    if not files:
        raise FileNotFoundError(f"no {QCD_GLOB} under {QCD_DIR}")
    embs = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        assert np.all(z["labels"] == 0), f"{f} is not pure QCD (label 0)"
        embs.append(z["embeddings"])
    pool = np.concatenate(embs, axis=0).astype(np.float64)
    print(f"loaded {len(files)} QCD files -> pool {pool.shape}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_test", type=int, required=True)
    ap.add_argument("--benchmark", type=str, default="4d_embedding_qcd")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_qcd_pool()
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
    out_dir = os.path.join(HERE, "data", name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "data_train.npy"), X_train)
    np.save(os.path.join(out_dir, "data_test.npy"), X_test)
    with open(os.path.join(out_dir, "data_config.json"), "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "N_train": args.n_train, "N_test": args.n_test,
            "seed": args.seed, "d": int(X_train.shape[1]),
            "source": "4d_embedding_data_JetClass QCD (ZJetsToNuNu_*.npz), "
                      "single-class, disjoint train/test split",
            "data_dir": out_dir,
        }, f, indent=2)

    print(f"wrote {out_dir}")
    print(f"  data_train {X_train.shape}  data_test {X_test.shape}")


if __name__ == "__main__":
    main()
