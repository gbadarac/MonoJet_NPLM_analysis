"""
Prepare the 4D embedding as a pipeline data cache.

The wifi_better_basis pipeline reads its data from
    data/<benchmark>_Ntrain<Ntr>_Ntest<Nte>_seed<seed>/{data_train.npy,data_test.npy}
and never regenerates when those exist (run.py:load_or_generate_data). So to run
on Gaia's 4D embedding we just drop subsamples of train_h/test_h into that path.
No change to run.py / run_gof.py / plot_marginals.py is needed; d=4 flows through
from the array shape.

train_h -> data_train.npy   (fit the density model: basis + wifi weights)
test_h  -> data_test.npy    (held-out target sample the GoF is run against)
val_h is intentionally ignored (it was the embedding-training val set).

Usage (sizes come from Gaia; benchmark name must match config.py):
    python make_4d_cache.py --n_train 100000 --n_test 100000 \
        --benchmark 4d_embedding --seed 42

Then set in config.py: benchmark="4d_embedding", N_train, N_test, seed to match,
and run WITHOUT the coverage step:
    python submit_slurm.py --steps run gof plot_gof
(coverage.py needs an analytic truth that real data doesn't have.)
"""

import os
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "data", "4d_embedding_Gaia")


def _subsample(src_path, n, seed):
    a = np.load(src_path).astype(np.float64)
    if n > a.shape[0]:
        raise ValueError(f"requested {n} > available {a.shape[0]} in {src_path}")
    rng = np.random.RandomState(seed)
    idx = rng.choice(a.shape[0], size=n, replace=False)
    return a[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_test", type=int, required=True)
    ap.add_argument("--benchmark", type=str, default="4d_embedding")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # train_h -> train (model fit);  test_h -> test (GoF observed). Independent files,
    # so different seed offsets are cosmetic but kept for reproducibility.
    X_train = _subsample(os.path.join(EMB, "train_h.npy"), args.n_train, args.seed)
    X_test  = _subsample(os.path.join(EMB, "test_h.npy"),  args.n_test,  args.seed + 1)

    name = (f"{args.benchmark}_Ntrain{args.n_train}_Ntest{args.n_test}"
            f"_seed{args.seed}")
    out_dir = os.path.join(HERE, "data", name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "data_train.npy"), X_train)
    np.save(os.path.join(out_dir, "data_test.npy"),  X_test)
    with open(os.path.join(out_dir, "data_config.json"), "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "N_train": args.n_train, "N_test": args.n_test,
            "seed": args.seed, "d": int(X_train.shape[1]),
            "source": "4d_embedding_Gaia (train_h/test_h)",
            "data_dir": out_dir,
        }, f, indent=2)

    print(f"wrote {out_dir}")
    print(f"  data_train {X_train.shape}  data_test {X_test.shape}")


if __name__ == "__main__":
    main()
