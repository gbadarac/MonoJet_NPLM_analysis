"""
Reproduce the 2D GMM + skew-normal "heavy tail" toymodel as a train/test cache.

Lives in <repo>/data/2d_gmm_toymodel/, next to the data it produces. This is the
SAME data generation Sean uses in wifi_better_basis (benchmarks.py
`generate_2d_gmm_skew`, seed 42), so the two pipelines run on identical samples.
The DGP is copied verbatim here so this folder is self-reproducing (no import
dependency on wifi_better_basis).

Distribution (d = 2):
  x0: bimodal Gaussian mixture  (50/50, mu=-0.70/sig=0.12 and mu=-0.30/sig=0.12)
  x1: skew-normal               (loc=1.0, scale=0.75, alpha=8.0)

Split convention (matches wifi_better_basis/run.py):
  data_train = generate(N_train, seed)         # seed 42 by default
  data_test  = generate(N_test,  seed + 1)     # seed 43 -> disjoint sample

Writes a cache dir alongside this script (d flows through from the array shape):
    <repo>/data/2d_gmm_toymodel/<benchmark>_Ntrain<Ntr>_Ntest<Nte>_seed<seed>/
        data_train.npy   (fit the density model / kernel ensemble)
        data_test.npy    (held-out target sample the GoF/LRT is run against)
        data_config.json

Usage (defaults reproduce the cached seed-42 100k/100k split byte-for-byte):
    python generate_2d_gaussian_heavy_tail_target_data.py
    python generate_2d_gaussian_heavy_tail_target_data.py --n-train 100000 \
        --n-test 100000 --seed 42
"""

import os
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # <repo>/data/2d_gmm_toymodel/
BENCHMARK = "2d_gmm_skew"


def generate_2d_gmm_skew(N, seed=42):
    """
    2D distribution:
      x0: bimodal Gaussian mixture  (50/50, mu=-0.70/sig=0.12 and mu=-0.30/sig=0.12)
      x1: skew-normal               (loc=1.0, scale=0.75, alpha=8.0)
    Verbatim copy of wifi_better_basis/benchmarks.py::generate_2d_gmm_skew so the
    output is byte-identical to Sean's cache. Do not change the RNG call order.
    """
    np.random.seed(seed)

    # Feature 0: bimodal Gaussian mixture
    wG = 0.50
    mu_a, sig_a = -0.70, 0.12
    mu_b, sig_b = -0.30, 0.12
    n_a = np.random.binomial(N, wG)
    x0 = np.concatenate([
        np.random.normal(mu_a, sig_a, n_a),
        np.random.normal(mu_b, sig_b, N - n_a),
    ])

    # Feature 1: skew-normal
    loc, scale, alpha = 1.0, 0.75, 8.0
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    z0 = np.random.randn(N)
    z1 = np.random.randn(N)
    x1 = loc + scale * (delta * np.abs(z0) + np.sqrt(1.0 - delta ** 2) * z1)

    data = np.column_stack([x0, x1])
    np.random.shuffle(data)
    return data.astype(np.float64)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--n-test", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="output cache dir (default: sibling "
                         "<benchmark>_Ntrain<..>_Ntest<..>_seed<..>/ next to this "
                         "script). Point elsewhere to verify without overwriting.")
    args = ap.parse_args()

    # Train and test are disjoint draws (seed and seed+1), matching run.py.
    X_train = generate_2d_gmm_skew(args.n_train, seed=args.seed)
    X_test = generate_2d_gmm_skew(args.n_test, seed=args.seed + 1)

    name = (f"{BENCHMARK}_Ntrain{args.n_train}_Ntest{args.n_test}"
            f"_seed{args.seed}")
    out_dir = args.out_dir or os.path.join(HERE, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "data_train.npy"), X_train)
    np.save(os.path.join(out_dir, "data_test.npy"), X_test)
    with open(os.path.join(out_dir, "data_config.json"), "w") as f:
        json.dump({
            "benchmark": BENCHMARK,
            "N_train": args.n_train, "N_test": args.n_test,
            "seed": args.seed, "d": int(X_train.shape[1]),
            "source": "wifi_better_basis/benchmarks.py::generate_2d_gmm_skew "
                      "(verbatim DGP); data_train=generate(N_train,seed), "
                      "data_test=generate(N_test,seed+1)",
            "data_dir": out_dir,
        }, f, indent=2)

    print(f"wrote {out_dir}")
    print(f"  data_train {X_train.shape}  data_test {X_test.shape}")


if __name__ == "__main__":
    main()
