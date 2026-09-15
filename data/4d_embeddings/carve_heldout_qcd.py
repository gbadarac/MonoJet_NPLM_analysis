#!/usr/bin/env python
"""
Build the LRT observed-data holdout for the 4D JetClass QCD embedding.

The NF ensemble + wifi weights were trained on data_train.npy (100k). The LRT
observed toys (calibration=0) must draw from events NOT used in training. The raw
pool (4d_embedding_data_JetClass/ZJetsToNuNu_*.npz, 2M jets, all QCD) is the
superset the 100k train / 100k test split was carved from; every train and test
row is an exact subset of pool[:, :4] (verified: 100000/100000 each, train n test = 0).

This excludes the 100k TRAINING rows from the 2M pool (exact float32 row-match) and
saves the remaining ~1.9M genuinely-non-training events as data_heldout_qcd.npy.
LRT.py --target_data draws N_test from this (bootstrap) for the observed toys, so
observed data is always disjoint from what the NF + wifi weights were fit on.
"""
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_DIR = os.path.join(HERE, "4d_embedding_data_JetClass")
SPLIT_DIR = os.path.join(HERE, "4d_embedding_qcd_Ntrain100000_Ntest100000_seed42")
OUT = os.path.join(SPLIT_DIR, "data_heldout_qcd.npy")


def as_void(A):
    """View each row of a (N, d) array as one opaque scalar so np.isin matches whole rows."""
    A = np.ascontiguousarray(A.astype(np.float32))
    return A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel()


pool_files = sorted(glob.glob(os.path.join(POOL_DIR, "ZJetsToNuNu_*.npz")))
if not pool_files:
    raise FileNotFoundError(f"no ZJetsToNuNu_*.npz under {POOL_DIR}")
P = np.concatenate([np.load(f)["embeddings"][:, :4] for f in pool_files], axis=0)  # (2M, 4) float32
tr = np.load(os.path.join(SPLIT_DIR, "data_train.npy"))

is_train = np.isin(as_void(P), as_void(tr))
if int(is_train.sum()) != tr.shape[0]:
    raise RuntimeError(f"only {int(is_train.sum())}/{tr.shape[0]} train rows matched in pool "
                       "— row-match failed (preprocessing?); do NOT trust the exclusion.")
held = P[~is_train].astype(np.float64)  # float64 to match data_train/test convention
np.save(OUT, held)
print(f"pool={P.shape[0]}  train_excluded={int(is_train.sum())}  held_out={held.shape[0]}")
print(f"saved {OUT}\n  shape={held.shape} dtype={held.dtype}")
