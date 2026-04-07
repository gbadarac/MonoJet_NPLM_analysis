"""
Script 1: Generate train and test data from a named benchmark distribution.

Data is stored in a shared directory under {PROJECT_ROOT}/data/ keyed by
benchmark name, N_train, N_test, and seed. Multiple model runs can share
the same data without regenerating.

If the data already exists, skips generation and just links it to the run.

Reads: {OUT_DIR}/pipeline_config.json
Saves: {DATA_DIR}/data_train.npy, data_test.npy, data_config.json
       {OUT_DIR}/data_config.json (with data_dir path for downstream scripts)
"""

import os
import json
import numpy as np
from benchmarks import get_benchmark, list_benchmarks

# ── Directories ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get("HIKER_OUT_DIR", os.path.join(PROJECT_ROOT, "output"))
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load pipeline config ─────────────────────────────────────────────
cfg_path = os.path.join(OUT_DIR, "pipeline_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
else:
    cfg = {}

BENCHMARK = cfg.get("benchmark", "2d_gaussian")
SEED = cfg.get("seed", 42)
N_TRAIN = cfg.get("N_train", 100_000)
N_TEST = cfg.get("N_test", 100_000)

# ── Shared data directory ─────────────────────────────────────────────
data_name = f"{BENCHMARK}_Ntrain{N_TRAIN}_Ntest{N_TEST}_seed{SEED}"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", data_name)

print("Generating data...")
print(f"  Benchmark = {BENCHMARK}")
print(f"  N_train = {N_TRAIN}, N_test = {N_TEST}, seed = {SEED}")
print(f"  Data dir = {DATA_DIR}")
print(f"  Available benchmarks: {list_benchmarks()}")

# ── Generate (or skip if already exists) ──────────────────────────────
train_path = os.path.join(DATA_DIR, "data_train.npy")
test_path = os.path.join(DATA_DIR, "data_test.npy")
config_path = os.path.join(DATA_DIR, "data_config.json")

if os.path.exists(train_path) and os.path.exists(test_path):
    print("  Data already exists — skipping generation.")
    data_train = np.load(train_path)
    data_test = np.load(test_path)
else:
    os.makedirs(DATA_DIR, exist_ok=True)
    generate = get_benchmark(BENCHMARK)
    data_train = generate(N_TRAIN, seed=SEED)
    data_test = generate(N_TEST, seed=SEED + 1)

    np.save(train_path, data_train)
    np.save(test_path, data_test)
    print(f"  Saved data_train.npy  shape={data_train.shape}")
    print(f"  Saved data_test.npy   shape={data_test.shape}")

d = data_train.shape[1]
print(f"  Dimensionality: d = {d}")
for dim in range(d):
    print(f"    x{dim}: mean={data_train[:, dim].mean():.4f}, "
          f"std={data_train[:, dim].std():.4f}, "
          f"range=[{data_train[:, dim].min():.3f}, {data_train[:, dim].max():.3f}]")

# ── Save data config into both DATA_DIR and OUT_DIR ───────────────────
data_config = {
    "benchmark": BENCHMARK,
    "N_train": N_TRAIN,
    "N_test": N_TEST,
    "seed": SEED,
    "d": d,
    "data_dir": DATA_DIR,
}

# Save to shared data dir
with open(config_path, "w") as f:
    json.dump(data_config, f, indent=2)

# Save to run dir (so downstream scripts can find the data)
with open(os.path.join(OUT_DIR, "data_config.json"), "w") as f:
    json.dump(data_config, f, indent=2)

print("Done.")
