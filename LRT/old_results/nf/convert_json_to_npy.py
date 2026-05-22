"""
convert_json_to_npy.py
----------------------
Converts old NF LRT results (lrt_outputs.json per toy) into the standard
seed*_T.npy format that analyse_LRT_output.py expects.

The old format (toys_LRT_with_unc.py) saved per-event log-LR arrays in JSON:
    <run_tag>/<mode>/toy_<N>/lrt_outputs.json  ->  {"num": [...], "den": [...], "test": [...]}

The new format (LRT.py) saves a single scalar T per seed:
    <run_tag>/<mode>/seed<N>/seed<N>_T.npy

T = sum(test) = sum(num - den)  (same formula, just summed)

Usage:
    python convert_json_to_npy.py          # converts all run tags in this directory
    python convert_json_to_npy.py --dry_run
"""

import os, json, glob, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))

converted, skipped = 0, 0

for json_path in sorted(glob.glob(os.path.join(HERE, "*", "*", "toy_*", "lrt_outputs.json"))):
    toy_dir  = os.path.dirname(json_path)          # .../toy_42
    mode_dir = os.path.dirname(toy_dir)            # .../calibration or .../comparison
    tag_dir  = os.path.dirname(mode_dir)           # .../run_tag
    toy_name = os.path.basename(toy_dir)           # toy_42
    toy_id   = toy_name.split("_")[-1]             # 42

    seed_dir = os.path.join(mode_dir, f"seed{toy_id}")
    npy_path = os.path.join(seed_dir, f"seed{toy_id}_T.npy")

    if os.path.exists(npy_path):
        skipped += 1
        continue

    with open(json_path) as f:
        data = json.load(f)

    test = np.array(data["test"], dtype=np.float64)
    T    = float(test.sum())

    if args.dry_run:
        print(f"[dry] {os.path.relpath(json_path, HERE)}  ->  T={T:.4f}")
    else:
        os.makedirs(seed_dir, exist_ok=True)
        np.save(npy_path, np.array(T, dtype=np.float64))
        print(f"Converted {os.path.relpath(json_path, HERE)}  ->  T={T:.4f}")

    converted += 1

print(f"\nDone: {converted} converted, {skipped} already existed.")
print("You can now run analyse_LRT_output.py --results_dir <run_tag> on these directories.")
