"""
Pipeline runner: configure and run the HIKER Hessian UQ pipeline.

Edit the CONFIG dictionary below to set all hyperparameters (model architecture,
training, ensemble, coverage, GoF). Results are stored under runs/<config_name>/.

Steps: data, train, plots, coverage, gof, plot_gof

Usage:
    python run_pipeline.py                              # run all default steps
    python run_pipeline.py --steps data train plots     # specific steps
    python run_pipeline.py --steps plots                # re-plot without retraining
    python run_pipeline.py --steps gof --force          # re-run GoF variants
    python run_pipeline.py --steps plot_gof             # re-plot GoF diagnostics
    python run_pipeline.py --name my_run                # custom folder name
    python run_pipeline.py --name existing_run --steps gof  # run on existing folder
    python run_pipeline.py --dry-run                    # preview without executing
"""

import os
import sys
import json
import argparse
import subprocess

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this dictionary to change hyperparameters
# ══════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data generation ───────────────────────────────────────────
    # "benchmark": which distribution to sample from.
    #   Available: "2d_gaussian", "2d_gmm_skew"
    #   (see benchmarks.py for the full list and to add new ones)
    "benchmark": "2d_gmm_skew",
    "seed": 42,
    "N_train": 100_000,
    "N_test": 100_000,

    # ── HIKER model ───────────────────────────────────────────────
    # Use int for single layer, list for multi-layer:
    #   "M_KERNELS": 50          -> one layer with 50 kernels
    #   "M_KERNELS": [50, 50]    -> two layers, 50 kernels each
    # KERNEL_SIGMA is the final kernel width per layer (= resolution scale).
    # EPOCHS follows the same int-or-list convention.
    "M_KERNELS": [50,80,100],
    "KERNEL_SIGMA": [0.15,0.10,0.05],
    # Width annealing (optional — omit to disable):
    # "WIDTH_INIT": [0.5, 0.3, ...],  # starting width per layer, anneals down to KERNEL_SIGMA
    # "T_INI": 0,                     # epoch at which annealing starts
    # "DECAY_EPOCHS": 0.5,            # fraction of epochs for annealing
    "LR": 1e-5,
    "EPOCHS": [100000, 1000,1000],
    "PATIENCE": 1000,
    "BATCH_SIZE": 10000,           # mini-batch size (None = full dataset)
    "TRAIN_MODE": "joint",         # "sequential" (layer-by-layer) or "joint" (all at once)
    "EARLY_STOPPING": 5,          # stop after N eval steps without val improvement (None = off)
    "TRAIN_CENTROIDS": True,
    "USE_NORM_KERNEL": True,       # True = norm kernel gauge, False = post-hoc + projection
    "TRAIN_NORM_CENTRE": False,    # False = freeze norm kernel at dataset mean (ignored if USE_NORM_KERNEL=False)
    "COEFFS_CLIP": None,
    "LAM_L2": 0.0,
    "N_ENSEMBLE": 10,               # 1 = single model, >1 = bootstrap ensemble
    "N_TRAIN_PER_MODEL": 50000,  # training points per model (None = use all N_train)

    # ── Coverage ──────────────────────────────────────────────────
    "N_BOOTSTRAP": 100,

    # ── GoF test ──────────────────────────────────────────────────
    # Each GoF config gets its own subfolder: gof_{GOF_LABEL}/
    # If GOF_LABEL is None, auto-generated from hyperparameters.
    # "GOF_LABEL": "my_test",
    "N_TEST_GOF": 100000,
    "N_KERNELS_NUM": 80,
    "KERNEL_WIDTH_NUM": 0.10,
    "CLIP_NUM": 0.1,
    "LAMBDA_NUM": 0.0,
    "EPOCHS_DEN": 100000,
    "EPOCHS_NUM": 100000,
    "LR_DEN": 5e-6,
    "LR_NUM": 5e-6,
    "PATIENCE_GOF": 2000,
    "N_TOYS": 30,
}

# ── Where to store all runs ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "runs")

# ══════════════════════════════════════════════════════════════════════
# Folder name from HIKER hyperparameters
# ══════════════════════════════════════════════════════════════════════

def _compact(v):
    """Format a scalar or list compactly for folder names."""
    if isinstance(v, list):
        return "-".join(str(x) for x in v)
    return str(v)

def make_run_name(cfg):
    """Build a descriptive folder name from the key hyperparameters."""
    M = cfg["M_KERNELS"]
    n_layers = len(M) if isinstance(M, list) else 1
    M_total = sum(M) if isinstance(M, list) else M
    parts = [
        cfg.get("benchmark", "2d_gaussian"),
        f"L{n_layers}_M{M_total}",
        f"N{cfg['N_train']}",
    ]
    N_ens = cfg.get("N_ENSEMBLE", 1)
    if N_ens > 1:
        parts.append(f"ens{N_ens}")
    return "_".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Step definitions
# ══════════════════════════════════════════════════════════════════════

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    "data":      os.path.join(SCRIPTS_DIR, "generate_data.py"),
    "train":     os.path.join(SCRIPTS_DIR, "train_and_uq.py"),
    "plots":     os.path.join(SCRIPTS_DIR, "plot_marginals.py"),
    "coverage":  os.path.join(SCRIPTS_DIR, "coverage.py"),
    "gof":       os.path.join(SCRIPTS_DIR, "run_gof.py"),
    "plot_gof":  os.path.join(SCRIPTS_DIR, "plot_gof.py"),
}

ALL_STEPS = ["data", "train", "plots", "coverage", "gof"]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run the HIKER Hessian UQ pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps", nargs="+", choices=list(STEPS.keys()), default=ALL_STEPS,
        help="Which pipeline steps to run (default: all). "
             "Choices: " + ", ".join(STEPS.keys()),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override the auto-generated run folder name.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run of GoF variants even if results already exist.",
    )
    args = parser.parse_args()

    # Build output directory
    run_name = args.name if args.name else make_run_name(CONFIG)
    out_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Write pipeline config
    cfg_path = os.path.join(out_dir, "pipeline_config.json")
    with open(cfg_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("=" * 60)
    print("HIKER Hessian UQ Pipeline")
    print("=" * 60)
    print(f"  Run name:   {run_name}")
    print(f"  Output dir: {out_dir}")
    print(f"  Steps:      {', '.join(args.steps)}")
    print(f"  Config:     {cfg_path}")
    print()

    # Print key hyperparameters
    print("  Key hyperparameters:")
    for key in ["M_KERNELS", "KERNEL_SIGMA", "LR", "EPOCHS",
                "TRAIN_CENTROIDS", "N_train", "N_test"]:
        print(f"    {key:20s} = {CONFIG[key]}")
    print()

    if args.dry_run:
        print("  [DRY RUN] Would execute:")
        for step in args.steps:
            print(f"    HIKER_OUT_DIR={out_dir} python {STEPS[step]}")
        return

    # Find python executable
    python = sys.executable

    # Run each step
    env = os.environ.copy()
    env["HIKER_OUT_DIR"] = out_dir

    for step in args.steps:
        script = STEPS[step]
        print("\n" + "=" * 60)
        print(f"STEP: {step}  ({os.path.basename(script)})")
        print("=" * 60 + "\n")

        cmd = [python, script]
        if step == "gof" and args.force:
            cmd.append("--force")
        ret = subprocess.run(cmd, env=env, cwd=SCRIPTS_DIR)
        if ret.returncode != 0:
            print(f"\n  ERROR: step '{step}' failed with return code {ret.returncode}")
            sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Results in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
