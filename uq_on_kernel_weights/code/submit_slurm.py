"""
Submit pipeline steps to the SLURM cluster.

Usage:
    python submit_slurm.py --steps train
    python submit_slurm.py --steps data train plots       # multiple steps in one job
    python submit_slurm.py --steps gof --force
    python submit_slurm.py --steps train --name my_run --time 4:00:00 --mem 16000
    python submit_slurm.py --steps gof --partition iaifi_gpu_priority --dry-run
    python submit_slurm.py --steps data --no-gpu

The script generates a SLURM batch file, writes it to the run's logs/ directory,
and submits it with sbatch (unless --dry-run is given). Multiple steps run
sequentially in one job, stopping on first failure.

Default conda env: $LAB/envs/Obfuscated_Activations_SparKer (override with --conda).
"""

import os
import sys
import json
import argparse
import subprocess

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

# Import CONFIG and make_run_name from run_pipeline
sys.path.insert(0, SCRIPTS_DIR)
from run_pipeline import CONFIG, make_run_name, STEPS, OUTPUT_ROOT

# ══════════════════════════════════════════════════════════════════════
# Step scripts
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Submit a pipeline step to SLURM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps", nargs="+", required=True, choices=list(STEPS.keys()),
        help="Which step(s) to submit. Multiple steps run sequentially in one job.",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override run folder name (same as run_pipeline.py --name).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Pass --force to the step script (e.g. for gof).",
    )
    # SLURM options
    parser.add_argument("--partition", "-p", type=str, default="iaifi_gpu_priority",
                        help="SLURM partition (default: iaifi_gpu_priority).")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Request a GPU (default: True).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Do not request a GPU.")
    parser.add_argument("--cpus", type=int, default=1,
                        help="Number of CPUs (default: 1).")
    parser.add_argument("--mem", type=int, default=10000,
                        help="Memory in MB (default: 10000).")
    parser.add_argument("--time", "-t", type=str, default="0-4:00",
                        help="Time limit (default: 0-4:00).")
    parser.add_argument("--module", type=str, default="",
                        help="Module to load (default: none). E.g. python/3.10.9-fasrc01.")
    parser.add_argument("--conda", type=str,
                        default="$LAB/envs/Obfuscated_Activations_SparKer",
                        help="Conda environment to activate (default: $LAB/envs/Obfuscated_Activations_SparKer). "
                             "Set to empty string to skip.")
    parser.add_argument("--venv", type=str, default=None,
                        help="Path to virtualenv to activate (alternative to --conda).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the batch script without submitting.")

    args = parser.parse_args()

    # ── Resolve output directory ──────────────────────────────────
    run_name = args.name if args.name else make_run_name(CONFIG)
    out_dir = os.path.join(OUTPUT_ROOT, run_name)

    # Write pipeline config if it doesn't exist yet
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "pipeline_config.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as f:
            json.dump(CONFIG, f, indent=2)

    # ── Build commands ──────────────────────────────────────────────
    python = sys.executable
    cmds = []
    for step in args.steps:
        cmd = f"{python} {STEPS[step]}"
        if step == "gof" and args.force:
            cmd += " --force"
        cmds.append(cmd)

    # ── Logs directory ────────────────────────────────────────────
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # ── Job name ──────────────────────────────────────────────────
    steps_tag = "+".join(args.steps)
    job_name = f"hiker_{steps_tag}_{run_name[:40]}"

    # ── Build SLURM script ────────────────────────────────────────
    use_gpu = args.gpu and not args.no_gpu

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -c {args.cpus}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -o {logs_dir}/{steps_tag}_%j.out",
        f"#SBATCH -e {logs_dir}/{steps_tag}_%j.err",
    ]
    if use_gpu:
        lines.append("#SBATCH --gpus 1")
    lines.append("")

    if args.module:
        lines.append(f"module load {args.module}")
        lines.append("")

    if args.conda:
        lines.append('eval "$(conda shell.bash hook)"')
        lines.append(f"conda activate {args.conda}")
        lines.append("")
    elif args.venv:
        lines.append(f"source {args.venv}/bin/activate")
        lines.append("")

    lines.append(f"export HIKER_OUT_DIR={out_dir}")
    lines.append(f"cd {SCRIPTS_DIR}")
    lines.append("")
    # Chain commands with && so it stops on first failure
    for i, cmd in enumerate(cmds):
        lines.append(f"echo '=== Step: {args.steps[i]} ==='")
        lines.append(cmd)
        if i < len(cmds) - 1:
            lines.append("if [ $? -ne 0 ]; then echo 'FAILED'; exit 1; fi")
            lines.append("")
    lines.append("")

    batch_content = "\n".join(lines)

    # ── Write and submit ──────────────────────────────────────────
    batch_path = os.path.join(logs_dir, f"submit_{steps_tag}.sh")
    with open(batch_path, "w") as f:
        f.write(batch_content)
    os.chmod(batch_path, 0o755)

    print("=" * 60)
    print(f"SLURM submission: steps={', '.join(args.steps)}")
    print("=" * 60)
    print(f"  Run name:   {run_name}")
    print(f"  Output dir: {out_dir}")
    print(f"  Partition:  {args.partition}")
    print(f"  GPU:        {use_gpu}")
    print(f"  Time:       {args.time}")
    print(f"  Memory:     {args.mem} MB")
    print(f"  Batch file: {batch_path}")
    print()
    print("--- Batch script ---")
    print(batch_content)
    print("--------------------")

    if args.dry_run:
        print("\n  [DRY RUN] Not submitting.")
    else:
        ret = subprocess.run(["sbatch", batch_path])
        if ret.returncode != 0:
            print(f"  ERROR: sbatch failed with return code {ret.returncode}")
            sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
