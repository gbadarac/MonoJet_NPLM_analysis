"""
Submit the wifi-ensemble pipeline to SLURM.

Usage:
    python submit_slurm.py                              # all default steps
    python submit_slurm.py --steps run                  # just wifi training
    python submit_slurm.py --steps coverage gof plot_gof
    python submit_slurm.py --steps gof --force          # re-run GoF
    python submit_slurm.py --name my_wifi_run --time 2:00
    python submit_slurm.py --no-gpu --cpus 4
    python submit_slurm.py --dry-run

Steps run sequentially in one SLURM job, stopping on first failure.

Available steps:
  run        : wifi training (basis + linear head + Σ_w + marginal plot)
  coverage   : first-moment coverage test (bootstrap pulls on <x_0>)
  gof        : classifier GoF (three variants, toy calibration)
  plot_gof   : per-variant null-distribution figure
"""

import os
import sys
import argparse
import subprocess

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))         # package dir
PROJECT_ROOT = SCRIPTS_DIR
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "runs")

sys.path.insert(0, SCRIPTS_DIR)
from config import CONFIG, make_run_name


# ══════════════════════════════════════════════════════════════════════
# Step registry
# ══════════════════════════════════════════════════════════════════════

STEPS = {
    "run":      "run.py",
    "coverage": "coverage.py",
    "gof":      "run_gof.py",
    "plot_gof": "plot_gof.py",
}
ALL_STEPS = ["run", "coverage", "gof", "plot_gof"]


def main():
    parser = argparse.ArgumentParser(
        description="Submit wifi-ensemble pipeline steps to SLURM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps", nargs="+", choices=list(STEPS.keys()),
        default=ALL_STEPS,
        help=f"Which step(s) to submit (default: all). Choices: {list(STEPS.keys())}",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override the auto-generated run folder name.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Pass --force to gof (re-run even if results exist).",
    )
    # Config overrides forwarded to run.py (which dumps the resolved CONFIG into the
    # run's wifi_config.json that coverage/gof/plot read downstream). These also feed
    # make_run_name so concurrent jobs of different size/benchmark get distinct run
    # folders and never clobber each other's live config.py read.
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Override CONFIG['benchmark'] (forwarded to run.py + run name).")
    parser.add_argument("--n-train", dest="n_train", type=int, default=None,
                        help="Override CONFIG['N_train'] (forwarded to run.py + run name).")
    parser.add_argument("--n-test", dest="n_test", type=int, default=None,
                        help="Override CONFIG['N_test'] (forwarded to run.py + run name).")
    # SLURM options (mirrors HIKER's submit_slurm.py defaults)
    parser.add_argument("--partition", "-p", type=str,
                        default="qgpu,gpu",
                        help="SLURM partition (comma-separated list allowed).")
    parser.add_argument("--account", type=str, default="gpu_gres",
                        help="SLURM account (this cluster needs gpu_gres for GPU "
                             "jobs). Set empty to omit.")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Request a GPU (default: True).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Do not request a GPU.")
    parser.add_argument("--cpus", type=int, default=1,
                        help="Number of CPUs (default: 1).")
    parser.add_argument("--mem", type=str, default="16G",
                        help="Memory, with units (default: 16G).")
    parser.add_argument("--time", "-t", type=str, default="0-4:00",
                        help="Time limit (default: 0-4:00).")
    parser.add_argument("--module", type=str, default="",
                        help="Module to load (default: none).")
    parser.add_argument("--conda", type=str, default="kernels_env",
                        help="Conda env to activate. Set empty to skip.")
    parser.add_argument("--venv", type=str, default=None,
                        help="Path to virtualenv (alternative to --conda).")
    parser.add_argument("--mail", type=str, default="",
                        help="Email for SLURM notifications. Empty to skip.")
    parser.add_argument("--mail-type", type=str, default="BEGIN,END,FAIL",
                        help="SLURM --mail-type value.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the batch script without submitting.")

    args = parser.parse_args()

    # ── Apply config overrides (mirror them into CONFIG so the run name reflects
    #    the actual size/benchmark this job will run) ───────────────
    if args.benchmark is not None:
        CONFIG["benchmark"] = args.benchmark
    if args.n_train is not None:
        CONFIG["N_train"] = args.n_train
    if args.n_test is not None:
        CONFIG["N_test"] = args.n_test

    # ── Resolve run dir ───────────────────────────────────────────
    run_name = args.name if args.name else make_run_name(CONFIG)
    out_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # ── Build commands ────────────────────────────────────────────
    python = "python3 -u"
    # Overrides go ONLY to run.py; it persists them to wifi_config.json, which the
    # downstream steps read via --name.
    run_overrides = ""
    if args.benchmark is not None:
        run_overrides += f" --benchmark {args.benchmark}"
    if args.n_train is not None:
        run_overrides += f" --n-train {args.n_train}"
    if args.n_test is not None:
        run_overrides += f" --n-test {args.n_test}"
    cmds = []
    for step in args.steps:
        script = os.path.join(SCRIPTS_DIR, STEPS[step])
        cmd = f"{python} {script} --name {run_name}"
        if step == "run":
            cmd += run_overrides
        if step == "gof" and args.force:
            cmd += " --force"
        cmds.append((step, cmd))

    # ── SLURM directives ──────────────────────────────────────────
    use_gpu = args.gpu and not args.no_gpu
    steps_tag = "+".join(args.steps)
    job_name = f"wifi_{steps_tag}_{run_name[:32]}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -n {args.cpus}",
        f"#SBATCH -t {args.time}",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH -o {logs_dir}/{steps_tag}_%j.out",
        f"#SBATCH -e {logs_dir}/{steps_tag}_%j.err",
    ]
    if use_gpu:
        lines.append("#SBATCH --gres=gpu:1")
        if args.account:
            lines.append(f"#SBATCH --account={args.account}")
    if args.mail:
        lines.append(f"#SBATCH --mail-type={args.mail_type}")
        lines.append(f"#SBATCH --mail-user={args.mail}")
    lines.append("")

    if args.module:
        lines += [f"module load {args.module}", ""]
    if args.conda:
        lines += [
            "source /work/gbadarac/miniforge3/bin/activate",
            f"conda activate {args.conda}",
            "",
        ]
    elif args.venv:
        lines += [f"source {args.venv}/bin/activate", ""]

    lines += [
        "export PYTHONUNBUFFERED=1",
        f"cd {SCRIPTS_DIR}",
        "",
    ]
    for i, (step, cmd) in enumerate(cmds):
        lines.append(f"echo '=== Step: {step} ==='")
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
    print(f"SLURM submission: steps = {', '.join(args.steps)}")
    print("=" * 60)
    print(f"  Run name:   {run_name}")
    print(f"  Output dir: {out_dir}")
    print(f"  Partition:  {args.partition}")
    print(f"  GPU:        {use_gpu}")
    print(f"  CPUs:       {args.cpus}")
    print(f"  Time:       {args.time}")
    print(f"  Memory:     {args.mem}")
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
