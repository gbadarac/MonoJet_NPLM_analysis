#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Re-run only the LRT toys that are actually MISSING on disk.
#
# Ground truth = presence of  <cfg>/<calibration|test>/seed<id>/seed<id>_T.txt
# for toy ids 0..99. This is deduplicated against every resubmit already done,
# so it never redoes a toy you already have (unlike replaying the SLURM history).
#
# The 200k/1M configs get --mem=18G (peak RSS ~14 G; the 12 G base default OOMs them).
#
# Usage:
#   ./rerun_missing_toys.sh            # DRYRUN: print sbatch lines, submit NOTHING (default)
#   DRYRUN=0 ./rerun_missing_toys.sh   # actually submit
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
DRYRUN=${DRYRUN:-1}

python3 - "$DRYRUN" <<'PY'
import os, glob, re, sys, subprocess
dryrun = sys.argv[1] != "0"
root = "results"
nref = {25000:125000, 50000:250000, 100000:500000, 200000:1000000}

leaves = glob.glob(os.path.join(root, "*", "*", "*", "*", "seed*"))
done = {}                       # cfg_dir -> set(completed toy ids)
for d in leaves:
    if not os.path.isdir(d): continue
    m = re.search(r"seed(\d+)$", os.path.basename(d))
    if not m: continue
    tid = int(m.group(1))
    if tid > 99: continue       # only the 0-99 toy arrays
    cfg = os.path.dirname(d)
    done.setdefault(cfg, set())
    if os.path.exists(os.path.join(d, f"seed{tid}_T.txt")):
        done[cfg].add(tid)

n_sub = n_toys = 0
for cfg in sorted(done):
    missing = sorted(set(range(100)) - done[cfg])
    if not missing: continue
    # cfg = results/<model>/<scenario_base>/<run_tag>/<calibration|test>
    parts = cfg.split(os.sep)
    mode  = parts[-1]                       # calibration | test
    tag   = parts[-2]                       # run_tag
    calib = 1 if mode == "calibration" else 0
    E   = int(re.search(r"Nens(\d+)",  tag).group(1))
    N   = int(re.search(r"Ntest(\d+)", tag).group(1))
    fix = "true" if "frozen_weights" in tag else "false"
    mem = ["--mem=18G"] if N >= 200000 else []
    exp = (f"ALL,SCENARIO=ens_2d,ENS={E},FIX_WIFI_WEIGHTS={fix},MODEL=kernels,"
           f"CALIBRATION={calib},NTEST={N},N_REF={nref[N]},N_KERNELS=500")
    cmd = ["sbatch", *mem, f"--array={','.join(map(str,missing))}",
           f"--export={exp}", "submit_LRT_toys.sh"]
    print(" ".join(cmd))
    if not dryrun:
        subprocess.run(cmd, check=True)
    n_sub += 1; n_toys += len(missing)

print(f"\n# {n_sub} resubmissions, {n_toys} toys  [DRYRUN={'1' if dryrun else '0'}]")
PY
