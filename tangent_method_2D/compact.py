"""Merge raw chunk files into results/tstats.parquet and delete them (ledger-safe).

  python compact.py <run_dir> [<run_dir> ...]     # bare tags looked up under runs/
  python compact.py runs/*/                       # everything

Just a wrapper around 'merge.py --compact' with the run folder as argument.
Workers treat ledgered keys as done, so compaction never breaks resumability.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
runs = sys.argv[1:]
if not runs:
    sys.exit(__doc__)
for run in runs:
    if not os.path.isdir(run):                       # bare tag -> runs/<tag>
        run = os.path.join(HERE, "runs", run)
    pj = os.path.join(run, "params.json")
    if not os.path.isfile(pj):
        print(f"skip (no params.json): {run}"); continue
    print(f"== {run}")
    env = dict(os.environ, GOF2D_PARAMS=os.path.abspath(pj))
    subprocess.run([sys.executable, os.path.join(HERE, "merge.py"), "--compact"],
                   env=env, check=True)
