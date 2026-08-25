#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Launcher for the 2D kernel LRT  Ntest/Nref  scan  (M held FIXED at 500).
#
# Matrix:
#   Ntest -> Nref (= 5*Ntest):  25k->125k, 50k->250k, 100k->500k, 200k->1M
#   Configs:
#     single           : single kernel model (1model_2d, frozen forced, no ENS)
#     ens_frozen       : kernel ensemble, w frozen at w_hat        (ENS scan)
#     ens_constrained  : kernel ensemble, w profiled under prior   (ENS scan)
#   ENS size scan (ensemble configs only): 16 32 64 128
#   CALIBRATION: 1 = null toys (SIR-from-q) , 0 = observed (bootstrap target)
#
# Each submission is an array of 100 toys (submit_LRT_toys.sh --array=0-99).
# The Ntest=200k / Nref=1M jobs get --mem=18G (peak RSS observed ~14 G; ~1.3x margin).
#
# Usage:
#   DRYRUN=1 ./run_lrt_scan.sh                 # print sbatch commands, submit NOTHING
#   ./run_lrt_scan.sh                          # actually submit
#
# Subset knobs (override at call time, space-separated):
#   CALIBS="1"                                 # null only            (default "1 0")
#   CONFIGS="ens_constrained"                  # one config           (default all three)
#   ENS_LIST="128"                             # one ensemble size    (default "16 32 64 128")
#   NTEST_LIST="100000"                        # one Ntest            (default all four)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

DRYRUN=${DRYRUN:-0}
CALIBS=${CALIBS:-"1 0"}
CONFIGS=${CONFIGS:-"single ens_frozen ens_constrained"}
ENS_LIST=${ENS_LIST:-"16 32 64 128"}
NTEST_LIST=${NTEST_LIST:-"25000 50000 100000 200000"}
M=500

nref_for() { case "$1" in
  25000) echo 125000 ;; 50000) echo 250000 ;;
  100000) echo 500000 ;; 200000) echo 1000000 ;;
  *) echo "bad NTEST=$1" >&2; exit 1 ;; esac; }
mem_for() { [[ "$1" -ge 200000 ]] && echo "--mem=18G" || echo ""; }

run() {  # run <sbatch-args...>
  if [[ "$DRYRUN" == "1" ]]; then echo "sbatch $*"; else sbatch "$@"; fi
}

n_submit=0
for ntest in $NTEST_LIST; do
  nref=$(nref_for "$ntest"); mem=$(mem_for "$ntest")
  for calib in $CALIBS; do
    for cfg in $CONFIGS; do
      base="MODEL=kernels,CALIBRATION=$calib,NTEST=$ntest,N_REF=$nref,N_KERNELS=$M"
      case "$cfg" in
        single)
          run $mem --export=ALL,SCENARIO=1model_2d,$base submit_LRT_toys.sh
          n_submit=$((n_submit+1)) ;;
        ens_frozen)
          for e in $ENS_LIST; do
            run $mem --export=ALL,SCENARIO=ens_2d,ENS=$e,FIX_WIFI_WEIGHTS=true,$base submit_LRT_toys.sh
            n_submit=$((n_submit+1))
          done ;;
        ens_constrained)
          for e in $ENS_LIST; do
            run $mem --export=ALL,SCENARIO=ens_2d,ENS=$e,FIX_WIFI_WEIGHTS=false,$base submit_LRT_toys.sh
            n_submit=$((n_submit+1))
          done ;;
        *) echo "unknown CONFIG=$cfg" >&2; exit 1 ;;
      esac
    done
  done
done
echo "# ${n_submit} array-submissions ($(( n_submit * 100 )) toys) [DRYRUN=$DRYRUN]"
