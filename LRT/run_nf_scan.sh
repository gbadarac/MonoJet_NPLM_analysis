#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# NF-uniform 2D LRT campaign driver.
#
# Submits the grid  N_test x ENS x {point,composite} x calib  on the UNIFORM-selected
# wifi ensembles (results_fit_weights_NF_uniform). For each ENS it resolves the member
# seeds the wifi fit used (select_member_seeds.py --mode uniform --rng_seed 0) and passes
# them to LRT.py --member_seeds so the loaded models match the fitted weights.
#
#   Curves : single (seed 100, point-null only) + ensembles {2,4,8,16,32,64,128} x {point,composite}
#   N_test : 100 200 1000 2000 10000 20000 100000 200000      (N_ref = N_test, 1:1)
#   Toys   : null (calib=1) = 250 (--array=0-249) ; test (calib=0) = 100 (--array=0-99)
#   Fixed  : M=500, sigma=0.4, clip=3, rng_seed=0
#   GPU    : partition 'gpu' (7-day max); ENS=128 gets LONG_TIME, the rest STD_TIME.
#   point = frozen weights (plug-in) ; composite = constrained (weight nuisances profiled).
#
# MEMBER_SEEDS is comma-separated, which sbatch --export cannot carry (comma = its delimiter),
# so it is exported into the environment and propagated via --export=ALL.
#
# Usage:
#   DRYRUN=1 ./run_nf_scan.sh                 # print sbatch commands, submit NOTHING (DEFAULT)
#   DRYRUN=0 ./run_nf_scan.sh                 # actually submit
# Subset knobs (space-separated, override at call time):
#   NTEST_LIST="100000"     ENS_LIST="128"     CALIBS="1"
#   MODES="composite"       INCLUDE_SINGLE=0   LONG_TIME="2-00:00:00"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
SELECT="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/select_member_seeds.py"
SELPY="/work/gbadarac/miniforge3/envs/nplm_env/bin/python"   # any python with numpy

DRYRUN=${DRYRUN:-1}
NTEST_LIST=${NTEST_LIST:-"100 200 1000 2000 10000 20000 100000 200000"}
ENS_LIST=${ENS_LIST:-"2 4 8 16 32 64 128"}
CALIBS=${CALIBS:-"1 0"}                       # 1 = null (250 toys) , 0 = test (100 toys)
MODES=${MODES:-"point composite"}             # point = frozen , composite = constrained
INCLUDE_SINGLE=${INCLUDE_SINGLE:-1}           # also submit the single-model (point-null) curve
SINGLE_SEED=${SINGLE_SEED:-100}               # the pinned NF single model

M=500; SIGMA=0.4; CLIP=3; RNG_SEED=0; NMEM=128
STD_TIME=${STD_TIME:-"11:59:00"}
LONG_TIME=${LONG_TIME:-"1-00:00:00"}          # ENS=128 wall (gpu allows up to 7-00:00:00)
LONG_ENS=128
COMMON="MODEL=nf,SCENARIO=ens_2d_uniform,N_KERNELS=$M,KERNEL_SIGMA=$SIGMA,CLIP_B=$CLIP"

array_for() { [[ "$1" -eq 1 ]] && echo "0-249" || echo "0-99"; }   # calib -> #toys
# --mem per N_test, right-sized to measured RSS (Derek, 2026-08: don't over-reserve). NF single
# N=10k used ~2 GB; the 128-member cell is heaviest (~2 GB of state_dicts + data that scales with
# N). Values kept comfortably > observed to avoid OOM. ⚠ seff the FIRST heavy (>=100k, 128-member)
# toys and tighten the top bucket further if they land well under 14 GB.
mem_for() {
  if   [[ "$1" -ge 100000 ]]; then echo "14G"    # 100k/200k
  elif [[ "$1" -ge 10000  ]]; then echo "8G"     # 10k/20k
  else echo "6G"                                  # <=2000 (measured single N=10k = 2 GB)
  fi
}

# resolve the uniform member seeds once per ENS (independent of N_test)
declare -A SEEDS
for ens in $ENS_LIST; do
  SEEDS[$ens]=$("$SELPY" "$SELECT" --mode uniform --n_members "$NMEM" --M "$ens" --rng_seed "$RNG_SEED" 2>/dev/null)
  [[ -z "${SEEDS[$ens]}" ]] && { echo "ERROR: could not resolve seeds for M=$ens" >&2; exit 1; }
done

submit() {  # $1=array $2=time $3=mem $4=export-extra ; MEMBER_SEEDS taken from the env
  if [[ "$DRYRUN" == "1" ]]; then
    echo "MEMBER_SEEDS=${MEMBER_SEEDS} sbatch --partition=gpu --time=$2 --mem=$3 --array=$1 --export=ALL,$COMMON,$4 submit_LRT_toys.sh"
  else
    sbatch --partition=gpu --time="$2" --mem="$3" --array="$1" --export=ALL,"$COMMON,$4" submit_LRT_toys.sh
  fi
}

n=0
for ntest in $NTEST_LIST; do
  mem=$(mem_for "$ntest")

  # ── single model (seed 100), point-null only ──
  if [[ "$INCLUDE_SINGLE" == "1" ]]; then
    export MEMBER_SEEDS="$SINGLE_SEED"
    for calib in $CALIBS; do
      submit "$(array_for "$calib")" "$STD_TIME" "$mem" "ENS=1,CALIBRATION=$calib,NTEST=$ntest"
      n=$((n+1))
    done
  fi

  # ── ensembles ──
  for ens in $ENS_LIST; do
    export MEMBER_SEEDS="${SEEDS[$ens]}"
    [[ "$ens" -eq "$LONG_ENS" ]] && time="$LONG_TIME" || time="$STD_TIME"
    for mode in $MODES; do
      [[ "$mode" == "point" ]] && fix="true" || fix="false"
      for calib in $CALIBS; do
        submit "$(array_for "$calib")" "$time" "$mem" \
          "ENS=$ens,FIX_WIFI_WEIGHTS=$fix,CALIBRATION=$calib,NTEST=$ntest"
        n=$((n+1))
      done
    done
  done
done

echo "# $n array-submissions [DRYRUN=$DRYRUN]  (null=250 toys/cell, test=100 toys/cell)"
echo "# ENS=128 -> --time=$LONG_TIME on partition gpu ; others -> $STD_TIME"
