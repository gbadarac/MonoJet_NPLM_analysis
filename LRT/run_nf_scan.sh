#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Unified NF LRT driver.  ONE script, THREE jobs, selected by two knobs:
#
#   DIM  = 2d | 4d           — feature dimension. Selects the SCENARIO, the wifi ensemble
#                              pool size (--n_members), the pinned single-model seed, and
#                              the default sigma.
#   SCAN = campaign | sigma  — WHAT to scan:
#       campaign (default) : the full grid
#                              N_test × { single + ENS × {point,composite} } × calib
#                            at a FIXED sigma, full toy counts.        (was run_nf_scan.sh)
#       sigma              : single-model σ-SELECTION scan — loop σ × calib at a FIXED
#                            N_test, LEAN toys. Results land in distinct _W{sigma} run-tag
#                            dirs (sigma is in the folder name → no mixing). Pick σ from the
#                            nulls (analyse_LRT_output.py per dir: clean null chi2/DOF_eff +
#                            min observed p), then rerun SCAN=campaign at that FIXED σ.
#                                                              (was run_nf_sigma_scan_4d.sh)
#
# Both modes are thin SLURM fan-outs over submit_LRT_toys.sh → LRT.py; the physics lives
# there. Fixed for both: MODEL=nf, N_ref=N_test (1:1, submit default), M=500, clip=3,
# rng_seed=0, --partition=gpu. ENS=1 (single model) ⇒ LRT.py synthesizes w=[1.0] and forces
# frozen / POINT-null (the ensemblecomponents1 wifi is ignored). MEMBER_SEEDS is comma-
# separated (sbatch --export cannot carry commas, its delimiter) so it is exported into the
# environment and propagated via --export=ALL.
#
# sigma candidates (4D): RAW pairwise-distance percentiles {P1,P25,P50,P75,P90,P99}+2×P99 on
# the training data = {0.05,0.53,1.24,2.11,2.8,3.46,6.92} (features on near-equal scales, LRT
# places kernels on RAW data). Sigma is a data-space length scale (N-independent), so the value
# chosen by the scan transfers to the campaign at larger N_test.
#
# Usage:
#   DRYRUN=1 ./run_nf_scan.sh                          # 2D full campaign, print only (DEFAULT)
#   DRYRUN=0 DIM=2d ./run_nf_scan.sh                   # 2D full campaign, SUBMIT
#   DRYRUN=0 DIM=4d SCAN=sigma ./run_nf_scan.sh        # 4D σ-selection scan, SUBMIT
#   DRYRUN=0 DIM=4d SIGMA=<chosen> ./run_nf_scan.sh    # 4D full campaign at the chosen σ
#
# Subset knobs (space-separated lists; override at call time):
#   campaign : NTEST_LIST  ENS_LIST  MODES  CALIBS  INCLUDE_SINGLE  SIGMA  STD_TIME  LONG_TIME
#   sigma    : SIGMA_LIST  NTEST  CALIBS  TIME  NULL_TOYS  OBS_TOYS
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
SELECT="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/select_member_seeds.py"
SELPY="/work/gbadarac/miniforge3/envs/nplm_env/bin/python"   # any python with numpy

DRYRUN=${DRYRUN:-1}
DIM=${DIM:-2d}
SCAN=${SCAN:-campaign}
CALIBS=${CALIBS:-"1 0"}                # 1 = null , 0 = observed/test
M=500; CLIP=3; RNG_SEED=0

# ── DIM config: SCENARIO, pool size (--n_members), pinned single seed, default sigma ──
case "$DIM" in
  2d) SCENARIO=ens_2d_uniform; NMEM=128; SINGLE_SEED_DEF=100; SIGMA_DEF=0.4 ;;
  4d) SCENARIO=ens_4d_uniform; NMEM=256; SINGLE_SEED_DEF=133; SIGMA_DEF=""  ;;  # σ := SCAN=sigma result
  *)  echo "unknown DIM=$DIM (want 2d|4d)" >&2; exit 1 ;;
esac
SINGLE_SEED=${SINGLE_SEED:-$SINGLE_SEED_DEF}
COMMON="MODEL=nf,SCENARIO=$SCENARIO,N_KERNELS=$M,CLIP_B=$CLIP"   # KERNEL_SIGMA passed per-iteration

# --mem per N_test, right-sized to measured RSS (Derek 2026-08: don't over-reserve). NF single
# N=10k used ~2 GB; the 128-member 100k/200k cell is heaviest (~2 GB state_dicts + data ∝ N).
# Kept comfortably > observed to avoid OOM.
mem_for() {
  if   [[ "$1" -ge 100000 ]]; then echo "14G"    # 100k / 200k
  elif [[ "$1" -ge 10000  ]]; then echo "8G"     # 10k / 20k
  else echo "6G"                                  # ≤2000
  fi
}

# resolve the uniform member seeds for an ensemble size (comma-free list, last = norm model)
resolve_seeds() { "$SELPY" "$SELECT" --mode uniform --n_members "$NMEM" --M "$1" --rng_seed "$RNG_SEED" 2>/dev/null; }

# array range from toy counts (calib 1=null → NULL_TOYS ; 0=observed → OBS_TOYS)
array_for() { [[ "$1" -eq 1 ]] && echo "0-$((NULL_TOYS-1))" || echo "0-$((OBS_TOYS-1))"; }

submit() {  # $1=array $2=time $3=mem $4=export-extra ; MEMBER_SEEDS taken from the env
  if [[ "$DRYRUN" == "1" ]]; then
    echo "MEMBER_SEEDS=${MEMBER_SEEDS} sbatch --partition=gpu --time=$2 --mem=$3 --array=$1 --export=ALL,$COMMON,$4 submit_LRT_toys.sh"
  else
    sbatch --partition=gpu --time="$2" --mem="$3" --array="$1" --export=ALL,"$COMMON,$4" submit_LRT_toys.sh
  fi
}

n=0
if [[ "$SCAN" == "sigma" ]]; then
  # ── σ-SELECTION scan: single model, fixed N_test, loop σ × calib, LEAN toys ──
  SIGMA_LIST=${SIGMA_LIST:-"0.05 0.53 1.24 2.11 2.8 3.46 6.92"}
  NTEST=${NTEST:-10000}
  SIG_TIME=${TIME:-"02:00:00"}
  NULL_TOYS=${NULL_TOYS:-100}; OBS_TOYS=${OBS_TOYS:-50}
  mem=$(mem_for "$NTEST")
  export MEMBER_SEEDS="$SINGLE_SEED"    # ENS=1 ⇒ LRT.py forces frozen/point-null, w=[1.0]
  for sigma in $SIGMA_LIST; do
    for calib in $CALIBS; do
      submit "$(array_for "$calib")" "$SIG_TIME" "$mem" "ENS=1,CALIBRATION=$calib,NTEST=$NTEST,KERNEL_SIGMA=$sigma"
      n=$((n+1))
    done
  done
  echo "# $n array-submissions [DRYRUN=$DRYRUN]  (DIM=$DIM σ-scan: single model seed $SINGLE_SEED, N_test=$NTEST, null=$NULL_TOYS / obs=$OBS_TOYS toys per σ)"

elif [[ "$SCAN" == "campaign" ]]; then
  # ── FULL campaign: fixed σ, loop N_test × { single + ENS × modes } × calib, FULL toys ──
  SIGMA=${SIGMA:-$SIGMA_DEF}
  [[ -z "$SIGMA" ]] && { echo "campaign mode (DIM=$DIM) needs SIGMA=<value> — run 'SCAN=sigma' first to pick it" >&2; exit 1; }
  NTEST_LIST=${NTEST_LIST:-"100 200 1000 2000 10000 20000 100000 200000"}
  ENS_LIST=${ENS_LIST:-"2 4 8 16 32 64 128"}
  MODES=${MODES:-"point composite"}       # point = frozen weights (plug-in) ; composite = constrained
  INCLUDE_SINGLE=${INCLUDE_SINGLE:-1}
  NULL_TOYS=${NULL_TOYS:-250}; OBS_TOYS=${OBS_TOYS:-100}
  STD_TIME=${STD_TIME:-"11:59:00"}; LONG_TIME=${LONG_TIME:-"1-00:00:00"}; LONG_ENS=128

  # resolve the uniform member seeds once per ENS (independent of N_test)
  declare -A SEEDS
  for ens in $ENS_LIST; do
    SEEDS[$ens]=$(resolve_seeds "$ens")
    [[ -z "${SEEDS[$ens]}" ]] && { echo "ERROR: could not resolve seeds for M=$ens (DIM=$DIM, n_members=$NMEM)" >&2; exit 1; }
  done

  for ntest in $NTEST_LIST; do
    mem=$(mem_for "$ntest")

    # ── single model (seed $SINGLE_SEED); ENS=1 ⇒ auto-frozen/point-null ──
    if [[ "$INCLUDE_SINGLE" == "1" ]]; then
      export MEMBER_SEEDS="$SINGLE_SEED"
      for calib in $CALIBS; do
        submit "$(array_for "$calib")" "$STD_TIME" "$mem" "ENS=1,CALIBRATION=$calib,NTEST=$ntest,KERNEL_SIGMA=$SIGMA"
        n=$((n+1))
      done
    fi

    # ── ensembles ──
    for ens in $ENS_LIST; do
      export MEMBER_SEEDS="${SEEDS[$ens]}"
      [[ "$ens" -eq "$LONG_ENS" ]] && etime="$LONG_TIME" || etime="$STD_TIME"
      for mode in $MODES; do
        [[ "$mode" == "point" ]] && fix="true" || fix="false"
        for calib in $CALIBS; do
          submit "$(array_for "$calib")" "$etime" "$mem" \
            "ENS=$ens,FIX_WIFI_WEIGHTS=$fix,CALIBRATION=$calib,NTEST=$ntest,KERNEL_SIGMA=$SIGMA"
          n=$((n+1))
        done
      done
    done
  done
  echo "# $n array-submissions [DRYRUN=$DRYRUN]  (DIM=$DIM campaign at σ=$SIGMA; null=$NULL_TOYS / test=$OBS_TOYS toys/cell; ENS=$LONG_ENS → $LONG_TIME on partition gpu, others → $STD_TIME)"

else
  echo "unknown SCAN=$SCAN (want campaign|sigma)" >&2; exit 1
fi
