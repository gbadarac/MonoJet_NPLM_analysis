#!/bin/bash
#SBATCH --job-name=LRT
#SBATCH --array=0-99
#SBATCH --time=11:59:00
#SBATCH --mem=12G          # peak RSS observed ~8 G on these (<=100k) jobs; 12 G = ~1.5x margin
#SBATCH --ntasks=1
#SBATCH --account=gpu_gres
#SBATCH --partition=qgpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -o /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs/%x-%A_%a.out
#SBATCH -e /work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/results/logs/%x-%A_%a.err
# Resource defaults above are GPU (qgpu,gpu / gpu_gres / --gres=gpu:1) because NF (torch) needs
# the GPU. KERNELS run on CPU (density eval is single-threaded numpy; a GPU would sit idle and
# throttle the campaign to the cluster's ~16 GPUs). For a KERNELS run, OVERRIDE back to CPU:
#   --partition=standard --account=t3 --gres=none
# (run_lrt_scan.sh drives the kernels scan -> add the CPU override there before a kernels campaign.)

set -euo pipefail

# ─── Scenario × Model (EDIT THESE) ───────────────────────────────────
# TWO orthogonal switches select the run:
#   MODEL    = kernels | nf                              — the model type (its own axis).
#   SCENARIO = 1model_2d | ens_2d | 1model_4d | ens_4d   — progression stage × dimension.
# The single-model preset is just member seed000 of the matching ensemble dir (NENSEMBLE=1),
# so 1model and ens are nested — only OUT_BASE differs. Kernel paths use the seed-42-aligned
# data (2d_gmm / 4d_embedding_qcd).
#   1model_2d / 1model_4d : single model (ensemble-of-one, frozen). NENSEMBLE=1 -> LRT.py
#       synthesizes w=[1.0], cov=0x0 (no wifi fit); W_PATH/W_COV_PATH stay empty and are NOT
#       passed; forces FIX_WIFI_WEIGHTS=true (frozen; below).
#   ens_2d / ens_4d       : ensemble. ONE training dir per dim; ENS={2,4,...} picks the size
#       (-e = first-ENS subset) AND the matching wifi fit (ensemblecomponents$ENS).
# Implemented MODEL:SCENARIO combos: kernels:{1model_2d,ens_2d,1model_4d,ens_4d}, nf:ens_2d
# (the NF leg — add more nf combos as they are produced). Anything else errors in the case below.
#   sbatch --export=ALL,SCENARIO=1model_2d,CALIBRATION=1 submit_LRT_toys.sh          # kernels (default)
#   sbatch --export=ALL,MODEL=nf,SCENARIO=ens_2d,CALIBRATION=1 submit_LRT_toys.sh    # nf
SCENARIO=${SCENARIO:-1model_2d}
MODEL=${MODEL:-kernels}

CALIBRATION=${CALIBRATION:-1}              # 1 = null toys (SIR + calib pool)  |  0 = observed (target data)
# FIX_WIFI_WEIGHTS=true -> frozen (w=w_hat); default false -> constrained (N(w_hat,Sigma) prior).
FIX_WIFI_WEIGHTS=${FIX_WIFI_WEIGHTS:-false}   # respects sbatch --export override

NTEST=${NTEST:-100000}
# M = # numerator kernels (--n_kernels). NPLM rule (Grosso-Letizia 2408.12296): M >= sqrt(N_test)
# (=316 for 100k, 447 for 200k). Default 300 = principled for the 100k base (old 500 was an
# overshoot per the M_sigma memory); held FIXED across the Ntest scan (not scaled) so DOF_eff is
# comparable. M is in the run_tag (_M%i_) so different M land in DISTINCT dirs -> no silent mixing.
# (run_lrt_scan.sh passes N_KERNELS=$M explicitly, so this default only affects ad-hoc launches.)
N_KERNELS=${N_KERNELS:-300}
# Z = ∫ f_ens(x)·exp(Σ b_j G_j(x)) dx is a genuine 2-SAMPLE test: the integral is estimated from
# a REFERENCE sample (importance-sampled from q). N_REF = # reference events, the 2-sample partner
# of NTEST. DEFAULT 1:1 (N_REF=NTEST) — Sean's choice (Slack, 2026-08): more N_ref only pulls the
# already-asymptotic null closer to chi2, but we CALIBRATE WITH TOYS anyway, so 1:1 is enough;
# extra N_ref just costs compute (esp. NF flow.sample x128). Bump (>>NTEST) only if a CLOSURE
# needs MC-Z suppression. The N_ref/N_test ratio is recorded in the run_tag (_Nrefx%g).
N_REF=${N_REF:-$NTEST}
FIRSTSEED=12345
# MEMBER_SEEDS: comma-separated member indices to load, IN ORDER (last=norm), passed to
# LRT.py --member_seeds. REQUIRED for the uniform/pinned wifi (the ensemble is a NON-first-k
# subset, so the models must match the seeds the wifi fit used). Resolve with
# select_member_seeds.py --rng_seed <same as the wifi fit>. Empty = first-k (legacy).
MEMBER_SEEDS=${MEMBER_SEEDS:-}

# ─── Numerator knobs (EDIT / OVERRIDE AT SUBMIT TIME) ────────────────
# The numerator is the NPLM exp-tilt  f_num = f_ens*exp(Σ b_j G_j)/Z  (log-space, clean chi2;
# matches LRT_one_model + classifier). FIX_WIFI_WEIGHTS=true => frozen w (convex); default
# (false) => constrained (profile w under the N(w_hat,Sigma) prior).
#   sbatch --export=ALL,FIX_WIFI_WEIGHTS=true,CALIBRATION=1 submit_LRT_toys.sh
KERNEL_SIGMA=${KERNEL_SIGMA:-0.4}   # embedding-dependent: 0.4 (2D & old-4D) | 0.7 (gaussian-4D)
LAM_PERT=${LAM_PERT:-1.0}           # L2 ridge on tilt coeffs b (= one-model/classifier)
CLIP_B=${CLIP_B:-3}                 # box |b_j|<=CLIP_B on tilt coeffs. NON-EMPTY BY DEFAULT (crash fix):
                                    # clip=None makes fit_nplm_tilt use scipy trust-exact, which Choleskys
                                    # the tilt Hessian; on a ~0.1% tilt runaway (separation) H goes non-finite
                                    # -> cho_solve "infs or NaNs" CRASH (job dies, writes nothing) before the
                                    # runaway guard can flag it. Setting clip switches fit_nplm_tilt to L-BFGS-B
                                    # (no Cholesky -> crash impossible) AND bounds b. Good fits have max|b|<=2.9
                                    # (p99=1.23) so 3 sits above them -> bounds runaways, distorts no good fit.
                                    # ⚠ clip CHANGES the estimator: keep it FIXED across a config's null+test;
                                    # do NOT mix clipped T's into an unclipped null/test dist. empty=off (validation).
Z_MODE=${Z_MODE:-sample}            # Z: sample (importance from q; any d, DEFAULT) | grid (2D cross-check)
GRID_POINTS=${GRID_POINTS:-300}     # [grid] points per dimension
GRID_PAD=${GRID_PAD:-0.2}           # [grid] padding beyond data range
W_COV_SCALE=${W_COV_SCALE:-1.0}     # [constrained] diagnostic scale on Sigma_w; <1 -> frozen limit (separate out dir)
# 4D: set Z_MODE=sample + (usually) FIX_WIFI_WEIGHTS=true; the null is self-built (SIR-from-q).

# ─── Per-scenario paths (EDIT THESE) ─────────────────────────────────
REPO_ROOT="/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
PY="$REPO_ROOT/LRT/LRT.py"

# defaults; the MODEL case sets env, each scenario overrides its paths (empties stay empty -> not passed)
MODEL_TYPE="$MODEL"; NENSEMBLE=""; PYBIN="python"   # PYBIN: NF (MODEL case) overrides to nplm_env absolute python
ENSEMBLE_DIR=""; NF_TRAIN_DIR=""; ARCH_CONFIG=""
W_PATH=""; W_COV_PATH=""; TARGET_DATA=""; OUT_BASE=""
# TARGET_TRUTH: observed run (calib=0) draws FRESH from this analytic truth each toy
# instead of bootstrapping TARGET_DATA. Set by the 2D scenarios (same truth for kernels
# and nf). Leave empty for 4D (no analytic truth -> bootstrap the finite holdout).
TARGET_TRUTH=""

# MODEL (kernels | nf) is orthogonal to the 1model/ens × 2d/4d scenario, so it is its OWN
# switch: it sets the conda env (and, for nf, the absolute python + the env-block
# LD_LIBRARY_PATH skip below). The SCENARIO case then fills in the model-specific paths.
case "$MODEL" in
  kernels) CONDA_ENV=kernels_env ;;
  nf)      CONDA_ENV=nplm_env
           PYBIN="/work/gbadarac/miniforge3/envs/nplm_env/bin/python" ;;
  *) echo "Unknown MODEL=$MODEL (expected: kernels | nf)"; exit 1 ;;
esac

# Only the MODEL:SCENARIO combos below are implemented (kernels: all four; nf: ens_2d).
case "$MODEL:$SCENARIO" in
  kernels:1model_2d)
    NENSEMBLE=1
    # -e 1 uses seed000 of the 2d_gmm ensemble dir as the single model; no wifi files (synthesized)
    ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_gmm/N_100000_dim_2_kernels_SparKer_models1_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/kernels/2d_single_model"
    ;;

  kernels:ens_2d)
    # 2d_gmm (seed-42 aligned). ONE 128-member training dir serves every size; ENS picks how
    # many members (-e = first-ENS subset) AND the matching wifi fit (ensemblecomponents$ENS).
    NENSEMBLE=${ENS:-64}
    ENSEMBLE_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Sparker_kernels/EstimationKernels_outputs/2_dim/2d_gmm/N_100000_dim_2_kernels_SparKer_models128_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035"
    WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_kernels/2d_toymodel/N_100000_dim_2_kernels_SparKer_models128_L5_K80_M300_Nboot100000_lr0.05_clip_10000000_joint_norm_wf0.15-0.035_2d_gmm_ensemblecomponents${NENSEMBLE}"
    W_PATH="$WIFI_DIR/w_i_fitted.npy"
    W_COV_PATH="$WIFI_DIR/cov_w.npy"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/kernels/2d_ensemble"
    ;;

  nf:1model_2d)
    NENSEMBLE=1
    # -e 1 uses seed000 of the 2d_gmm ensemble dir as the single model; no wifi files (synthesized)
    ENSEMBLE_DIR="$REPO_ROOT//work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_1_4_4_64_8"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/kernels/2d_single_model"
    ;;

  nf:ens_2d)
    # NF recipe (validated 08-17): the conda env + absolute python are set in the MODEL case
    # above, and the env block below SKIPS LD_LIBRARY_PATH for nf (prepending env/lib breaks
    # torch _C). IS loader: model_%03d/model.pth under NF_TRAIN_DIR (f_i.pth retired); null via
    # SIR-from-q (flow.sample). SIZE SCAN: set ENS + the matching wifi ensemblecomponents
    # {2,4,...,128} (smaller = first-k subset of the same 128-member training dir).
    NENSEMBLE=${ENS:-128}
    NF_TRAIN_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_128_4_4_64_8"
    ARCH_CONFIG="$NF_TRAIN_DIR/architecture_config.json"
    NF_WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/2d_toymodel/N_100000_dim_2_seeds_128_4_4_64_8_2d_bimodal_gaussian_heavy_tail_ensemblecomponents${NENSEMBLE}"
    W_PATH="$NF_WIFI_DIR/w_i_fitted.npy"
    W_COV_PATH="$NF_WIFI_DIR/cov_w.npy"
    # Same 2D truth as kernels (identical DGP) — observed draws fresh from it, no data file.
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/nf/2d_ensemble"
    ;;

  nf:ens_2d_uniform)
    # NF campaign on the UNIFORM-selected wifi ensembles (results_fit_weights_NF_uniform). The
    # ensemble members are a NON-first-k random subset, so the driver MUST pass MEMBER_SEEDS
    # (resolved via select_member_seeds.py --rng_seed 0) so LRT.py loads the SAME members, in the
    # SAME order, that the wifi fit used. Single model (ENS=1) = the pinned NF single (seed 100):
    # driver passes MEMBER_SEEDS=100; the ensemblecomponents1 wifi is ignored (M=1 synthesizes w).
    NENSEMBLE=${ENS:-128}
    NF_TRAIN_DIR="$REPO_ROOT/Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_128_4_4_64_8"
    ARCH_CONFIG="$NF_TRAIN_DIR/architecture_config.json"
    NF_WIFI_DIR="$REPO_ROOT/Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF_uniform/2d_toymodel/N_100000_dim_2_seeds_128_4_4_64_8_2d_bimodal_gaussian_heavy_tail_ensemblecomponents${NENSEMBLE}"
    W_PATH="$NF_WIFI_DIR/w_i_fitted.npy"
    W_COV_PATH="$NF_WIFI_DIR/cov_w.npy"
    TARGET_TRUTH="2d_gmm_skew"   # observed: draw fresh from the analytic truth each toy (no data file)
    OUT_BASE="$REPO_ROOT/LRT/results/nf/2d_uniform_ensemble"
    ;;

  *)
    echo "Unsupported MODEL/SCENARIO combo: MODEL=$MODEL SCENARIO=$SCENARIO"
    echo "  implemented -> kernels: 1model_2d | ens_2d | 1model_4d | ens_4d ; nf: ens_2d | ens_2d_uniform"
    exit 1 ;;
esac

# A single model (NENSEMBLE=1) has no wifi weights to profile, so only the frozen fit is valid.
if [[ "$NENSEMBLE" -eq 1 ]]; then
    [[ "$FIX_WIFI_WEIGHTS" != "true" ]] && echo "[single model] forcing FIX_WIFI_WEIGHTS=true (frozen)"
    FIX_WIFI_WEIGHTS=true
fi

# ─── Environment ─────────────────────────────────────────────────────
. /work/gbadarac/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Pin BLAS/OMP to 1 thread. SLURM gives each job 1 CPU, but numpy's OpenBLAS otherwise
# spawns 64 threads -> when many jobs land on the same CPU node they oversubscribe it
# (dozens x 64 threads on 128 cores), thrash, and the heavy toys blow past the 8h wall
# (the calib=1 TIMEOUTs on constr/200k e128). 1 thread = 1 core = no contention; the
# kernel eval is a single-threaded python loop anyway, so this doesn't slow the fit.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ "$MODEL_TYPE" == "nf" ]]; then
  # NF recipe (validated 08-17): use nplm_env ABSOLUTE python ($PYBIN) and DO NOT touch
  # LD_LIBRARY_PATH — prepending env/lib breaks torch _C (ByteStorageBase). torch 1.11+cu115
  # in nplm_env finds CUDA via its bundled libs. (conda activate can't change `which python`.)
  :
else
  export LD_LIBRARY_PATH="/work/gbadarac/miniforge3/envs/nplm_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  CUDA_PATH=$("$PYBIN" - <<'PY'
import torch; print(torch.version.cuda or "")
PY
)
  if [[ -n "$CUDA_PATH" ]]; then
    export CUDA_HOME=/usr/local/cuda-$CUDA_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# ─── Run ─────────────────────────────────────────────────────────────
mkdir -p "$OUT_BASE" "$REPO_ROOT/LRT/results/logs"

TOY_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((FIRSTSEED + TOY_ID + 1))

cd "$REPO_ROOT"
echo "[$(date)] MODEL_TYPE=${MODEL_TYPE}, CALIBRATION=${CALIBRATION}, Task=${TOY_ID}, seed=${SEED}, host=${HOSTNAME}"

CMD=("$PYBIN" -u "$PY"
  --model_type    "$MODEL_TYPE"
  --out_base      "$OUT_BASE"
  -n              "$NTEST"
  -e              "$NENSEMBLE"
  -s              "$SEED"
  --toy_id        "$TOY_ID"
  -c              "$CALIBRATION"
  --n_kernels     "$N_KERNELS"
  --kernel_sigma  "$KERNEL_SIGMA"
  --lam_pert      "$LAM_PERT"
  --z_mode        "$Z_MODE"
  --grid_points   "$GRID_POINTS"
  --grid_pad      "$GRID_PAD"
  --n_ref         "$N_REF"
)

# wifi weights: only for a real ensemble (NENSEMBLE>1); a single model synthesizes them
[[ -n "$W_PATH" ]] && CMD+=(--w_path "$W_PATH" --w_cov_path "$W_COV_PATH")

if [[ "$MODEL_TYPE" == "kernels" ]]; then
    CMD+=(--ensemble_dir "$ENSEMBLE_DIR" --seed_format "seed%03d")
elif [[ "$MODEL_TYPE" == "nf" ]]; then
    CMD+=(--nf_train_dir "$NF_TRAIN_DIR" --arch_config "$ARCH_CONFIG")
fi

[[ -n "$CLIP_B" ]] && CMD+=(--clip_b "$CLIP_B")
[[ -n "$MEMBER_SEEDS" ]] && CMD+=(--member_seeds "$MEMBER_SEEDS")
[[ "$W_COV_SCALE" != "1.0" ]] && CMD+=(--w_cov_scale "$W_COV_SCALE")
[[ "$FIX_WIFI_WEIGHTS" == "true" ]] && CMD+=(--fix_wifi_weights)

# calib=1 null is self-contained (SIR-from-q). calib=0 observed: 2D toy draws FRESH from
# the analytic truth each toy (--target_truth: genuine oversampling, no bootstrap, no
# replace=True); 4D (no truth) bootstraps the finite --target_data holdout.
if [[ "$CALIBRATION" -eq 0 ]]; then
  if [[ -n "$TARGET_TRUTH" ]]; then
    CMD+=(--target_truth "$TARGET_TRUTH")
  else
    CMD+=(--target_data "$TARGET_DATA")
  fi
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
RC=$?
echo "[$(date)] Task ${TOY_ID} finished with exit code ${RC}"
exit $RC
