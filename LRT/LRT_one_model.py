import glob, math, time, os, json, argparse, datetime, sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist

# -------------------------------------------------------------------
# Make Sparker_utils importable
# -------------------------------------------------------------------
THIS_DIR  = Path(__file__).resolve().parent    # .../LRT
REPO_ROOT = THIS_DIR.parent                    # .../MonoJet_NPLM_analysis
SPARKER_UTILS = REPO_ROOT / "shared" / "Sparker_utils"
sys.path.insert(0, str(SPARKER_UTILS))

import LRTGOFutils_v2 as lrt
import GENutils as gen

# ===================================================================
# One-model kernel GoF — multiplicative NPLM exp-tilt numerator (Sean Option 1).
#
# The single SParKer model is trained as a positive, sum-normalized mixture, so its
# component weights live on the probability simplex. The DENOMINATOR fits them to the
# data by EM (fit_simplex_em):   DEN:  f = sum_k w_k g_k,  w on the K-simplex.
#
# The NUMERATOR multiplies the DEN-optimal model by an exponential tilt, the standard
# NPLM form (= wifi_better_basis/classifier_gof.py):
#     NUM: f = f_mix(x; w_den) * exp(sum_j b_j G_j(x)) / Z,   Z = E_{f_mix}[exp tau].
# The shared log f_mix term cancels and
#     T = 2 * max_b [ sum_data tau(x_i) - N * log Z ].
# The normalization Z is computed two ways (--z_mode):
#   sample : MC average over a reference drawn from the DEN-optimal model. This is
#            two-sample-LOOKING (finite R broadens the null); Z(0)=1 by construction.
#   grid   : deterministic quadrature over a fixed grid, a_r = log f_mix(y_r). This is
#            a GENUINE one-sample test (no reference sample, no MC noise), exact in low
#            d -> use in 2D; infeasible in high d (grid_points**d), use sample in 4D.
# With w FROZEN at w_den (only b fit) this is CONVEX (linear - N*log-sum-exp, PSD
# Hessian) -> robust, always f>0, T >= 0 (b=0 recovers DEN). Solved by
# lrt.fit_nplm_tilt (scipy trust-exact + closed-form grad/Hessian). An L2 ridge
# lam_pert on b sets the effective DOF.
#
# (The earlier additive positive-simplex numerator (Sean Option 2) was removed: its
# positivity constraint gives a degenerate, over-sensitive null; the multiplicative
# form gives a clean Wilks chi^2 null and matches the classifier. fit_simplex_em is
# still used for the DENOMINATOR above.)
# ===================================================================

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True,
                    help="Dir with config.json and seed*/ histories (Train_Ensembles output).")
parser.add_argument('--model_seed', type=int, required=True,
                    help="Index of the single model to use (e.g. 0 -> seed000).")
parser.add_argument('--out_base', type=str, required=True,
                    help="Base output dir (e.g. .../LRT/Sparker_kernels/results).")
parser.add_argument('--seed_format', type=str, default="seed%03d",
                    help="Seed folder format: seed%03d for seed000, seed%01d for seed0.")
parser.add_argument('--target_data', type=str, default=None,
                    help="Target/ground truth data: single .npy file. Required for CALIBRATION=0.")
parser.add_argument('-n', '--ntest', type=int, required=True,
                    help="Number of points in the test data.")
parser.add_argument('-c', '--calibration', type=int, required=True,
                    help="1 = calibration toy (sample from single model), 0 = test on target data.")
parser.add_argument('-s', '--seed', type=int, default=None,
                    help="Toy seed.")
parser.add_argument('--toy_id', type=int, default=None,
                    help="Toy index used for folder/file naming (0-based). Falls back to seed.")
parser.add_argument('--save_arrays', action='store_true',
                    help="If set, also save per-event numerator/denominator/test arrays.")
parser.add_argument('--z_mode', type=str, default='sample', choices=['sample', 'grid'],
                    help="Normalization Z for the numerator. 'sample' = MC over a "
                         "reference drawn from the DEN-optimal model (two-sample-looking; "
                         "finite-R broadens the null). 'grid' = deterministic quadrature "
                         "over a fixed grid (genuine one-sample; exact in low d -> use in "
                         "2D, infeasible in high d).")
parser.add_argument('--grid_points', type=int, default=300,
                    help="[z_mode=grid] points per dimension (total = grid_points**d).")
parser.add_argument('--grid_pad', type=float, default=0.2,
                    help="[z_mode=grid] fractional padding beyond the data range per dim.")
parser.add_argument('--n_ref', type=int, default=None,
                    help="[z_mode=sample] # reference points ~ DEN-optimal model "
                         "for the normalization estimate (default: Ntest).")
parser.add_argument('--lam_pert', type=float, default=1.0,
                    help="[multiplicative] L2 ridge 0.5*lam*||b||^2 on tilt coeffs "
                         "(= classifier_gof lam_pert). lam_pert=0 SEPARATES (tilt "
                         "runaway even under the null: max|b|~1e5 in 2D); 1.0 keeps "
                         "max|b|<1. Must be > 0.")
parser.add_argument('--clip_b', type=float, default=None,
                    help="[multiplicative] optional symmetric box |b_j| <= clip_b.")
parser.add_argument('--n_kernels', type=int, default=100,
                    help="Number M of exp-tilt kernels in the numerator (centred on the "
                         "first M data points). NPLM guideline: M >= sqrt(N_train) (~316 "
                         "for 100k) for a good kernel approximation; M below that can make "
                         "the test under-powered. Was hardcoded to 100.")
parser.add_argument('--kernel_sigma', type=float, default=0.3,
                    help="Isotropic width sigma of the numerator tilt kernels. Drives test "
                         "power; keep FIXED across calibration and test (a per-toy value "
                         "biases the Z estimate). candidate_sigma anchor is printed at run "
                         "start. Was hardcoded to 0.3.")
args = parser.parse_args()

# -------------------------------------------------------------------
# Seed / label
# -------------------------------------------------------------------
seed = args.seed
if seed is None:
    seed = (datetime.datetime.now().microsecond
            + datetime.datetime.now().second
            + datetime.datetime.now().minute)
print('Random seed:', seed)
np.random.seed(seed)

label = args.toy_id if args.toy_id is not None else seed

# -------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------
Ntest                  = args.ntest
n_kernels_numerator    = args.n_kernels
kernel_width_numerator = args.kernel_sigma

# -------------------------------------------------------------------
# Output folder
# -------------------------------------------------------------------
mode_tag = "calibration" if args.calibration else "test"

seed_fmt = args.seed_format
run_tag = "SparKer1_%s_Ntest%i_M%i_W%s" % (
    seed_fmt % args.model_seed,
    Ntest,
    n_kernels_numerator,
    str(kernel_width_numerator),
)
# Multiplicative NPLM exp-tilt (convex fit); folder tagged with the L2 ridge.
run_tag += "_num_multiplicative"
if args.lam_pert > 0:
    run_tag += "_L%g" % args.lam_pert
if args.clip_b is not None:
    run_tag += "_clipb%s" % str(args.clip_b)
# 'sample' keeps the historical folder names; tag only the new deterministic grid-Z.
if args.z_mode != 'sample':
    run_tag += "_zgrid"

out_dir = os.path.join(args.out_base, run_tag, mode_tag, "seed%i" % label)
os.makedirs(out_dir, exist_ok=True)
print("Writing outputs to:", out_dir, flush=True)

# -------------------------------------------------------------------
# Load single model
# -------------------------------------------------------------------
with open(os.path.join(args.model_dir, "config.json"), "r") as f:
    config_json = json.load(f)

seed_dir = os.path.join(args.model_dir, seed_fmt % args.model_seed)
tmp = np.load(os.path.join(seed_dir, "widths_history.npy"))
count = -1
for j in range(tmp.shape[0]):
    if tmp[j][0].sum():
        count += 1
    else:
        print(count)
        break

centroids    = np.load(os.path.join(seed_dir, "centroids_history.npy"))[count]       # (K, d)
coefficients = np.load(os.path.join(seed_dir, "coeffs_history.npy"))[count]          # (K,)
widths       = np.load(os.path.join(seed_dir, "widths_history.npy"))[count, :, 0]    # (K,)

# Normalize coefficients to a proper probability vector (already non-negative
# from training; this puts them on the simplex, sum = 1).
coefficients = coefficients / coefficients.sum()

print(f"Loaded model {seed_fmt % args.model_seed}: "
      f"centroids {centroids.shape}, coefficients {coefficients.shape}, widths {widths.shape}",
      flush=True)
print(f"  trained coeffs: min={coefficients.min():.3e}  n_neg={(coefficients < 0).sum()}", flush=True)

# -------------------------------------------------------------------
# Helper: sample N points directly from the single GMM
# -------------------------------------------------------------------
def sample_from_gmm(centroids, coefficients, widths, n_samples, rng):
    K, d = centroids.shape
    k_indices = rng.choice(K, size=n_samples, p=coefficients)
    noise = rng.standard_normal((n_samples, d))
    samples = centroids[k_indices] + widths[k_indices, np.newaxis] * noise
    return samples.astype(np.float32)

def candidate_sigma(data, perc=90, n_sub=2000):
    """NPLM bandwidth heuristic (= FLKutils_model.candidate_sigma): the perc-th
    percentile of pairwise distances on a subsample. Informational anchor only —
    pass a FIXED value via --kernel_sigma; a per-toy data-driven sigma would differ
    between calibration and test and bias the Z estimate."""
    sub = np.asarray(data[:n_sub], dtype=np.float64)
    return float(np.around(np.percentile(pdist(sub), perc), 1))

# -------------------------------------------------------------------
# Load / generate test data
# -------------------------------------------------------------------
if args.calibration:
    rng = np.random.default_rng(seed=seed)
    data_all = sample_from_gmm(centroids, coefficients, widths, Ntest, rng)
    bootstrap_sample = data_all          # already exactly Ntest points
    print(f"Generated {bootstrap_sample.shape[0]} calibration samples from GMM.", flush=True)
else:
    if args.target_data is None:
        raise ValueError("calibration=0 but --target_data not provided.")
    data_all = np.load(args.target_data)
    print(f"Loaded {data_all.shape[0]} target data points.", flush=True)
    # replace=True (bootstrap): with replace=False and Ntest == file size every
    # "toy" is the identical dataset.
    idx = np.random.choice(len(data_all), Ntest, replace=True)
    bootstrap_sample = data_all[idx]

N = bootstrap_sample.shape[0]

# NPLM candidate_sigma anchor for choosing --kernel_sigma (informational; see helper).
sigma_anchor = candidate_sigma(bootstrap_sample)
print(f"[sigma anchor] NPLM candidate_sigma(perc=90) on this data = {sigma_anchor:.3f}; "
      f"running with kernel_sigma={kernel_width_numerator}, M={n_kernels_numerator} "
      f"(sqrt(N)={math.sqrt(N):.0f}). Keep kernel_sigma FIXED across calib+test.",
      flush=True)

# -------------------------------------------------------------------
# Component densities.
#   comps  (N, K) : the K trained Gaussian components g_k.
#   Kmat   (N, M) : M numerator kernels G_j (fixed shape, centred at the first
#                   M data points), same normalized Gaussians as GENutils.
# Both are strictly positive -> any simplex mixture of them is a positive density.
# -------------------------------------------------------------------
comps = gen.evaluate_gaussian_components(bootstrap_sample, centroids, widths)  # (N, K)
K = comps.shape[1]

centers = bootstrap_sample[:n_kernels_numerator].astype(np.float64)
Kmat = lrt.gaussian_kernel_matrix(bootstrap_sample, centers, kernel_width_numerator)  # (N, M)
M = Kmat.shape[1]

print("comps shape:", comps.shape, " min:", float(comps.min()), " max:", float(comps.max()), flush=True)
f_init = comps @ coefficients
print("model density at init: min", float(f_init.min()), " (must be > 0)", flush=True)
if f_init.min() <= 0:
    raise RuntimeError("Model density not strictly positive at the data at init.")

# -------------------------------------------------------------------
# Denominator: EM over the K model components (weights on the K-simplex).
# w_den is also the reference model for the numerator; its log-likelihood
# cancels via the reference-sample Z estimate.
# -------------------------------------------------------------------
res_den = lrt.fit_simplex_em(comps, theta_init=coefficients, name="DEN")
if not res_den["converged"]:
    print(f"WARNING: DEN EM hit max_iter without converging (loglik={res_den['loglik']:.3f})",
          flush=True)
if res_den["fmin"] <= 0:
    raise RuntimeError(f"DEN density non-positive (fmin={res_den['fmin']:.3e}) — "
                       "impossible for a simplex mixture.")

w_den        = res_den["theta"]          # (K,)
den_log_data = np.log(res_den["f"])      # (N,)

# ---------------------------------------------------------------
# NUM: multiplicative NPLM exp-tilt. The shared log f_mix term cancels, so
# T = 2*max_b[ sum_data tau - N*log Z ]. Convex (w frozen). Z is computed over
# reference points y_r with log-weights a_r (sum exp(a_r)=1):
#   sample : y_r ~ DEN-optimal model, uniform a_r = -log R (MC; two-sample-looking).
#   grid   : y_r on a fixed grid, a_r = log f_mix(y_r) (deterministic; one-sample).
# ---------------------------------------------------------------
if args.z_mode == 'sample':
    n_ref = args.n_ref if args.n_ref is not None else Ntest
    rng_ref = np.random.default_rng(seed=seed + 987654321)   # independent stream
    ref_samples = sample_from_gmm(centroids, w_den, widths, n_ref, rng_ref)
    K_ref = lrt.gaussian_kernel_matrix(ref_samples, centers, kernel_width_numerator)  # (R, M)
    log_w_ref = None
    print(f"Z via SAMPLE: {n_ref} refs ~ DEN-optimal model; K_ref {K_ref.shape}", flush=True)
else:  # grid: deterministic quadrature -> genuine one-sample test
    d = bootstrap_sample.shape[1]
    G_total = args.grid_points ** d
    if G_total > 4_000_000:
        raise ValueError(f"grid too large: {args.grid_points}**{d} = {G_total} points. "
                         "Grid-Z is for low d (2D); use --z_mode sample in high d.")
    lo = bootstrap_sample.min(axis=0).astype(np.float64)
    hi = bootstrap_sample.max(axis=0).astype(np.float64)
    span = hi - lo
    lo = lo - args.grid_pad * span
    hi = hi + args.grid_pad * span
    # ensure the model's own support is covered (centroids +/- 5 widths)
    cmin = (centroids - 5.0 * widths[:, None]).min(axis=0)
    cmax = (centroids + 5.0 * widths[:, None]).max(axis=0)
    lo = np.minimum(lo, cmin)
    hi = np.maximum(hi, cmax)
    axes = [np.linspace(lo[k], hi[k], args.grid_points) for k in range(d)]
    mesh = np.meshgrid(*axes, indexing='ij')
    grid = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float64)   # (G, d)
    step = (hi - lo) / (args.grid_points - 1)
    comps_grid = gen.evaluate_gaussian_components(grid, centroids, widths)  # (G, K)
    f0_grid = comps_grid @ w_den                                           # (G,)
    K_ref = lrt.gaussian_kernel_matrix(grid, centers, kernel_width_numerator)  # (G, M)
    log_w_ref = np.log(np.maximum(f0_grid, 1e-300))   # self-normalized in fit_nplm_tilt
    grid_mass = float((f0_grid * np.prod(step)).sum())
    print(f"Z via GRID: {args.grid_points}/dim, {grid.shape[0]} pts, step={np.array2string(step, precision=3)}, "
          f"sigma={kernel_width_numerator}; grid f0 mass = {grid_mass:.4f} (should be ~1)", flush=True)
    if not (0.95 <= grid_mass <= 1.05):
        print(f"WARNING: grid f0 mass = {grid_mass:.4f} far from 1 -> grid may not cover "
              f"the model support; widen extent or raise grid_points.", flush=True)

res_num = lrt.fit_nplm_tilt(Kmat, K_ref, lam_pert=args.lam_pert, clip=args.clip_b,
                            log_w_ref=log_w_ref, name="NUM", verbose=True)
if not res_num["converged"]:
    print(f"WARNING: NUM tilt fit not converged (||g||={res_num['grad_norm']:.2e})",
          flush=True)

b_num        = res_num["b"]                          # (M,) signed tilt coeffs
logZ         = res_num["logZ"]
tau_data     = res_num["tau_data"]                   # (N,)
num_log_data = den_log_data + tau_data - logZ        # log f_num at the data
test         = 2.0 * (tau_data - logZ)               # per-event 2*logLR
# T = the PURE (unpenalized) 2*log-likelihood ratio (= 2*(loglik_num - loglik_den)),
# equal to sum(test). The L2 ridge only shapes the fit (bounds b); it is NOT part
# of the reported statistic. NOTE: res_num["T"] = 2*ll is the PENALIZED value
# (= T - lam*||b||^2) and would break the sum(test)==T identity for lam_pert>0 —
# do NOT use it here.
T            = 2.0 * (num_log_data.sum() - den_log_data.sum())
assert abs(T - test.sum()) < 1e-4 * (1.0 + abs(T)), (T, float(test.sum()))

w_num      = w_den                   # frozen in the numerator (w_num == w_den)
coeffs_out = b_num                   # saved as coeffs.npy (signed tilt coeffs)
# keep max_b in the report to monitor tilt runaway across the toy array
num_report = {k: res_num[k] for k in ("loglik", "n_iter", "converged",
                                       "hit_max_iter", "grad_norm", "max_b")}

assert T >= -1e-4, f"T = {T} < 0: nested LRT violated — a fit did not converge."

print(f"T = {T:.6f}", flush=True)
print(f"mean per-event log LR = {float(test.mean()):.6f}", flush=True)

# -------------------------------------------------------------------
# Save outputs. Same filenames as LRT.py where sensible so analyse_LRT_output.py
# keeps working. Saved "coeffs" = the signed exp-tilt coefficients b, and
# num_weights == den_weights (w is frozen in the numerator).
# -------------------------------------------------------------------
with open(os.path.join(out_dir, f"seed{label}_T.txt"), "w") as f:
    f.write(f"{T}\n")
np.save(os.path.join(out_dir, f"seed{label}_T.npy"), np.array(T, dtype=np.float64))

if args.save_arrays:
    np.save(os.path.join(out_dir, f"seed{label}_test.npy"),        test)
    np.save(os.path.join(out_dir, f"seed{label}_numerator.npy"),   num_log_data)
    np.save(os.path.join(out_dir, f"seed{label}_denominator.npy"), den_log_data)

np.save(os.path.join(out_dir, f"seed{label}_coeffs.npy"), coeffs_out)
np.save(os.path.join(out_dir, f"seed{label}_kernel_centers.npy"), centers)
np.save(os.path.join(out_dir, f"seed{label}_den_weights.npy"),  w_den)
np.save(os.path.join(out_dir, f"seed{label}_num_weights.npy"),  w_num)
np.save(os.path.join(out_dir, f"seed{label}_init_weights.npy"), coefficients)

fit_report = {"den": {k: res_den[k] for k in ("loglik", "n_iter", "converged",
                                              "hit_max_iter", "fmin", "n_active")},
              "num": num_report}
with open(os.path.join(out_dir, f"seed{label}_fit_report.json"), "w") as f:
    json.dump(fit_report, f, indent=2)

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print(f"--- Denominator mixture weights ---")
print(f"  DEN: {K} model components, sum(w)={w_den.sum():.6f}, min={w_den.min():.3e}, "
      f"active={int((w_den>1e-8).sum())}")
print(f"--- Numerator (multiplicative NPLM exp-tilt) ---")
print(f"  w frozen at w_den; {M} tilt coeffs b, max|b|={np.abs(b_num).max():.3e}, "
      f"logZ={logZ:.4f}")
