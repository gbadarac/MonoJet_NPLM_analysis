import gc, os, json, argparse, sys, ctypes, math, datetime
from pathlib import Path
import torch
import numpy as np


def _np2t(arr, dtype=torch.float64):
    """numpy → tensor compatible with numpy 2.x + torch 1.x (ctypes fallback)."""
    arr_c = np.ascontiguousarray(arr, dtype=np.float64 if dtype == torch.float64 else np.float32)
    try:
        return torch.from_numpy(arr_c).to(dtype)
    except TypeError:
        t = torch.empty(arr_c.shape, dtype=dtype)
        ctypes.memmove(t.data_ptr(),
                       arr_c.ctypes.data_as(ctypes.c_void_p),
                       arr_c.nbytes)
        return t


def _t2np(t):
    """Safely convert torch tensor to plain numpy float64 array (numpy 2.x compatible)."""
    t_cpu = t.double().cpu().contiguous()
    arr = np.empty(t_cpu.shape, dtype=np.float64)
    ctypes.memmove(arr.ctypes.data_as(ctypes.c_void_p), t_cpu.data_ptr(), arr.nbytes)
    return arr


# -------------------------------------------------------------------
# Repo paths
# -------------------------------------------------------------------
THIS_DIR      = Path(__file__).resolve().parent          # .../LRT
REPO_ROOT     = THIS_DIR.parent                           # .../MonoJet_NPLM_analysis
SPARKER_UTILS = REPO_ROOT / "shared" / "Sparker_utils"
NF_UTILS      = REPO_ROOT / "Train_Ensembles" / "Train_Models"

sys.path.insert(0, str(SPARKER_UTILS))
sys.path.insert(0, str(NF_UTILS))

import LRTGOFutils_v2 as lrt
import GENutils as gen        # candidate_sigma (also re-used by the kernels block below)

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Model type
parser.add_argument('--model_type', type=str, required=True, choices=['kernels', 'nf'],
                    help="'kernels' = Sparker kernel ensemble; 'nf' = normalizing flow ensemble.")

# Kernels-specific
parser.add_argument('--ensemble_dir', type=str, default=None,
                    help="[kernels] Dir with config.json and seed*/ histories.")
parser.add_argument('--seed_format', type=str, default='seed%03d',
                    help="[kernels] Seed folder format, e.g. seed%03d.")

# NF-specific
parser.add_argument('--nf_train_dir', type=str, default=None,
                    help="[nf] Dir with model_%03d/model.pth members + architecture_config.json "
                         "(the training layout; mirrors the kernels --ensemble_dir). Preferred "
                         "loader; the f_i.pth bundle was retired.")
parser.add_argument('--fi_path', type=str, default=None,
                    help="[nf] LEGACY: single f_i.pth bundle of NF state dicts. Fallback only "
                         "if --nf_train_dir is not given.")
parser.add_argument('--arch_config', type=str, default=None,
                    help="[nf] Path to architecture_config.json (default: "
                         "<nf_train_dir>/architecture_config.json).")

# Common WiFi / data
parser.add_argument('--w_path', type=str, default=None,
                    help="Path to fitted WiFi weights .npy. Required for nensemble>1; "
                         "ignored for nensemble=1 (single model = ensemble-of-one, w=[1.0]).")
parser.add_argument('--w_cov_path', type=str, default=None,
                    help="Path to covariance of fitted WiFi weights .npy ((M-1)x(M-1)). "
                         "Required for nensemble>1; ignored for nensemble=1 (cov=0x0).")
parser.add_argument('--w_cov_scale', type=float, default=1.0,
                    help="Scale factor on Sigma_w (diagnostic). <1 shrinks the weight prior "
                         "toward frozen (Sigma_w->0 must reduce constrained T to frozen T); "
                         "!=1 appends _covscale%%g to the run_tag so it never clobbers the "
                         "nominal (scale=1) outputs.")
parser.add_argument('--member_seeds', type=str, default=None,
                    help="Comma-separated member indices to load, IN ORDER (kernels: seed<s:03d>/, "
                         "nf: model_<s:03d>/); LAST one is the norm model. MUST match the wifi "
                         "fit's --member_seeds so w_hat pairs with the right models — the uniform/"
                         "pinned selection loads a NON-first-k subset. Length must == nensemble. "
                         "Default None = first-k (0..M-1), the legacy behaviour. Resolve with "
                         "shared select_member_seeds.py (same rng_seed as the wifi fit).")
parser.add_argument('--out_base', type=str, required=True,
                    help="Base output directory.")
parser.add_argument('--target_data', type=str, default=None,
                    help="Target data: single .npy file. Bootstrapped (replace=True) per "
                         "toy for the observed run — the ONLY option when there is no analytic "
                         "truth (4D: finite data holdout). For the 2D toy prefer --target_truth.")
parser.add_argument('--target_truth', type=str, default=None, choices=['2d_gmm_skew'],
                    help="Observed run (calibration=0): draw Ntest events FRESH from this "
                         "analytic truth every toy instead of bootstrapping --target_data. "
                         "Only for the 2D toy model (known DGP) — gives genuinely independent "
                         "events (no replace=True) so the Ntest/Ntrain oversampling factor is "
                         "real even at factor>1. 4D has no analytic truth -> leave None.")
parser.add_argument('-e', '--nensemble', type=int, required=True,
                    help="Number of ensemble models (incl. norm model).")
parser.add_argument('-n', '--ntest', type=int, required=True,
                    help="Number of test/calibration events.")
parser.add_argument('-c', '--calibration', type=int, required=True,
                    help="1 = calibration (null) toys; 0 = test on target data.")
parser.add_argument('-s', '--seed', type=int, default=None)
parser.add_argument('--toy_id', type=int, default=None)
parser.add_argument('--save_arrays', action='store_true')
parser.add_argument('--fix_wifi_weights', action='store_true',
                    help="Freeze WiFi weights at w_hat (frozen mode). Default (no flag) = "
                         "constrained: profile w under the N(w_hat, Sigma_w) prior.")

# Numerator = NPLM exp-tilt  f_num = f_ens * exp(sum_j b_j G_j) / Z  (log-space, clean chi2).
parser.add_argument('--n_kernels', type=int, default=100, help="M numerator kernels.")
parser.add_argument('--kernel_sigma', type=float, default=0.3, help="Kernel width sigma.")
parser.add_argument('--lam_pert', type=float, default=1.0,
                    help="L2 ridge on tilt coeffs b (= one-model/classifier).")
parser.add_argument('--clip_b', type=float, default=None,
                    help="Symmetric box |b_j| <= clip_b on the tilt coeffs. Bounds the tilt "
                         "runaway on kernels with data support but little reference support (the "
                         "NUM non-convergence). Tune to b's O(1) scale (max|b|~1 here). Default "
                         "None = off (L2 ridge only).")
parser.add_argument('--z_mode', type=str, default='sample', choices=['sample', 'grid'],
                    help="Z normalization. 'sample' (DEFAULT) = importance sampling from q "
                         "(equal-weight member mixture); works in any d, the only option in 4D, "
                         "and matches the one-model sample-Z choice. 'grid' = deterministic "
                         "quadrature (exact one-sample; 2D only) — kept as a 2D cross-check to "
                         "validate the importance-sample Z is unbiased.")
parser.add_argument('--grid_points', type=int, default=300,
                    help="[grid] points per dimension.")
parser.add_argument('--grid_pad', type=float, default=0.2,
                    help="[grid] fractional padding beyond the data range.")
parser.add_argument('--n_ref', type=int, default=None,
                    help="N_ref = # REFERENCE events used to estimate the normalization "
                         "Z = integral f_ens*exp(sum b_j G_j) dx. This makes the test a genuine "
                         "2-SAMPLE test: N_ref is the reference partner of Ntest (the test "
                         "sample). Importance-sampled from q (z_mode=sample). Default None -> "
                         "Ntest (1:1); use >> Ntest in 4D to suppress MC-Z noise.")

args = parser.parse_args()

train_wifi_weights = not args.fix_wifi_weights
use_prior = train_wifi_weights   # constrained (N(w_hat, Sigma) prior) unless frozen (--fix_wifi_weights)

# M=1 (ensemble-of-one) has ZERO free weights: w is pinned to 1.0 by sum-to-one and Sigma_w
# is 0x0, so there is nothing to profile -- constrained is mathematically IDENTICAL to frozen
# (empty weight-prior, null carries no weight uncertainty). Force frozen so the constrained
# path (fit_nplm_tilt_constrained) never hits the degenerate empty-w case. Makes LRT.py robust
# for -e 1 regardless of how it is invoked (no longer relies on the submit-script guardrail).
if args.nensemble == 1 and train_wifi_weights:
    print("[M=1] ensemble-of-one has no free weights to profile; forcing frozen "
          "(constrained == frozen here).", flush=True)
    train_wifi_weights = False
    use_prior          = False

seed  = args.seed
if seed is None:
    seed = (datetime.datetime.now().microsecond
            + datetime.datetime.now().second
            + datetime.datetime.now().minute)
print('Random seed:', seed)
label = args.toy_id if args.toy_id is not None else seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

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

prefix = "SparKer" if args.model_type == "kernels" else "NF"
# Folder name exposes, up front, the five knobs that define a run: ensemble size (Nens),
# test-sample size (Ntest), # numerator kernels (M), kernel width (W); frozen/constrained is
# appended below. Keep these labelled so the run is identifiable from the path alone.
run_tag = "%s_Nens%i_Ntest%i_M%i_W%s" % (
    prefix, args.nensemble, Ntest,
    n_kernels_numerator, str(kernel_width_numerator),
)
# N_ref / N_test scaling of the 2-sample reference, ALWAYS in the path so the ratio is never
# ambiguous (1 = N_ref=N_test, the default; 5 = N_ref=5*N_test, ...). Different ratios land in
# DISTINCT dirs (like the clip label) so a rerun at a new N_ref can't silently mix null/test.
n_ref_eff = args.n_ref if args.n_ref is not None else Ntest
run_tag += "_Nrefx%g" % (n_ref_eff / Ntest)
run_tag += "_Lp%g" % args.lam_pert   # NPLM exp-tilt is the only numerator now (no additive/mult label)
# Clip status is ALWAYS in the path (clipb%g when on, clipoff when off). clip CHANGES the
# estimator, so clipped and unclipped T's must NEVER be mixed into one null/test dist; the
# explicit label makes the two land in DISTINCT dirs so a future rerun can't silently splice
# a clipped toy into an unclipped campaign (or vice versa). The 2D scan on disk predates this
# label and has NO clip token -> it is the (unclipped) legacy set; do not extend it with clip.
if args.clip_b is not None:
    run_tag += "_clipb%g" % args.clip_b
else:
    run_tag += "_clipoff"
if args.z_mode != 'sample':
    run_tag += "_zgrid"

if args.nensemble == 1:
    wifi_tag = "single"          # single model = ensemble-of-one, no wifi fit
else:
    wifi_tag = '_'.join(os.path.basename(os.path.dirname(
        os.path.abspath(args.w_cov_path))).split('_')[-2:])
run_tag += "_wifi_%s" % wifi_tag

# Null TYPE in the path (group terminology, aligned with the paper): frozen w = the plug-in
# POINT-NULL (fitted density treated as exact); constrained w = the COMPOSITE-NULL (weight
# nuisances profiled under N(w_hat, Sigma_w)). --fix_wifi_weights stays mechanism-named; these
# tokens are the statistical labels used in the folders + plots.
if not train_wifi_weights:
    run_tag += "_point_null"         # frozen weights: plug-in at w_hat (point-null)
else:
    run_tag += "_composite_null"     # w profiled under the N(w_hat, Sigma) prior (composite-null)

if args.w_cov_scale != 1.0:          # diagnostic runs land in a SEPARATE dir (no clobber)
    run_tag += "_covscale%g" % args.w_cov_scale

out_dir = os.path.join(args.out_base, run_tag, mode_tag, "seed%i" % label)
os.makedirs(out_dir, exist_ok=True)
print("Writing outputs to:", out_dir, flush=True)

# -------------------------------------------------------------------
# Load WiFi weights (common to both model types)
# -------------------------------------------------------------------
if args.nensemble == 1:
    # Single model = ensemble-of-one: trivial weight 1.0, no weight uncertainty.
    # There is no wifi fit for one model, so --w_path/--w_cov_path are not required;
    # the frozen path then reduces to a plain single-model sample-Z test.
    weights_centralv = np.array([1.0], dtype=np.float64)         # (1,)
    weights_cov_init = np.zeros((0, 0), dtype=np.float64)        # (0, 0)
    if args.w_path is not None or args.w_cov_path is not None:
        print("[M=1] ignoring --w_path/--w_cov_path; single model uses w=[1.0], cov=0x0",
              flush=True)
else:
    if args.w_path is None or args.w_cov_path is None:
        raise ValueError("--w_path and --w_cov_path are required for nensemble > 1.")
    weights_centralv = np.load(args.w_path)       # (M,)
    weights_cov_init = np.load(args.w_cov_path)   # (M-1, M-1)
    assert weights_cov_init.shape == (len(weights_centralv) - 1, len(weights_centralv) - 1), \
        f"cov_w shape {weights_cov_init.shape} != ({len(weights_centralv)-1}, {len(weights_centralv)-1})"
    if args.w_cov_scale != 1.0:
        weights_cov_init = weights_cov_init * args.w_cov_scale
        print(f"[w_cov_scale={args.w_cov_scale:g}] scaled Sigma_w (trace now "
              f"{np.trace(weights_cov_init):.4e}); constrained T should -> frozen T as scale->0",
              flush=True)

n_wifi_components = args.nensemble   # M total models (last is norm model)

# Member indices to load, IN ORDER (last = norm model). The uniform/pinned wifi ensemble is a
# NON-first-k subset, so the LRT MUST load the same seeds in the same order the wifi fit used,
# else w_hat is paired with the wrong models. Resolved by the submit script via
# select_member_seeds.py (same rng_seed as the wifi fit) and passed as --member_seeds.
# Default None -> first-k (0..M-1), the legacy behaviour.
if args.member_seeds is not None:
    member_seeds = [int(t) for t in args.member_seeds.split(",") if t.strip() != ""]
    if len(member_seeds) != n_wifi_components:
        raise ValueError(f"--member_seeds has {len(member_seeds)} seeds but nensemble="
                         f"{n_wifi_components}")
else:
    member_seeds = list(range(n_wifi_components))
print(f"Member seeds (load order; last=norm): {member_seeds}", flush=True)

# -------------------------------------------------------------------
# Load ensemble model (model-type specific)
# -------------------------------------------------------------------
if args.model_type == 'kernels':
    with open(os.path.join(args.ensemble_dir, "config.json"), "r") as f:
        config_json = json.load(f)
    n_kernels_list = config_json["number_centroids"]
    model_type_cfg = config_json["model"]
    seed_fmt = args.seed_format

    centroids_init, coefficients_init, widths_init = [], [], []
    centroids_norm, coefficients_norm, widths_norm = [], [], []

    for i, s in enumerate(member_seeds):
        seed_dir = os.path.join(args.ensemble_dir, seed_fmt % s)
        tmp = np.load(os.path.join(seed_dir, "widths_history.npy"))
        count = -1
        for j in range(tmp.shape[0]):
            if tmp[j][0].sum():
                count += 1
            else:
                print(count)
                break
        centroids_all = np.load(os.path.join(seed_dir, "centroids_history.npy"))[count]
        coeffs_all    = np.load(os.path.join(seed_dir, "coeffs_history.npy"))[count]
        widths_all    = np.load(os.path.join(seed_dir, "widths_history.npy"))[count, :, 0]

        if i < n_wifi_components - 1:
            centroids_init.append(centroids_all)
            coefficients_init.append(coeffs_all)
            widths_init.append(widths_all)
        else:
            centroids_norm.append(centroids_all)
            coefficients_norm.append(coeffs_all)
            widths_norm.append(widths_all)

    # Norm model always exists (the last of the M members); stack it first so we can
    # shape the mixture arrays from it when M==1 (no mixture members -> empty stacks).
    centroids_norm      = np.stack(centroids_norm, axis=0)                       # (1, K, d)
    coefficients_norm   = np.stack(coefficients_norm, axis=0)
    coefficients_norm   = coefficients_norm / np.sum(coefficients_norm, axis=1, keepdims=True)
    widths_norm         = np.stack(widths_norm, axis=0)
    if centroids_init:                                                          # M > 1
        centroids_init    = np.stack(centroids_init, axis=0)
        coefficients_init = np.stack(coefficients_init, axis=0)
        coefficients_init = coefficients_init / np.sum(coefficients_init, axis=1, keepdims=True)
        widths_init       = np.stack(widths_init, axis=0)
    else:                                                                       # M == 1: no mixture members
        centroids_init    = np.empty((0,) + centroids_norm.shape[1:], dtype=centroids_norm.dtype)     # (0, K, d)
        coefficients_init = np.empty((0,) + coefficients_norm.shape[1:], dtype=coefficients_norm.dtype)
        widths_init       = np.empty((0,) + widths_norm.shape[1:], dtype=widths_norm.dtype)
    print(centroids_init.shape, coefficients_init.shape, widths_init.shape)

elif args.model_type == 'nf':
    from utils_flows import make_flow
    # Per-member state dicts. Preferred: iterate <nf_train_dir>/model_{i:03d}/model.pth
    # (the training layout, mirrors the kernels seed%03d loop; the f_i.pth bundle was
    # retired). Legacy fallback: a single --fi_path bundle.
    if args.nf_train_dir is not None:
        arch_path = args.arch_config or os.path.join(args.nf_train_dir,
                                                      "architecture_config.json")
        f_i_statedicts = [
            torch.load(os.path.join(args.nf_train_dir, "model_%03d" % s, "model.pth"),
                       map_location="cpu")
            for s in member_seeds
        ]
    elif args.fi_path is not None:
        arch_path = args.arch_config
        f_i_statedicts = torch.load(args.fi_path, map_location="cpu")
    else:
        raise ValueError("nf model_type needs --nf_train_dir (dir of model_%03d/model.pth) "
                         "or a legacy --fi_path bundle.")
    with open(arch_path) as f:
        arch_config = json.load(f)
    print(f"Loaded {len(f_i_statedicts)} NF state dicts, arch: {arch_config}", flush=True)

# -------------------------------------------------------------------
# Ensemble evaluation helper
# -------------------------------------------------------------------
NF_BATCH = 5000   # events per forward pass for NF evaluation

def _eval_ensemble(data_np):
    """
    Evaluate the M-model ensemble on data_np (N, d) numpy array.
    Returns numpy array (N, M): column i is the density of model i.
    Layout: [:, :-1] = mixture models, [:, -1] = norm model.
    """
    if args.model_type == 'kernels':
        probs = []
        for i in range(n_wifi_components - 1):
            comps = gen.evaluate_gaussian_components(
                data_np, centroids_init[i], widths_init[i])
            probs.append((comps * coefficients_init[i]).sum(axis=1))
        norm_comps = gen.evaluate_gaussian_components(
            data_np, centroids_norm[0], widths_norm[0])
        norm_prob = (norm_comps * coefficients_norm[0]).sum(axis=1)
        return np.column_stack(probs + [norm_prob])   # (N, M)

    elif args.model_type == 'nf':
        per_model = []
        for state_dict in f_i_statedicts:
            flow_kwargs = {k: v for k, v in arch_config.items() if k != 'backend'}
            flow = make_flow(**flow_kwargs)
            flow.load_state_dict(state_dict)
            flow = flow.to(device).float().eval()
            vals = []
            with torch.no_grad():
                for i in range(0, len(data_np), NF_BATCH):
                    xb = _np2t(data_np[i:i + NF_BATCH], dtype=torch.float32).to(device)
                    vals.append(torch.exp(flow.log_prob(xb)).detach().cpu().double())
            per_model.append(torch.cat(vals))
            del flow
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        return _t2np(torch.stack(per_model, dim=1))  # (N, M)


def _sample_ensemble_q(n, rng):
    """Sample n points from q = (1/M) sum_m model_m — the equal-weight member
    mixture. A positive, samplable proposal for importance-sampling / SIR of the
    signed-weight ensemble density f_ens (which is NOT a mixture, so cannot be
    sampled by picking a member ~ w). Wired for BOTH kernels (component ~ coeffs +
    Gaussian draw) and NF (flow.sample) — the universal IS proposal that replaces the
    external hit-or-miss pool (Generate_Ensemble_Samples) for every model type."""
    M = n_wifi_components
    midx = rng.integers(0, M, size=n)                         # member per draw ~ Uniform(M)

    if args.model_type == 'kernels':
        d = centroids_init.shape[2]
        out = np.empty((n, d), dtype=np.float64)
        for m in range(M):
            sel = np.nonzero(midx == m)[0]
            if sel.size == 0:
                continue
            if m < M - 1:
                cen, coef, wid = centroids_init[m], coefficients_init[m], widths_init[m]
            else:
                cen, coef, wid = centroids_norm[0], coefficients_norm[0], widths_norm[0]
            k = rng.choice(len(coef), size=sel.size, p=coef)  # component ~ coeffs
            out[sel] = cen[k] + wid[k, None] * rng.standard_normal((sel.size, d))
        return out

    elif args.model_type == 'nf':
        # q = (1/M) sum_m flow_m: pick member ~ Uniform(M), draw from that flow via
        # flow.sample. Torch's RNG is seeded from the passed numpy rng so the draw is
        # reproducible and tied to the same seed stream as the kernel branch.
        d = int(arch_config["num_features"])
        out = np.empty((n, d), dtype=np.float64)
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        flow_kwargs = {k: v for k, v in arch_config.items() if k != 'backend'}
        for m in range(M):
            sel = np.nonzero(midx == m)[0]
            if sel.size == 0:
                continue
            flow = make_flow(**flow_kwargs)
            flow.load_state_dict(f_i_statedicts[m])
            flow = flow.to(device).float().eval()
            with torch.no_grad():
                sm = flow.sample(int(sel.size))               # (sel, d) on device
            out[sel] = _t2np(sm)
            del flow
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        return out

    raise NotImplementedError(f"q-sampling not wired for model_type={args.model_type}")


# -------------------------------------------------------------------
# Observed sample (calibration=0)
# -------------------------------------------------------------------
def _sample_2d_gmm_skew(N, rng):
    """Fresh i.i.d. draw from the 2D analytic truth behind the target data:
    x0 = bimodal Gaussian mixture (50/50, mu=-0.70/-0.30, sig=0.12),
    x1 = skew-normal (loc=1.0, scale=0.75, alpha=8.0).
    This is the SAME distribution for BOTH the kernels (generate_2d_gmm_skew) and NF
    (generate_2d_gaussian_heavy_tail_target_data) branches — identical params, only the
    old cache seed/dtype differed. Drawing per toy makes the observed sample genuinely
    independent (no replace=True), so the Ntest/Ntrain oversampling factor is real even
    at factor>1. Params are the verbatim DGP; do not change."""
    wG = 0.50
    mu_a, sig_a = -0.70, 0.12
    mu_b, sig_b = -0.30, 0.12
    n_a = int(rng.binomial(N, wG))
    x0 = np.concatenate([rng.normal(mu_a, sig_a, n_a),
                         rng.normal(mu_b, sig_b, N - n_a)])
    loc, scale, alpha = 1.0, 0.75, 8.0
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    z0 = rng.standard_normal(N)
    z1 = rng.standard_normal(N)
    x1 = loc + scale * (delta * np.abs(z0) + np.sqrt(1.0 - delta ** 2) * z1)
    data = np.column_stack([x0, x1]).astype(np.float64)
    rng.shuffle(data)
    return data


_TARGET_TRUTHS = {'2d_gmm_skew': _sample_2d_gmm_skew}


def _draw_observed(Ntest, seed):
    """calibration=0 observed events. --target_truth set (2D toy) -> fresh analytic
    draw per toy (genuine, non-bootstrap; the oversampling factor is real). Else (4D,
    no analytic truth) -> bootstrap the finite --target_data holdout (replace=True)."""
    if args.target_truth is not None:
        rng_obs = np.random.default_rng((seed if seed is not None else 0) + 20240517)
        return _TARGET_TRUTHS[args.target_truth](Ntest, rng_obs)
    if args.target_data is None:
        raise ValueError("calibration=0 needs --target_truth (2D toy) or --target_data (4D).")
    data_all = np.load(args.target_data)
    idx = np.random.choice(len(data_all), Ntest, replace=True)   # finite holdout bootstrap
    return data_all[idx]


# ===================================================================
# NUMERATOR = NPLM exp-tilt:  f_num = f_ens(x; w) * exp(tau) / Z,  tau = sum_j b_j G_j.
#   T = 2[sum_i tau(x_i) - N logZ]  (shared log f_ens cancels in the frozen case).
# Data + null are self-contained (no external hit-or-miss pool):
#   calibration=1 : null via SIR from q (equal-weight mixture).
#   calibration=0 : bootstrap of the target data (replace=True).
# Z (--z_mode): grid = deterministic quadrature (2D); sample = importance from q (any d, 4D).
#   CONSTRAINED (use_prior): jointly profile w under the N(w_hat, Sigma_w) prior AND the tilt b
#     (the "uncertainty propagated" leg); sample-Z only. See fit_nplm_tilt_constrained.
#   FROZEN (--fix_wifi_weights): w fixed at w_hat, only b fit — convex; Z: grid | sample.
# ===================================================================
np.random.seed(seed)
w_hat  = weights_centralv.astype(np.float64)     # (M,) fitted wifi weights
w_free = w_hat[:-1]
def _f_ens(P):
    return P[:, :-1] @ w_free + P[:, -1] * (1.0 - w_free.sum())

if use_prior:
    # =====================================================================
    # CONSTRAINED: profile w under N(w_hat, Sigma_w) prior jointly with the tilt b.
    # T = 2[(ll_num + logprior_num) - (ll_den + logprior_den)]  (b-ridge excluded from the
    # priors; reduces to the frozen T as Sigma_w -> 0). Sample-Z only.
    # =====================================================================
    if args.z_mode == 'grid':
        raise NotImplementedError(
            "constrained multiplicative uses sample-Z (importance from q); "
            "z_mode=grid is a frozen-only 2D cross-check.")
    # ---- data + posterior-predictive null ----
    if args.calibration:
        rng = np.random.default_rng(seed)
        n_pool = max(2 * Ntest, (args.n_ref if args.n_ref is not None else Ntest))
        q_pool = _sample_ensemble_q(n_pool, rng)
        pool_probs = _eval_ensemble(q_pool)
        q_dens = pool_probs.mean(axis=1)
        # w_toy ~ N(w_hat, Sigma_w) then SIR-from-q to f_ens(w_toy): the null carries
        # the weight uncertainty (the whole point of the constrained test).
        cov = weights_cov_init
        eps_chol = 1e-8 * np.trace(cov) / cov.shape[0]
        L_chol = np.linalg.cholesky(cov + eps_chol * np.eye(cov.shape[0]))
        w_toy_free = w_free + L_chol @ rng.standard_normal(len(w_free))
        w_toy = np.append(w_toy_free, 1.0 - w_toy_free.sum())
        f_gen = pool_probs @ w_toy
        sir_w = np.maximum(f_gen, 0.0) / q_dens
        sir_w /= sir_w.sum()
        ess = 1.0 / (sir_w ** 2).sum()
        print(f"[constrained] posterior-predictive null: ||w_toy-w_hat||="
              f"{np.linalg.norm(w_toy - w_hat):.4f}, SIR-from-q ESS={ess:.0f}/{n_pool}",
              flush=True)
        idx = rng.choice(n_pool, size=Ntest, replace=True, p=sir_w)
        bootstrap_sample = q_pool[idx]
        probs_np = pool_probs[idx]
    else:
        # calib=0 observed. 2D toy: fresh i.i.d. draw from the analytic truth
        # (--target_truth) -> genuine oversampling, NO bootstrap. 4D: bootstrap the
        # finite --target_data holdout (replace=True). See _draw_observed.
        bootstrap_sample = _draw_observed(Ntest, seed)
        probs_np = _eval_ensemble(bootstrap_sample)
    N = bootstrap_sample.shape[0]

    sigma_anchor = gen.candidate_sigma(bootstrap_sample)
    print(f"[sigma diag] RAW candidate_sigma(perc90)={sigma_anchor:.3f} — UNRESCALED; do NOT tune "
          f"to this (raw P90 is the WRONG anchor per the M_sigma memory; use the rescaled-quantile "
          f"scan, 2D fixed value=0.4). Running kernel_sigma={kernel_width_numerator}, "
          f"M={n_kernels_numerator} (sqrt(N)={math.sqrt(N):.0f}); keep sigma FIXED + MATCHED.",
          flush=True)

    centers_np = bootstrap_sample[:n_kernels_numerator].astype(np.float64)
    K_data = lrt.gaussian_kernel_matrix(bootstrap_sample.astype(np.float64),
                                        centers_np, kernel_width_numerator)
    # reference for Z: importance-sample from q (Z(w,b) recomputed inside the fitter)
    n_ref = args.n_ref if args.n_ref is not None else Ntest
    rng_ref = np.random.default_rng((seed if seed is not None else 0) + 987654321)
    y = _sample_ensemble_q(n_ref, rng_ref)
    yprobs = _eval_ensemble(y)
    q_y = yprobs.mean(axis=1)
    K_ref = lrt.gaussian_kernel_matrix(y, centers_np, kernel_width_numerator)
    n_neg = int((_f_ens(yprobs) <= 0).sum())
    if n_neg:
        print(f"WARNING: {n_neg}/{n_ref} reference points have f_ens(w_hat)<=0.", flush=True)

    den = lrt.fit_nplm_tilt_constrained(
        probs_np, w_hat, weights_cov_init, fit_w=True, name="DEN")
    # Warm-start the joint NUM at (u_den, b*(u_den)): u already at its b=0 optimum,
    # b at the frozen tilt optimum given u_den — near the joint optimum, so trust-exact
    # polishes in a few iters (from (u_hat, 0) it stalls at production M=500 scale).
    w_den_full = den["w"]                                       # (M,)
    lw_ref_den = np.log(np.maximum(yprobs @ w_den_full, 1e-300)) - np.log(np.maximum(q_y, 1e-300))
    b_init = lrt.fit_nplm_tilt(K_data, K_ref, lam_pert=args.lam_pert, clip=args.clip_b,
                               log_w_ref=lw_ref_den, name="NUM-binit")["b"]
    num = lrt.fit_nplm_tilt_constrained(
        probs_np, w_hat, weights_cov_init, K_data, yprobs, K_ref, q_y,
        fit_w=True, lam_pert=args.lam_pert, clip=args.clip_b,
        u_init=den["u"], b_init=b_init, name="NUM")
    if not (den["converged"] and num["converged"]):
        print(f"WARNING: constrained fit not converged (DEN ||g||={den['grad_norm']:.2e}, "
              f"NUM ||g||={num['grad_norm']:.2e})", flush=True)

    den_log_np = den["log_model_data"]        # (N,) log f_ens(x; w_den)
    num_log_np = num["log_model_data"]        # (N,) log f_num(x; w_num, b)
    lp_den, lp_num = den["logprior"], num["logprior"]
    b_num, logZ = num["b"], num["logZ"]
    # PENALIZED 2*logLR: the b-ridge 1/2 lam||b||^2 is INCLUDED in T (matches
    # classifier_gof.py `t = 2(L_den - L_num)` and the gof2d_gmm notebook — both keep
    # the kernel ridge in t). T = 2[(ll_num + logprior_num - 1/2 lam||b||^2)
    #                              - (ll_den + logprior_den)].
    # The weight-prior AND the b-ridge are spread over events so sum(test_np) == T.
    # b=0 recovers the den fit (ridge=0) => T >= 0 preserved.
    b_ridge = 0.5 * args.lam_pert * float(np.dot(b_num, b_num))
    test_np = 2.0 * ((num_log_np - den_log_np) + (lp_num - lp_den - b_ridge) / N)
    T = 2.0 * ((num["ll"] + lp_num - b_ridge) - (den["ll"] + lp_den))
    assert abs(T - test_np.sum()) < 1e-4 * (1.0 + abs(T)), (T, float(test_np.sum()))
    assert T >= -1e-4, f"T = {T} < 0: nested LRT violated — a fit did not converge."
    print(f"T = {T:.6f}", flush=True)
    print(f"mean per-event log LR = {float(test_np.mean()):.6f}", flush=True)

    with open(os.path.join(out_dir, f"seed{label}_T.txt"), "w") as f:
        f.write(f"{T}\n")
    np.save(os.path.join(out_dir, f"seed{label}_T.npy"), np.array(T, dtype=np.float64))
    if args.save_arrays:
        np.save(os.path.join(out_dir, f"seed{label}_test.npy"),        test_np)
        np.save(os.path.join(out_dir, f"seed{label}_numerator.npy"),   num_log_np)
        np.save(os.path.join(out_dir, f"seed{label}_denominator.npy"), den_log_np)
    np.save(os.path.join(out_dir, f"seed{label}_coeffs.npy"),         b_num)
    np.save(os.path.join(out_dir, f"seed{label}_kernel_centers.npy"), centers_np)
    np.save(os.path.join(out_dir, f"seed{label}_den_weights.npy"),    den["w"][:-1])
    np.save(os.path.join(out_dir, f"seed{label}_num_weights.npy"),    num["w"][:-1])
    np.save(os.path.join(out_dir, f"seed{label}_init_weights.npy"),   w_free)
    with open(os.path.join(out_dir, f"seed{label}_fit_report.json"), "w") as f:
        json.dump({"den": {**{k: den[k] for k in ("ll", "logprior", "grad_norm",
                                               "n_iter", "converged", "hit_max_iter")},
                           "veto": den["veto"]},
                   "num": {**{k: num[k] for k in ("ll", "logprior", "logZ", "grad_norm",
                                               "max_b", "n_iter", "converged", "hit_max_iter")},
                           "veto": num["veto"]}},
                  f, indent=2)
    # Feasibility-constraint diagnostic (Sean Opt 1): how hard the f_ens>0 constraint
    # bound on this toy. Aggregate `start_infeasible` / `n_feas_capped_steps` across the
    # 100 null toys -> if the constraint rarely binds (~1/100) the toy distortion is small.
    print(f"  [veto] DEN start_infeasible={den['veto']['start_infeasible']} "
          f"nneg_data@start={den['veto']['n_neg_data_at_start']} "
          f"capped_steps={den['veto']['n_feas_capped_steps']} "
          f"min_dens={den['veto']['min_data_dens_soln']:.2e} | "
          f"NUM start_infeasible={num['veto']['start_infeasible']} "
          f"capped_steps={num['veto']['n_feas_capped_steps']}", flush=True)
    print("--- Numerator (multiplicative NPLM exp-tilt, CONSTRAINED w) ---")
    print(f"  {len(b_num)} tilt coeffs b, max|b|={np.abs(b_num).max():.3e}, logZ={logZ:.4f}; "
          f"||w_num-w_hat||={np.linalg.norm(num['w']-w_hat):.4f}", flush=True)

else:
    # =====================================================================
    # FROZEN (--fix_wifi_weights): w fixed at w_hat, only b fit — convex. Z: grid | sample.
    # =====================================================================
    # ---- data + null (self-contained) ----
    if args.calibration:
        rng = np.random.default_rng(seed)
        n_pool = max(2 * Ntest, (args.n_ref if args.n_ref is not None else Ntest))
        q_pool = _sample_ensemble_q(n_pool, rng)                  # (n_pool, d) ~ q
        pool_probs = _eval_ensemble(q_pool)                       # (n_pool, M)
        f_ens_pool = _f_ens(pool_probs)
        q_dens     = pool_probs.mean(axis=1)
        sir_w = np.maximum(f_ens_pool, 0.0) / q_dens
        sir_w /= sir_w.sum()
        ess = 1.0 / (sir_w ** 2).sum()
        print(f"[frozen] null via SIR-from-q: ESS = {ess:.0f}/{n_pool}", flush=True)
        idx = rng.choice(n_pool, size=Ntest, replace=True, p=sir_w)
        bootstrap_sample = q_pool[idx]
        probs_np = pool_probs[idx]
    else:
        # calib=0 observed. 2D toy: fresh i.i.d. draw from the analytic truth
        # (--target_truth) -> genuine oversampling, NO bootstrap. 4D: bootstrap the
        # finite --target_data holdout (replace=True). See _draw_observed.
        bootstrap_sample = _draw_observed(Ntest, seed)
        probs_np = _eval_ensemble(bootstrap_sample)
    N = bootstrap_sample.shape[0]

    sigma_anchor = gen.candidate_sigma(bootstrap_sample)
    print(f"[sigma diag] RAW candidate_sigma(perc90)={sigma_anchor:.3f} — UNRESCALED; do NOT tune "
          f"to this (raw P90 is the WRONG anchor per the M_sigma memory; use the rescaled-quantile "
          f"scan, 2D fixed value=0.4). Running kernel_sigma={kernel_width_numerator}, "
          f"M={n_kernels_numerator} (sqrt(N)={math.sqrt(N):.0f}); keep sigma FIXED + MATCHED.",
          flush=True)

    f_ens_data = np.maximum(_f_ens(probs_np), 1e-300)             # (N,)
    den_log_np = np.log(f_ens_data)
    if not np.isfinite(den_log_np).all():
        raise RuntimeError("DEN loglik not finite (frozen).")

    centers_np = bootstrap_sample[:n_kernels_numerator].astype(np.float64)
    K_data = lrt.gaussian_kernel_matrix(bootstrap_sample.astype(np.float64),
                                        centers_np, kernel_width_numerator)   # (N, M_ker)

    # ---- normalization Z: grid (2D) or importance sample from q (any d) ----
    if args.z_mode == 'grid':
        grid, step = lrt.build_eval_grid(bootstrap_sample, args.grid_points, args.grid_pad)
        f_ref = np.maximum(_f_ens(_eval_ensemble(grid)), 1e-300)   # (G,)
        K_ref = lrt.gaussian_kernel_matrix(grid, centers_np, kernel_width_numerator)
        log_w_ref = np.log(f_ref)                                  # self-normalized in fit
        grid_mass = float((f_ref * np.prod(step)).sum())
        print(f"Z via GRID: {args.grid_points}/dim, {grid.shape[0]} pts, "
              f"step={np.array2string(step, precision=3)}; grid f_ens mass = {grid_mass:.4f} "
              f"(should be ~1)", flush=True)
        if not (0.95 <= grid_mass <= 1.05):
            print(f"WARNING: grid f_ens mass = {grid_mass:.4f} far from 1 -> widen extent "
                  f"or raise grid_points.", flush=True)
    else:  # sample: importance sampling from q = equal-weight member mixture (4D)
        n_ref = args.n_ref if args.n_ref is not None else Ntest
        rng_ref = np.random.default_rng((seed if seed is not None else 0) + 987654321)
        y = _sample_ensemble_q(n_ref, rng_ref)                    # (R, d) ~ q
        yprobs = _eval_ensemble(y)                                # (R, M)
        f_ens_y = _f_ens(yprobs)
        q_y     = yprobs.mean(axis=1)
        n_neg = int((f_ens_y <= 0).sum())
        if n_neg:
            print(f"WARNING: {n_neg}/{n_ref} reference points have f_ens<=0 (floored).",
                  flush=True)
        f_ens_y = np.maximum(f_ens_y, 1e-300)
        K_ref = lrt.gaussian_kernel_matrix(y, centers_np, kernel_width_numerator)
        log_w_ref = np.log(f_ens_y) - np.log(q_y)                 # importance log-weights
        iw = f_ens_y / q_y
        ess = float(iw.sum() ** 2 / (iw ** 2).sum())
        print(f"Z via SAMPLE (importance from q): {n_ref} refs; ESS = {ess:.0f}/{n_ref}",
              flush=True)

    res_num = lrt.fit_nplm_tilt(K_data, K_ref, lam_pert=args.lam_pert, clip=args.clip_b,
                                log_w_ref=log_w_ref, name="NUM", verbose=True)
    if not res_num["converged"]:
        print(f"WARNING: NUM tilt fit not converged (||g||={res_num['grad_norm']:.2e})",
              flush=True)
    b_num    = res_num["b"]
    logZ     = res_num["logZ"]
    tau_data = res_num["tau_data"]                                # (N,)
    num_log_np = den_log_np + tau_data - logZ
    # PENALIZED 2*logLR: subtract the b-ridge 1/2 lam||b||^2 so T matches classifier_gof.py
    # + the gof2d_gmm notebook (both keep the kernel ridge in t). Spread over events so
    # sum(test_np) == T; b=0 recovers den (ridge=0) => T >= 0 preserved.
    b_ridge  = 0.5 * args.lam_pert * float(np.dot(b_num, b_num))
    test_np  = 2.0 * (tau_data - logZ - b_ridge / N)
    T = 2.0 * (num_log_np.sum() - den_log_np.sum()) - 2.0 * b_ridge
    assert abs(T - test_np.sum()) < 1e-4 * (1.0 + abs(T)), (T, float(test_np.sum()))
    assert T >= -1e-4, f"T = {T} < 0: nested LRT violated — a fit did not converge."
    print(f"T = {T:.6f}", flush=True)
    print(f"mean per-event log LR = {float(test_np.mean()):.6f}", flush=True)

    with open(os.path.join(out_dir, f"seed{label}_T.txt"), "w") as f:
        f.write(f"{T}\n")
    np.save(os.path.join(out_dir, f"seed{label}_T.npy"), np.array(T, dtype=np.float64))
    if args.save_arrays:
        np.save(os.path.join(out_dir, f"seed{label}_test.npy"),        test_np)
        np.save(os.path.join(out_dir, f"seed{label}_numerator.npy"),   num_log_np)
        np.save(os.path.join(out_dir, f"seed{label}_denominator.npy"), den_log_np)
    np.save(os.path.join(out_dir, f"seed{label}_coeffs.npy"),         b_num)
    np.save(os.path.join(out_dir, f"seed{label}_kernel_centers.npy"), centers_np)
    np.save(os.path.join(out_dir, f"seed{label}_den_weights.npy"),    w_free)
    np.save(os.path.join(out_dir, f"seed{label}_num_weights.npy"),    w_free)   # frozen: num==den
    np.save(os.path.join(out_dir, f"seed{label}_init_weights.npy"),   w_free)
    with open(os.path.join(out_dir, f"seed{label}_fit_report.json"), "w") as f:
        json.dump({"num": {k: res_num[k] for k in ("loglik", "n_iter", "converged",
                                                   "hit_max_iter", "grad_norm", "max_b")}},
                  f, indent=2)
    print("--- Numerator (multiplicative NPLM exp-tilt, frozen w) ---")
    print(f"  {len(b_num)} tilt coeffs b, max|b|={np.abs(b_num).max():.3e}, logZ={logZ:.4f}",
          flush=True)
