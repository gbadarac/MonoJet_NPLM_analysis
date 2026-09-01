"""Parameter loader. Do NOT edit parameters here - edit PARAMS in launch.py."""
import json, os

_pfile = os.environ.get("GOF4D_PARAMS", "")
if _pfile:
    P = json.load(open(_pfile))
    RUN_DIR = os.path.dirname(os.path.abspath(_pfile))
else:
    from launch import PARAMS as P, run_dir as _rd
    RUN_DIR = _rd(P)

FAST = os.environ.get("NB_FAST", "0") == "1"

# ---- data ----
DATA_GLOB   = P["data_glob"]
DATA_KEY    = P["data_key"]
STANDARDIZE = P["standardize"]
BASE_SEED   = P["base_seed"]

# ---- grids ----
N_FIT_LIST = list(P["n_fit_list"])
K_LIST     = list(P["k_list"])
COMP_K_LIST = list(P.get("comp_k_list") or P["k_list"])
NTEST_GRID = list(P["ntest_grid"])
N_WFIT     = P["n_wfit"]

# ---- statistics budgets ----
POINT_B_CALIB = P["point_b_calib"]
POINT_REPEATS = P["point_repeats"]
COMP_B_CALIB  = P["comp_b_calib"]
COMP_REPEATS  = P["comp_repeats"]

# ---- tangent composite ----
M_EIG    = P["m_eig"] if P["m_eig"] != "all" else "all"
LAM_MAX  = P["lam_max"]
COV_MODE = P["cov_mode"]
THETA_SAMPLED_CALIB = P["theta_sampled_calib"]

# ---- alternative dictionary ----
J_CENTERS   = P["j_centers"]
SCALE_FRACS = tuple(P["scale_fracs"])
RIDGE_A     = P["ridge_a"]
ALPHA_CLIP  = P["alpha_clip"]

S_REF       = P["s_ref"]
OPT_OPTIONS = dict(maxiter=P.get("opt_maxiter", 300),   # L-BFGS-B, all 3 solves
                   maxcor=P.get("opt_maxcor", 10),
                   ftol=P.get("opt_ftol", 2.22e-9),
                   gtol=P.get("opt_gtol", 1e-5))
EM_TOL      = P.get("em_tol", 1e-3)       # legacy sklearn default
EM_MAX_ITER = P.get("em_max_iter", 500)
EM_N_INIT   = P.get("em_n_init", 3)
S_EVAL_FACTOR = P.get("s_eval_factor", 10)
S_EVAL_MAX    = P.get("s_eval_max", 2_000_000)
ALPHA_LEVEL = P["alpha_level"]

def chunk_size(test_type, n_test):
    if test_type == "point":
        n = P["chunk_budget_point"] // max(n_test, 1)
        return int(min(max(n, P["chunk_min"]), P["chunk_max"]))
    n = P["chunk_budget_comp"] // max(n_test, 1)
    return int(min(max(n, P["chunk_min"]), P["chunk_max"]))

# ---- paths ----
ART_DIR   = os.path.join(RUN_DIR, "artifacts")
RAW_DIR   = os.path.join(RUN_DIR, "results", "raw")
OUT_DIR   = os.path.join(RUN_DIR, "results")
FIG_DIR   = os.path.join(RUN_DIR, "figs")
ARTIFACTS = os.path.join(ART_DIR, "artifacts.npz")
MANIFEST  = os.path.join(RUN_DIR, "manifest.csv")
