#!/usr/bin/env python3
"""SINGLE ENTRY POINT for the 2D toy GOF experiment (tangent composite test).

1) All parameters live in PARAMS below (documented inline).
2) Each launch creates a run directory whose name encodes the key parameters.
3) The full parameter set is saved to <run_dir>/params.json; every job reads it via
   env GOF2D_PARAMS.
4) Steps:
     python launch.py prep      [--local]                 # data+fits+tangents (1 job)
     python launch.py workers   [--local] [--coverage] [--power 0.05,0.1] [--limit N]
     python launch.py analyze                             # merge + plots + diagnostics

NB_FAST=1 python launch.py ...  runs a minutes-scale smoke version in its own run dir.
"""
import argparse, json, os, subprocess, sys

FAST = os.environ.get("NB_FAST", "0") == "1"

# ============================ PARAMETERS ====================================
PARAMS = dict(
    base_seed=20260826,        # every RNG stream (fits, toys, observed draws)

    # ---- experiment grids -------------------------------------------------------
    n_fit_list=[1_000, 10_000, 100_000] if not FAST else [1_000],
    k_list=[2, 4, 8, 16] if not FAST else [4],  # GMM components (P = 6K-1 at d=2)
    comp_k_list=None,          # K values for the COMPOSITE sweep (None = all of k_list)
    ntest_grid=[100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
               if not FAST else [200, 2_000],
    n_wfit=20_000 if not FAST else 4_000,     # validation sample (K selection only)

    # ---- statistics budget per (N_fit, K, N_test) working point -----------------
    point_b_calib=250 if not FAST else 12,    # null toys (from the model: no data cost)
    point_repeats=100 if not FAST else 4,     # observed samples (drawn from test pool)
    comp_b_calib=200 if not FAST else 12,
    comp_repeats=50 if not FAST else 4,

    # ---- tangent composite -------------------------------------------------------
    m_eig="all",               # profile all identifiable directions ("all" or int)
    lam_max=0.8,               # identifiability threshold: directions with eigenvalue
                               # above it are DROPPED (non-identifiable), never capped
    cov_mode="sandwich",       # "fisher" | "sandwich" (robust to misspecification)
    theta_sampled_calib=True,  # calib toys from exact model at theta_b ~ N(theta, Cov)

    # ---- alternative dictionary ---------------------------------------------------
    j_centers=20,              # kernels (fixed k-means centers on a model sample)
    scale_fracs=[0.25, 0.6],   # widths x median pairwise distance (two scales)
    ridge_a=1.0,               # L2 on alpha (standardized features)
    alpha_clip=1.0,            # sup-norm bound on each kernel's log-distortion

    s_ref=500_000 if not FAST else 3_000,  # Z-hat bank; keep >> largest N_test
    alpha_level=0.05,

    # ---- chunking (target ~15-40 min per single-core task) ------------------------
    chunk_budget_point=1_000_000, # ~ N_test * items per point chunk (analytic scores)
    chunk_budget_comp=400_000,    # composite is ~3x slower per t
    chunk_min=2, chunk_max=50,
)
# ============================================================================

def run_dir(P):
    # The tag encodes only what DEFINES the statistic + data seed. Grids and budgets
    # are growable (see check_compatible): extending them reuses the same run dir and
    # only the new working points / chunks are computed.
    def srt(x): return "-".join(str(v) for v in x)
    tag = (f"s{P['base_seed']}"
           f"_cm-{P['cov_mode']}_M-{P['m_eig']}_lmax{P['lam_max']:g}"
           f"_ts{int(P['theta_sampled_calib'])}"
           f"_J{P['j_centers']}_sc{srt(P['scale_fracs'])}"
           f"_rA{P['ridge_a']:g}_ac{P['alpha_clip']:g}_S{P['s_ref']}")
    return os.path.join("runs", tag + ("_FAST" if FAST else ""))

# keys that may GROW between launches without invalidating existing results
def check_compatible(old, new):
    problems = []
    for key in ["n_fit_list", "k_list"]:
        if not set(old[key]) <= set(new[key]):
            problems.append(f"{key} may only gain values (analytic truth: no "
                            "partition constraints)")
    cko = old.get("comp_k_list") or old["k_list"]
    ckn = new.get("comp_k_list") or new["k_list"]
    if not set(cko) <= set(ckn):
        problems.append("comp_k_list may only gain values")
    if not set(old["ntest_grid"]) <= set(new["ntest_grid"]):
        problems.append("ntest_grid may only gain values")
    for b in ["point_b_calib", "point_repeats", "comp_b_calib", "comp_repeats"]:
        if new[b] < old[b]:
            problems.append(f"{b} may only increase (banks grow by appending chunks)")
    growable = {"n_fit_list", "k_list", "ntest_grid",
                "point_b_calib", "point_repeats", "comp_b_calib", "comp_repeats"}
    for k in old:
        if k not in growable and old[k] != new.get(k):
            problems.append(f"'{k}' changed ({old[k]!r} -> {new.get(k)!r}) - this "
                            "defines the statistic/data and cannot change within a run")
    return problems

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "workers", "analyze"])
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--power", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-concurrent", type=int, default=500)
    ap.add_argument("--compact", action="store_true",
                    help="(analyze) fold raw chunk files into tstats + ledger, delete them")
    ap.add_argument("--seed", type=int, default=0,
                    help="override PARAMS['base_seed'] (multi-seed campaigns: each "
                         "seed gets its own run dir; all other params unchanged)")
    ap.add_argument("--run", type=str, default="",
                    help="operate on an EXISTING run dir using its params.json as-is "
                         "(PARAMS above is ignored; works for every step)")
    a = ap.parse_args()

    if a.seed:
        PARAMS["base_seed"] = a.seed
    if a.run:
        rd = a.run.rstrip("/")
        if not os.path.isdir(rd) and os.path.isdir(os.path.join("runs", rd)):
            rd = os.path.join("runs", rd)          # bare tag name -> runs/<tag>
        pfile = os.path.abspath(os.path.join(rd, "params.json"))
        if not os.path.exists(pfile):
            sys.exit(f"--run: no params.json in {rd}")
        env = os.environ.copy(); env["GOF2D_PARAMS"] = pfile
        print("run dir (from --run):", rd)
        return run_step(a, rd, pfile, env)
    rd = run_dir(PARAMS)
    os.makedirs(rd, exist_ok=True)
    pfile = os.path.abspath(os.path.join(rd, "params.json"))
    if os.path.exists(pfile):
        old_p = json.load(open(pfile))
        if old_p != PARAMS:
            probs = check_compatible(old_p, PARAMS)
            if probs:
                sys.exit(f"{pfile} exists and PARAMS are INCOMPATIBLE:\n  - "
                         + "\n  - ".join(probs))
            json.dump(PARAMS, open(pfile, "w"), indent=2)
            print("compatible growth of the grids/budgets - updated", pfile)
            print("NOTE: rerun `prep` (refits are deterministic; adds the new models), "
                  "then `workers` (only new chunks compute). Do not extend while an "
                  "array is still running against the old params.json.")
    else:
        json.dump(PARAMS, open(pfile, "w"), indent=2)
        print("wrote", pfile)
    env = os.environ.copy(); env["GOF2D_PARAMS"] = pfile
    print("run dir:", rd)
    return run_step(a, rd, pfile, env)

def run_step(a, rd, pfile, env):

    if a.step == "prep":
        if a.local:
            subprocess.run([sys.executable, "prep.py"], env=env, check=True)
        else:
            out = subprocess.run(["sbatch", "--parsable",
                                  f"--export=ALL,GOF2D_PARAMS={pfile}", "prep.sbatch"],
                                 capture_output=True, text=True)
            if out.returncode != 0: sys.exit(f"sbatch failed: {out.stderr}")
            print("submitted prep: job", out.stdout.strip())

    elif a.step == "workers":
        if not os.path.exists(os.path.join(rd, "artifacts", "artifacts.npz")):
            sys.exit("artifacts not found - run `python launch.py prep` first.")
        mm = [sys.executable, "make_manifest.py"]
        if a.coverage: mm.append("--coverage")
        if a.power: mm += ["--power", a.power]
        subprocess.run(mm, env=env, check=True)
        manifest = os.path.join(rd, "manifest.csv")
        ntask = sum(1 for _ in open(manifest)) - 1
        if a.local:
            n = min(ntask, a.limit) if a.limit else ntask
            for i in range(n):
                subprocess.run([sys.executable, "worker.py", "--task-id", str(i)],
                               env=env, check=True)
        else:
            out = subprocess.run(
                ["sbatch", "--parsable", f"--export=ALL,GOF2D_PARAMS={pfile}",
                 f"--array=0-{ntask-1}%{a.max_concurrent}", "worker.sbatch"],
                capture_output=True, text=True)
            if out.returncode != 0: sys.exit(f"sbatch failed: {out.stderr}")
            print(f"submitted worker array (0-{ntask-1}): job", out.stdout.strip())

    elif a.step == "analyze":
        failed = []
        for s in ["merge.py", "plots.py", "diagnostics.py", "fitcheck.py"]:
            cmd = [sys.executable, s]
            if s == "merge.py" and a.compact:
                cmd.append("--compact")
            r = subprocess.run(cmd, env=env)
            if r.returncode != 0:
                failed.append(s)
                print(f"[analyze] WARNING: {s} failed (exit {r.returncode}); "
                      "continuing with the remaining steps.")
        if failed:
            print("[analyze] completed with failures:", ", ".join(failed))

if __name__ == "__main__":
    main()
