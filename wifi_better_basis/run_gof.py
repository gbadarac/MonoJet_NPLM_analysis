"""
wifi_better_basis classifier-GoF orchestrator.

Two variants with DIFFERENT semantics and DIFFERENT toy ensembles:

  - constrained: posterior-predictive null. Toy DGPs draw a fresh
        w_k ~ N(ŵ, Σ_w_sandwich) per toy and SIR-resample N_data points
        from r(.; w_k) · q. Tests "is the data consistent with the model
        within its weight uncertainty?"
  - frozen: plug-in (point) null. Every toy DGP is r(.; ŵ) · q (no w
        sampling). Tests "is the data exactly r(.; ŵ) · q (up to the
        finite-N power of the test set)?"

Both variants share the same q-pool per toy and the same fresh reference
draw — only the SIR-importance weights differ. The variants' fits also
differ: constrained fits w under the Σ_w prior, frozen holds w ≡ ŵ.

There is no longer a cross-variant ordering check (e.g. t_constrained ≤
t_frozen): with different toy DGPs and different fits, no clean
inequality survives.

For the observed test we use the cached `data_test.npy` shard so the GoF
doesn't re-use the linear-head fitting half (half_B). Reference samples
are fresh draws from the fitted Gaussian q.

Reads:
    runs/<name>/wifi_config.json, w_hat.npy, Sigma_w_sandwich.npy,
                 mu_q.npy, Sigma_q.npy, basis/*.pt
    data/<benchmark>_*/data_test.npy

Saves (per variant):
    runs/<name>/gof_<variant>_results.json
    runs/<name>/gof_<variant>_toys_t.npy
    runs/<name>/gof_<variant>_toy_diag.npz
    runs/<name>/gof_<variant>_obs_diag.npz
Saves (constrained-only):
    runs/<name>/gof_toys_w_k.npy            # (N_toys, K+1) posterior draws

Usage:
    python run_gof.py --name <run_name>
    python run_gof.py --name <run_name> --force         # ignore cache
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = PACKAGE_DIR
sys.path.insert(0, PACKAGE_DIR)

from reference import sample_reference
from basis import MLPLogit, evaluate_features
from classifier_gof import (
    PerturbationRBF, test_stat_all_variants, test_stat_one_variant,
    sir_toy, VARIANTS, GRAD_NORM_OK,
)


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def _load_basis(basis_dir, d_in, hidden):
    fnames = sorted(f for f in os.listdir(basis_dir) if f.endswith(".pt"))
    models = []
    for f in fnames:
        m = MLPLogit(d_in, hidden=tuple(hidden))
        m.load_state_dict(torch.load(os.path.join(basis_dir, f), weights_only=True))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models.append(m)
    return models


def _load_run(out_dir):
    with open(os.path.join(out_dir, "wifi_config.json")) as f:
        cfg = json.load(f)
    mu_q = torch.from_numpy(np.load(os.path.join(out_dir, "mu_q.npy"))).double()
    Sigma_q = torch.from_numpy(np.load(os.path.join(out_dir, "Sigma_q.npy"))).double()
    w_hat = torch.from_numpy(np.load(os.path.join(out_dir, "w_hat.npy"))).double()
    Sigma_w = torch.from_numpy(np.load(os.path.join(out_dir, "Sigma_w_sandwich.npy"))).double()
    basis_dir = os.path.join(out_dir, "basis")
    d_in = mu_q.numel()
    models = _load_basis(basis_dir, d_in, cfg["MLP_HIDDEN"])
    return cfg, mu_q, Sigma_q, w_hat, Sigma_w, models


def _load_data_test(cfg):
    # Decoupled N_train / N_test: cache directory encodes both. The original
    # wifi code had N_train in the Ntest slot, which was a bug — fixed here.
    name = (
        f"{cfg['benchmark']}_Ntrain{cfg['N_train']}_Ntest{cfg['N_test']}"
        f"_seed{cfg['seed']}"
    )
    return np.load(os.path.join(PROJECT_ROOT, "data", name, "data_test.npy"))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run folder under runs/")
    p.add_argument("--force", action="store_true",
                   help="re-run even if results JSONs already exist")
    args = p.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "runs", args.name)
    cfg, mu_q, Sigma_q, w_hat, Sigma_w, models = _load_run(out_dir)
    seed = int(cfg.get("seed", 0))

    # Hyperparameters
    M_pert     = int(cfg.get("GOF_M_PERT", 80))
    pert_sigma = float(cfg.get("GOF_PERT_SIGMA", 0.10))
    lam_pert   = float(cfg.get("GOF_LAM_RIDGE_PERT", 1e-3))
    max_iter   = int(cfg.get("GOF_MAX_ITER", 500))
    tol        = float(cfg.get("GOF_TOL", 1e-9))
    N_toys     = int(cfg.get("GOF_N_TOYS", 100))
    oversample = int(cfg.get("GOF_TOY_OVERSAMPLE", 10))
    ridge_rel  = float(cfg.get("SANDWICH_RIDGE_REL", 1e-8))
    cfg_n_ref  = cfg.get("GOF_N_REF", None)

    # Skip-if-already-done check (any-variant)
    result_paths = {
        v: os.path.join(out_dir, f"gof_{v}_results.json") for v in VARIANTS
    }
    if (not args.force) and all(os.path.exists(p) for p in result_paths.values()):
        print("[run_gof] all variant results already exist; pass --force to re-run")
        return

    # ── Load data and build feature matrices ──────────────────────
    X_test = _load_data_test(cfg)
    N_data = X_test.shape[0]
    N_ref  = int(cfg_n_ref) if cfg_n_ref else N_data
    if cfg_n_ref is None:
        # also allow the wifi training's choice if the user wants parity
        n_lh = cfg.get("N_REF_LINHEAD")
        if n_lh:
            N_ref = int(n_lh)
    print("=" * 62)
    print(f"wifi_better_basis classifier GoF — run {args.name}")
    print("=" * 62)
    print(f"  N_data (test) = {N_data}, N_ref = {N_ref}")
    print(f"  M_pert = {M_pert}, pert_sigma = {pert_sigma}, lam_pert = {lam_pert}")
    print(f"  optimiser: scipy trust-exact (Newton trust-region, exact Hessian), "
          f"max_iter = {max_iter}, gtol = {tol}")
    print(f"  N_toys = {N_toys}, oversample = {oversample}")
    print(f"  variants = {VARIANTS}  "
          f"(constrained: posterior-predictive null;  frozen: plug-in null at ŵ)")

    Xt_data = torch.from_numpy(X_test).double()
    F_data  = evaluate_features(models, Xt_data)

    X_ref = sample_reference(mu_q, Sigma_q, N_ref, seed=seed + 314159)
    F_ref = evaluate_features(models, X_ref)

    # ── Perturbation kernels at random test-data points ───────────
    rng = np.random.RandomState(seed + 271828)
    cent_idx = rng.choice(N_data, size=M_pert, replace=False)
    centroids = Xt_data[cent_idx]
    pert = PerturbationRBF(centroids, pert_sigma)
    G_data = pert.matrix(Xt_data)
    G_ref  = pert.matrix(X_ref)
    print(f"  built {M_pert} RBF perturbation kernels (sigma = {pert_sigma})")
    print(f"  G_data range = [{G_data.min().item():.3e}, {G_data.max().item():.3e}]")
    print(f"  G_ref  range = [{G_ref.min().item():.3e}, {G_ref.max().item():.3e}]")

    # ── Observed test stat for each variant ───────────────────────
    # Both variants see the same (F_data, F_ref) for the OBSERVED stat;
    # only their fits differ. The toy DGPs differ further (handled below).
    print("\n[obs] running denominator + numerator for each variant")
    t0 = time.time()
    obs = test_stat_all_variants(
        F_data, F_ref, G_data, G_ref,
        w_hat, Sigma_w,
        lam_pert=lam_pert, max_iter=max_iter, tol=tol,
        ridge_rel=ridge_rel, verbose=True,
    )
    print(f"  obs done in {time.time()-t0:.1f}s")
    print()
    print(f"  {'variant':<13s}  {'t_obs':>9s}  {'L_den':>10s}  {'L_num':>10s}  "
          f"{'den ||∇||':>10s}  {'num ||∇||':>10s}  {'den n_eval':>10s}  {'num n_eval':>10s}")
    for v in VARIANTS:
        d = obs[v]["den"]; n = obs[v]["num"]
        print(f"  {v:<13s}  {obs[v]['t']:+9.3f}  {d['loss']:10.4f}  {n['loss']:10.4f}  "
              f"{d['grad_norm']:10.2e}  {n['grad_norm']:10.2e}  "
              f"{d['n_eval']:10d}  {n['n_eval']:10d}")
        for side, fit in (("den", d), ("num", n)):
            if not fit["converged"]:
                print(f"    WARN: [{v}/{side}] grad_norm = {fit['grad_norm']:.2e} "
                      f"> {GRAD_NORM_OK:.0e}; may not be at the optimum.")
            if fit["hit_max_iter"]:
                print(f"    WARN: [{v}/{side}] trust-exact hit maxiter "
                      f"({max_iter}); raise GOF_MAX_ITER.")
        if obs[v]["t"] < 0:
            print(f"    WARN: [{v}] t_obs < 0 — numerator should always fit at "
                  f"least as well as denominator. Likely an optimisation issue.")

    # ── Calibration toys ─────────────────────────────────────────
    # Per toy:
    #   1. Draw a fresh q-pool of size M_per_toy and a fresh reference draw
    #      (shared across both variants).
    #   2. constrained variant — posterior-predictive null:
    #        - draw w_k ~ N(ŵ, Σ_w_sandwich) via Cholesky
    #        - SIR N_data points from the q-pool weighted by exp(F · w_k)
    #        - fit num+den with the constrained Sigma_w prior
    #   3. frozen variant — plug-in null at w = ŵ:
    #        - SIR N_data points from the SAME q-pool weighted by exp(F · ŵ)
    #          (no w sampling)
    #        - fit num+den with w fixed at ŵ (only b is fit)
    M_per_toy = oversample * N_data
    print(f"\n[toys] {N_toys} pseudo-experiments per variant")
    print(f"  shared per toy: q-pool size {M_per_toy}, fresh reference draw "
          f"size {N_ref}")
    print(f"  constrained DGP: SIR weighted by exp(F·w_k), w_k ~ N(ŵ, Σ_w)")
    print(f"  frozen      DGP: SIR weighted by exp(F·ŵ)  (plug-in null)")

    # Cholesky of Σ_w with a small ridge for numerical stability
    K1 = w_hat.numel()
    sigma_w_trace = float(torch.trace(Sigma_w))
    chol_eps = ridge_rel * (sigma_w_trace / K1)
    Sigma_w_pd = Sigma_w + chol_eps * torch.eye(K1, dtype=Sigma_w.dtype)
    L_w = torch.linalg.cholesky(Sigma_w_pd)

    sw_diag = torch.diag(Sigma_w).cpu().numpy()
    print(f"  Σ_w_sandwich diag: min={sw_diag.min():.3e}, "
          f"max={sw_diag.max():.3e}, "
          f"trace={sigma_w_trace:.3e}, eff_dim≈{K1}")

    # Per-toy diagnostics
    toys_w_k = np.zeros((N_toys, K1), dtype=np.float64)   # constrained-only
    t_toys = {v: np.zeros(N_toys) for v in VARIANTS}
    toy_diag = {
        v: {side: {
            "n_eval":       np.zeros(N_toys, dtype=np.int32),
            "grad_norm":    np.zeros(N_toys, dtype=np.float64),
            "converged":    np.zeros(N_toys, dtype=bool),
            "hit_max_iter": np.zeros(N_toys, dtype=bool),
        } for side in ("den", "num")} for v in VARIANTS
    }
    t1 = time.time()
    for ti in range(N_toys):
        # ── Shared resources for this toy ─────────────────────────
        Xq_pool_t = sample_reference(mu_q, Sigma_q, M_per_toy,
                                     seed=seed + 90909 + 23 * ti)
        F_pool_t = evaluate_features(models, Xq_pool_t)
        X_ref_t  = sample_reference(mu_q, Sigma_q, N_ref,
                                    seed=seed + 11 * ti + 2)
        F_ref_t  = evaluate_features(models, X_ref_t)
        G_ref_t  = pert.matrix(X_ref_t)

        # ── constrained variant: posterior-predictive null ────────
        gen_w = torch.Generator().manual_seed(int(seed) + 80808 + 31 * ti)
        z = torch.randn(K1, generator=gen_w, dtype=torch.float64)
        w_k = w_hat + L_w @ z
        toys_w_k[ti] = w_k.cpu().numpy()

        X_toy_c, F_toy_c = sir_toy(
            F_pool_t, Xq_pool_t, w_k, N_data,
            gen_seed=seed + 7 * ti + 1,
        )
        G_toy_c = pert.matrix(X_toy_c)

        out_c = test_stat_one_variant(
            "constrained",
            F_toy_c, F_ref_t, G_toy_c, G_ref_t,
            w_hat, Sigma_w,
            lam_pert=lam_pert, max_iter=max_iter, tol=tol,
            ridge_rel=ridge_rel, verbose=False,
        )
        t_toys["constrained"][ti] = out_c["t"]
        for side in ("den", "num"):
            fit = out_c[side]
            toy_diag["constrained"][side]["n_eval"][ti]       = fit["n_eval"]
            toy_diag["constrained"][side]["grad_norm"][ti]    = fit["grad_norm"]
            toy_diag["constrained"][side]["converged"][ti]    = fit["converged"]
            toy_diag["constrained"][side]["hit_max_iter"][ti] = fit["hit_max_iter"]

        # ── frozen variant: plug-in null at w = ŵ ─────────────────
        # Different SIR seed offset so toy_c and toy_f are independent
        # resamples from the (shared) q-pool, not the same indices.
        X_toy_f, F_toy_f = sir_toy(
            F_pool_t, Xq_pool_t, w_hat, N_data,
            gen_seed=seed + 7 * ti + 1_000_001,
        )
        G_toy_f = pert.matrix(X_toy_f)

        out_f = test_stat_one_variant(
            "frozen",
            F_toy_f, F_ref_t, G_toy_f, G_ref_t,
            w_hat, Sigma_w,
            lam_pert=lam_pert, max_iter=max_iter, tol=tol,
            ridge_rel=ridge_rel, verbose=False,
        )
        t_toys["frozen"][ti] = out_f["t"]
        for side in ("den", "num"):
            fit = out_f[side]
            toy_diag["frozen"][side]["n_eval"][ti]       = fit["n_eval"]
            toy_diag["frozen"][side]["grad_norm"][ti]    = fit["grad_norm"]
            toy_diag["frozen"][side]["converged"][ti]    = fit["converged"]
            toy_diag["frozen"][side]["hit_max_iter"][ti] = fit["hit_max_iter"]

        if (ti + 1) % max(1, N_toys // 10) == 0:
            elapsed = time.time() - t1
            est_total = elapsed / (ti + 1) * N_toys
            print(f"  toy {ti+1:3d}/{N_toys}  "
                  f"elapsed {elapsed:.0f}s, est total {est_total:.0f}s")

    # ── Posterior-draw diagnostics (constrained only) ────────────
    w_hat_np = w_hat.cpu().numpy()
    sw_diag_np = np.sqrt(np.clip(np.diag(Sigma_w.cpu().numpy()), 1e-30, None))
    pulls_per_dim = (toys_w_k - w_hat_np) / sw_diag_np                # (N_toys, K+1)
    dw_l2 = np.linalg.norm(toys_w_k - w_hat_np, axis=1)               # (N_toys,)
    expected_dw_l2 = np.sqrt(sigma_w_trace)
    print(f"\n  posterior-draw sanity (constrained variant only):")
    print(f"    mean ||w_k - ŵ||_2     = {dw_l2.mean():.3f}  "
          f"(expected ≈ √trace(Σ_w) = {expected_dw_l2:.3f})")
    print(f"    max  ||w_k - ŵ||_2     = {dw_l2.max():.3f}")
    print(f"    per-dim pull |z|: mean = {np.mean(np.abs(pulls_per_dim)):.3f}, "
          f"max = {np.max(np.abs(pulls_per_dim)):.3f}  "
          f"(target half-normal mean ≈ 0.80)")
    np.save(os.path.join(out_dir, "gof_toys_w_k.npy"), toys_w_k)

    # ── p-values, toy convergence summary, save per variant ──────
    print("\n[results] per-variant p-values and toy convergence")
    for v in VARIANTS:
        toys = t_toys[v]
        t_obs = obs[v]["t"]
        n_ge = int(np.sum(toys >= t_obs))
        p_emp = (n_ge + 1) / (N_toys + 1)              # add-one estimator
        med = float(np.median(toys))
        std = float(np.std(toys))

        n_unconv_den = int((~toy_diag[v]["den"]["converged"]).sum())
        n_unconv_num = int((~toy_diag[v]["num"]["converged"]).sum())
        n_maxit_den  = int(toy_diag[v]["den"]["hit_max_iter"].sum())
        n_maxit_num  = int(toy_diag[v]["num"]["hit_max_iter"].sum())
        max_gn_den   = float(toy_diag[v]["den"]["grad_norm"].max())
        max_gn_num   = float(toy_diag[v]["num"]["grad_norm"].max())
        med_neval_den = float(np.median(toy_diag[v]["den"]["n_eval"]))
        med_neval_num = float(np.median(toy_diag[v]["num"]["n_eval"]))
        n_neg_t       = int((toys < 0).sum())

        null_label = ("posterior-predictive (w_k ~ N(ŵ, Σ_w))"
                      if v == "constrained" else "plug-in at ŵ")
        print(f"  [{v:<11s}] null: {null_label}")
        print(f"               t_obs = {t_obs:+.3f}, "
              f"toy median = {med:+.3f}, std = {std:.3f}, "
              f"n_toys ≥ obs = {n_ge}/{N_toys}, p = {p_emp:.3f}")
        print(f"               toy fits:  unconv den/num = {n_unconv_den}/{n_unconv_num},  "
              f"hit-max-iter den/num = {n_maxit_den}/{n_maxit_num},  "
              f"max ||∇|| den/num = {max_gn_den:.1e}/{max_gn_num:.1e},  "
              f"median n_eval den/num = {med_neval_den:.0f}/{med_neval_num:.0f},  "
              f"n_toys with t<0 = {n_neg_t}")
        if n_unconv_den or n_unconv_num:
            print(f"    WARN: some toy fits did not converge (||∇|| > {GRAD_NORM_OK:.0e}). "
                  f"Toy null distribution may be biased.")

        np.save(os.path.join(out_dir, f"gof_{v}_toys_t.npy"), toys)
        np.savez(
            os.path.join(out_dir, f"gof_{v}_toy_diag.npz"),
            den_n_eval=toy_diag[v]["den"]["n_eval"],
            den_grad_norm=toy_diag[v]["den"]["grad_norm"],
            den_converged=toy_diag[v]["den"]["converged"],
            den_hit_max_iter=toy_diag[v]["den"]["hit_max_iter"],
            num_n_eval=toy_diag[v]["num"]["n_eval"],
            num_grad_norm=toy_diag[v]["num"]["grad_norm"],
            num_converged=toy_diag[v]["num"]["converged"],
            num_hit_max_iter=toy_diag[v]["num"]["hit_max_iter"],
        )

        d = obs[v]["den"]; n = obs[v]["num"]
        np.savez(
            os.path.join(out_dir, f"gof_{v}_obs_diag.npz"),
            den_loss_hist=d["loss_hist"],
            num_loss_hist=n["loss_hist"],
            den_grad_norm=np.float64(d["grad_norm"]),
            num_grad_norm=np.float64(n["grad_norm"]),
            den_n_eval=np.int64(d["n_eval"]),
            num_n_eval=np.int64(n["n_eval"]),
            w_den=obs[v]["w_den"],
            w_num=obs[v]["w_num"],
            w_hat=w_hat.cpu().numpy(),
            b_num=(obs[v]["b_num"]
                   if obs[v]["b_num"] is not None
                   else np.zeros(0, dtype=np.float64)),
        )

        results = {
            "variant": v,
            "null_kind": ("posterior_predictive" if v == "constrained"
                          else "plug_in_at_w_hat"),
            "t_obs": float(t_obs),
            "L_den_obs": obs[v]["L_den"],
            "L_num_obs": obs[v]["L_num"],
            "n_toys": int(N_toys),
            "toy_t_mean": float(np.mean(toys)),
            "toy_t_median": med,
            "toy_t_std": std,
            "n_toys_ge_obs": n_ge,
            "p_value_addone": p_emp,
            "n_toys_with_negative_t": n_neg_t,
            "obs_den_grad_norm": float(d["grad_norm"]),
            "obs_num_grad_norm": float(n["grad_norm"]),
            "obs_den_n_eval": int(d["n_eval"]),
            "obs_num_n_eval": int(n["n_eval"]),
            "obs_den_converged": bool(d["converged"]),
            "obs_num_converged": bool(n["converged"]),
            "obs_den_hit_max_iter": bool(d["hit_max_iter"]),
            "obs_num_hit_max_iter": bool(n["hit_max_iter"]),
            "toy_n_unconverged_den": n_unconv_den,
            "toy_n_unconverged_num": n_unconv_num,
            "toy_n_hit_max_iter_den": n_maxit_den,
            "toy_n_hit_max_iter_num": n_maxit_num,
            "toy_max_grad_norm_den": max_gn_den,
            "toy_max_grad_norm_num": max_gn_num,
            "M_pert": M_pert,
            "pert_sigma": pert_sigma,
            "lam_ridge_pert": lam_pert,
            "max_iter": max_iter,
            "tol": tol,
        }
        with open(result_paths[v], "w") as f:
            json.dump(results, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
