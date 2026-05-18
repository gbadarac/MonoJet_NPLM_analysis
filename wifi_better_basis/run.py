"""
wifi_better_basis pipeline runner.

Diff vs code/wifi/run.py:
  - N_train and N_test are decoupled. The data cache directory name encodes
    BOTH sizes correctly (the original wifi code put N_train in the Ntest
    slot too; that bug is fixed here, in run_gof.py, and in plot_marginals.py).
  - half_A is split once more into a basis train pool and a held-out val
    pool (BASIS_VAL_FRAC). The val pool drives per-member train/val BCE/AUC
    diagnostics; the val pool is NOT used for the linear-head fit.
  - Adds an SVD-based effective-rank diagnostic on F_data so you can tell
    whether adding basis members is buying you new directions.
  - DEVICE accepts "auto" / "cuda" / "cpu"; "auto" picks cuda when available
    else cpu. Basis training and feature evaluation use the resolved device.

Pipeline:
  1. Load (or generate) train and test data from a benchmark.
  2. Fit Gaussian reference q on the full training set.
  3. 50/50 split of train into half_A (basis training) and half_B
     (linear-head fit, sandwich, bootstrap covariance).
  4. Inside half_A, carve a val pool (BASIS_VAL_FRAC); rest is the basis
     train pool.
  5. Train K bootstrap-MLP basis members on the train pool + fresh
     oversampled q-samples; track val BCE/AUC per member.
  6. Build feature matrices on half_B + a fresh q-sample.
  7. Fit linear head w via BCE; compute sandwich Σ_w and bootstrap Σ_w.
  8. Wald χ² sanity test on (w_hat, Σ_w) against w = 0.
  9. Save artifacts (including basis_diag.json with per-member metrics and
     F_data effective-rank diagnostics).
 10. Marginal plots.

Usage:
    python run.py
    python run.py --name wbb_test
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch

# ── Resolve package dir (data/ and runs/ live next to this script) ───
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = PACKAGE_DIR
sys.path.insert(0, PACKAGE_DIR)

from benchmarks import get_benchmark
from config import CONFIG, make_run_name
from reference import fit_gaussian_reference, sample_reference
from basis import train_bootstrap_basis, evaluate_features
from wifi_train import (
    fit_linear_head, sandwich_cov, naive_inv_hessian_cov, bootstrap_cov,
)
from wald_gof import wald_chi2


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _resolve_device(spec):
    """Map a DEVICE config entry ("auto" / "cuda" / "cpu") to a concrete
    device string. "auto" picks cuda if available else cpu; explicit "cuda"
    falls back to cpu with a warning when CUDA is missing."""
    spec = (spec or "auto").lower()
    cuda_ok = torch.cuda.is_available()
    if spec == "auto":
        return "cuda" if cuda_ok else "cpu"
    if spec == "cuda" and not cuda_ok:
        print("      WARNING: DEVICE=cuda requested but CUDA unavailable; "
              "falling back to cpu.")
        return "cpu"
    return spec


def _data_dir_name(cfg):
    return (
        f"{cfg['benchmark']}_Ntrain{cfg['N_train']}_Ntest{cfg['N_test']}"
        f"_seed{cfg['seed']}"
    )


def load_or_generate_data(cfg):
    """Load (or generate and cache) train and test samples of sizes N_train
    and N_test. Cached under data/<benchmark>_Ntrain<…>_Ntest<…>_seed<…>/."""
    data_dir = os.path.join(PROJECT_ROOT, "data", _data_dir_name(cfg))
    train_path = os.path.join(data_dir, "data_train.npy")
    test_path  = os.path.join(data_dir, "data_test.npy")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  reusing cached data at {data_dir}")
        X_train = np.load(train_path)
        X_test  = np.load(test_path)
    else:
        os.makedirs(data_dir, exist_ok=True)
        gen = get_benchmark(cfg["benchmark"])
        X_train = gen(cfg["N_train"], seed=cfg["seed"])
        X_test  = gen(cfg["N_test"],  seed=cfg["seed"] + 1)
        np.save(train_path, X_train)
        np.save(test_path,  X_test)
        with open(os.path.join(data_dir, "data_config.json"), "w") as f:
            json.dump({
                "benchmark": cfg["benchmark"],
                "N_train": cfg["N_train"], "N_test": cfg["N_test"],
                "seed": cfg["seed"], "d": X_train.shape[1],
                "data_dir": data_dir,
            }, f, indent=2)
        print(f"  generated and cached data at {data_dir}")
    return X_train, X_test


def split_5050(X, seed):
    """Deterministic 50/50 split into (half_A, half_B)."""
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(X.shape[0])
    half = X.shape[0] // 2
    return X[perm[:half]], X[perm[half:]]


def carve_val(X_A, val_frac, seed):
    """Deterministic split of half_A into (train_pool, val_pool). Uses a
    different seed offset than split_5050 so the two splits are independent."""
    rng = np.random.RandomState(int(seed) + 424_242)
    perm = rng.permutation(X_A.shape[0])
    n_val = int(round(val_frac * X_A.shape[0]))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    return X_A[train_idx], X_A[val_idx]


def f_data_effective_rank(F):
    """Return SVD-based effective-rank diagnostics of (centred) F. Useful
    for catching basis collapse: K+1 columns but ⟨effective rank⟩ ≪ K+1
    means the basis members are highly correlated and Σ_w is poorly
    conditioned."""
    F_np = F.cpu().numpy().astype(np.float64)
    F_c = F_np - F_np.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(F_c, compute_uv=False)
    sv0 = max(sv[0], 1e-30)
    sv_norm = sv / sv0
    eff_1pct  = int((sv_norm > 0.01).sum())
    eff_10pct = int((sv_norm > 0.10).sum())
    sv_sq = sv ** 2
    p = sv_sq / max(sv_sq.sum(), 1e-30)
    entropy_eff = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    return {
        "K1": int(F_np.shape[1]),
        "n_rows": int(F_np.shape[0]),
        "singular_values": sv.tolist(),
        "sv_norm_top": sv_norm[: min(10, len(sv_norm))].tolist(),
        "eff_rank_1pct":  eff_1pct,
        "eff_rank_10pct": eff_10pct,
        "entropy_eff_rank": entropy_eff,
    }


def basis_diag_summary(diags):
    """Aggregate per-member diagnostics into mean/min/max."""
    arr = lambda key: np.array([d[key] for d in diags], dtype=np.float64)
    out = {"per_member": diags}
    for k in ("train_bce_first", "train_bce_final",
              "val_bce_first", "val_bce_final",
              "val_auc_first", "val_auc_final"):
        a = arr(k)
        out[k] = {
            "mean": float(np.nanmean(a)),
            "std":  float(np.nanstd(a)),
            "min":  float(np.nanmin(a)),
            "max":  float(np.nanmax(a)),
        }
    return out


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.name or make_run_name(CONFIG)
    out_dir = os.path.join(PROJECT_ROOT, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "wifi_config.json")
    with open(cfg_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    device = _resolve_device(CONFIG.get("DEVICE", "auto"))

    print("=" * 62)
    print("wifi_better_basis pipeline (Gaussian reference, no GoF)")
    print("=" * 62)
    print(f"  Run:    {run_name}")
    print(f"  Output: {out_dir}")
    print(f"  Device: {device}")
    print()

    t0 = time.time()
    torch.manual_seed(int(CONFIG["seed"]))

    # 1. Data
    print("[1/9] load data")
    X_train, X_test = load_or_generate_data(CONFIG)
    d = X_train.shape[1]
    print(f"      X_train shape = {X_train.shape}, X_test shape = {X_test.shape}, d = {d}")

    # 2. Fit reference on full training data
    print("[2/9] fit Gaussian reference q(x) on full X_train")
    Xt = torch.from_numpy(X_train).double()
    mu_q, Sigma_q = fit_gaussian_reference(Xt)
    print(f"      mu    = {mu_q.numpy()}")
    print(f"      Sigma = {Sigma_q.numpy().tolist()}")

    # 3. 50/50 split (basis training half / linear-head fit half)
    print("[3/9] 50/50 split (half_A: basis,  half_B: linhead)")
    X_A, X_B = split_5050(X_train, seed=CONFIG["seed"])
    print(f"      half_A = {X_A.shape[0]} (basis training)")
    print(f"      half_B = {X_B.shape[0]} (linhead + Σ_w)")

    # 4. Carve a val pool out of half_A for basis diagnostics
    print(f"[4/9] carve val pool ({CONFIG['BASIS_VAL_FRAC']*100:.0f}% of half_A)")
    X_A_train, X_A_val = carve_val(X_A, CONFIG["BASIS_VAL_FRAC"], seed=CONFIG["seed"])
    print(f"      basis train pool = {X_A_train.shape[0]}")
    print(f"      val pool         = {X_A_val.shape[0]}")
    X_A_train_t = torch.from_numpy(X_A_train).double()
    X_A_val_t   = torch.from_numpy(X_A_val).double()
    X_B_t       = torch.from_numpy(X_B).double()

    # 5. Train bootstrap basis on the train pool
    print(f"[5/9] train basis (K={CONFIG['K']} MLPs, hidden={CONFIG['MLP_HIDDEN']}, "
          f"epochs={CONFIG['BASIS_EPOCHS']}, schedule={CONFIG['BASIS_LR_SCHEDULE']}, "
          f"ref_oversample={CONFIG['BASIS_REF_OVERSAMPLE']}x) on device={device}")
    models, basis_diags = train_bootstrap_basis(
        X_A_train_t, X_A_val_t, mu_q, Sigma_q,
        K=CONFIG["K"], hidden=tuple(CONFIG["MLP_HIDDEN"]),
        epochs=CONFIG["BASIS_EPOCHS"], lr=CONFIG["BASIS_LR"],
        weight_decay=CONFIG["BASIS_WEIGHT_DECAY"],
        batch_size=CONFIG["BASIS_BATCH_SIZE"],
        lr_schedule=CONFIG["BASIS_LR_SCHEDULE"],
        ref_oversample=int(CONFIG["BASIS_REF_OVERSAMPLE"]),
        device=device, seed=CONFIG["seed"], verbose=True,
    )
    basis_dir = os.path.join(out_dir, "basis")
    os.makedirs(basis_dir, exist_ok=True)
    for k, m in enumerate(models):
        torch.save(m.cpu().state_dict(),
                   os.path.join(basis_dir, f"basis_{k:02d}.pt"))

    # Surface aggregate basis quality
    train_bce_arr = np.array([d["train_bce_final"] for d in basis_diags])
    val_bce_arr   = np.array([d["val_bce_final"]   for d in basis_diags])
    val_auc_arr   = np.array([d["val_auc_final"]   for d in basis_diags])
    print(f"      basis diagnostics across K={CONFIG['K']} members:")
    print(f"        train BCE: mean={train_bce_arr.mean():.4f}  "
          f"min={train_bce_arr.min():.4f}  max={train_bce_arr.max():.4f}")
    print(f"        val   BCE: mean={val_bce_arr.mean():.4f}  "
          f"min={val_bce_arr.min():.4f}  max={val_bce_arr.max():.4f}")
    print(f"        val   AUC: mean={val_auc_arr.mean():.4f}  "
          f"min={val_auc_arr.min():.4f}  max={val_auc_arr.max():.4f}")
    gap = float(np.mean(val_bce_arr - train_bce_arr))
    print(f"        mean (val − train) BCE gap = {gap:+.4f} "
          f"(positive = generalization gap; ≈0 = saturated)")
    log2 = float(np.log(2.0))
    print(f"        reference: log(2) = {log2:.4f} = chance-level BCE (basis useless if val BCE ≈ this)")

    # 6. Build feature matrices on half_B + fresh reference draw
    print("[6/9] build feature matrices on half_B + fresh q-sample")
    N_ref = CONFIG["N_REF_LINHEAD"] or X_B.shape[0]
    X_ref_lh = sample_reference(mu_q, Sigma_q, N_ref, seed=CONFIG["seed"] + 9991)
    F_data = evaluate_features(models, X_B_t,    device=device)
    F_ref  = evaluate_features(models, X_ref_lh, device=device)
    F_data = F_data.cpu()
    F_ref  = F_ref.cpu()
    print(f"      F_data shape = {tuple(F_data.shape)}, "
          f"F_ref shape = {tuple(F_ref.shape)}")

    # Effective rank of F_data
    eff = f_data_effective_rank(F_data)
    print(f"      F_data effective rank: K+1 = {eff['K1']}  "
          f"eff(>1%) = {eff['eff_rank_1pct']}  "
          f"eff(>10%) = {eff['eff_rank_10pct']}  "
          f"entropy_eff = {eff['entropy_eff_rank']:.2f}")
    if eff['eff_rank_1pct'] < eff['K1'] // 2:
        print(f"      WARN: effective rank ≪ K+1 — basis members are highly "
              f"correlated; adding K may not be buying you new directions.")

    # 7. Fit linear head + covariances
    print("[7/9] fit linear head and covariances")
    w_hat, final_loss = fit_linear_head(
        F_data, F_ref, max_iter=CONFIG["LINHEAD_MAX_ITER"], verbose=True,
    )
    print(f"      w_hat = {w_hat.numpy()}")

    Sigma_sw, H, J = sandwich_cov(
        w_hat, F_data, F_ref, ridge_rel=CONFIG["SANDWICH_RIDGE_REL"],
    )
    Sigma_naive = naive_inv_hessian_cov(
        w_hat, F_data, F_ref, ridge_rel=CONFIG["SANDWICH_RIDGE_REL"],
    )
    print(f"      sandwich diag: {Sigma_sw.diag().numpy()}")
    print(f"      naive H^-1 diag: {Sigma_naive.diag().numpy()}")

    print(f"      bootstrap covariance (B={CONFIG['BOOTSTRAP_B']})...")
    Sigma_bs, Ws_boot, w_boot_mean = bootstrap_cov(
        F_data, F_ref, B=CONFIG["BOOTSTRAP_B"],
        max_iter=CONFIG["BOOTSTRAP_LBFGS_MAX_ITER"],
        seed=CONFIG["seed"] + 5555, verbose=True,
    )
    print(f"      bootstrap diag: {Sigma_bs.diag().numpy()}")

    diag_ratio = (Sigma_sw.diag() / Sigma_bs.diag().clamp_min(1e-30)).numpy()
    print(f"      diag ratio (sandwich / bootstrap): {diag_ratio}")

    # 8. Wald sanity test
    print("[8/9] Wald χ² against w = 0  (tests 'is q sufficient?')")
    wald_sw = wald_chi2(w_hat, Sigma_sw)
    wald_bs = wald_chi2(w_hat, Sigma_bs)
    print(f"      sandwich:  T = {wald_sw['T']:.3f}  dof = {wald_sw['dof']}  "
          f"p = {wald_sw['p_value']:.3e}")
    print(f"      bootstrap: T = {wald_bs['T']:.3f}  dof = {wald_bs['dof']}  "
          f"p = {wald_bs['p_value']:.3e}")

    # ── Save artifacts ────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "w_hat.npy"), w_hat.numpy())
    np.save(os.path.join(out_dir, "Sigma_w_sandwich.npy"), Sigma_sw.numpy())
    np.save(os.path.join(out_dir, "Sigma_w_naive_inv_hessian.npy"),
            Sigma_naive.numpy())
    np.save(os.path.join(out_dir, "Sigma_w_bootstrap.npy"), Sigma_bs.numpy())
    np.save(os.path.join(out_dir, "bootstrap_w_estimates.npy"), Ws_boot.numpy())
    np.save(os.path.join(out_dir, "mu_q.npy"), mu_q.numpy())
    np.save(os.path.join(out_dir, "Sigma_q.npy"), Sigma_q.numpy())

    with open(os.path.join(out_dir, "basis_diag.json"), "w") as f:
        json.dump({
            "summary": basis_diag_summary(basis_diags),
            "f_data_effective_rank": eff,
        }, f, indent=2)

    summary = {
        "run_name": run_name,
        "benchmark": CONFIG["benchmark"],
        "seed": CONFIG["seed"],
        "N_train": CONFIG["N_train"],
        "N_test": CONFIG["N_test"],
        "K": CONFIG["K"],
        "MLP_HIDDEN": CONFIG["MLP_HIDDEN"],
        "BASIS_EPOCHS": CONFIG["BASIS_EPOCHS"],
        "BASIS_LR_SCHEDULE": CONFIG["BASIS_LR_SCHEDULE"],
        "BASIS_VAL_FRAC": CONFIG["BASIS_VAL_FRAC"],
        "BASIS_REF_OVERSAMPLE": CONFIG["BASIS_REF_OVERSAMPLE"],
        "device": device,
        "linhead_dim_K_plus_1": int(w_hat.numel()),
        "bce_final_loss": final_loss,
        "w_hat": w_hat.numpy().tolist(),
        "sandwich_diag": Sigma_sw.diag().numpy().tolist(),
        "naive_inv_hessian_diag": Sigma_naive.diag().numpy().tolist(),
        "bootstrap_diag": Sigma_bs.diag().numpy().tolist(),
        "bootstrap_mean_w": w_boot_mean.numpy().tolist(),
        "diag_ratio_sandwich_over_bootstrap": diag_ratio.tolist(),
        "wald_sandwich":  {k: v for k, v in wald_sw.items() if k != "Sigma_inv"},
        "wald_bootstrap": {k: v for k, v in wald_bs.items() if k != "Sigma_inv"},
        "basis_train_bce_mean": float(train_bce_arr.mean()),
        "basis_val_bce_mean":   float(val_bce_arr.mean()),
        "basis_val_auc_mean":   float(val_auc_arr.mean()),
        "basis_val_minus_train_bce_gap": gap,
        "f_data_eff_rank_1pct":  eff["eff_rank_1pct"],
        "f_data_eff_rank_10pct": eff["eff_rank_10pct"],
        "f_data_entropy_eff_rank": eff["entropy_eff_rank"],
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "wifi_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 9. Marginal plots
    print("[9/9] marginal plots")
    try:
        from plot_marginals import plot_marginals
        plot_marginals(out_dir)
    except Exception as e:
        print(f"      warn: plot_marginals failed: {e}")

    print(f"\nDone in {summary['elapsed_seconds']:.1f}s. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
