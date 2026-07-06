"""
wifi_better_basis pipeline configuration.

Diff vs code/wifi/config.py:
  - N_train and N_test are decoupled (separate config keys, separate cache
    files). The original wifi code's data-cache path put N_train in both
    Ntrain<…> and Ntest<…> slots, so N_test was effectively ignored — fixed
    here, with both sizes encoded in the cache directory name.
  - Wider MLPs (128/128/64), more epochs (600), cosine LR decay.
  - BASIS_VAL_FRAC carves a held-out slice of half_A per run; per-member
    train/val BCE/AUC are saved so you can tell undertraining from capacity.
  - BASIS_REF_OVERSAMPLE inflates the y=0 reference pool per member at no
    asymptotic cost (the constant logit shift is absorbed by the bias column).

Kept torch-free so submit_slurm.py can import on a login node.
"""


CONFIG = {
    # ── Data ─────────────────────────────────────────────────────
    "benchmark": "2d_gmm_skew",  # 2d_gmm_skew | 2d_gaussian | 4d_embedding (real data via make_4d_cache.py)
    "seed": 42,
    # Compute scales with N_train × ref_oversample × epochs × |MLP|. Defaults
    # below are roughly 2-3× the original wifi compute on GPU; bump N_train if
    # you have headroom. N_test is small on purpose so the GoF test isn't
    # crushed by power alone.
    "N_train": 100000,                 # data budget for basis training + linear-head fit
    "N_test": 100000,                   # GoF observed sample + plot_marginals histogram

    # Sean (meeting note): using the SAME data to train the basis and to fit the
    # wifi weights is fine (coverage still holds) and improves marginally, so it
    # is the default. True  -> full X_train feeds BOTH basis and linear head
    # (no 50/50 split); a small BASIS_VAL_FRAC slice is still held out for basis
    # BCE/AUC monitoring only. False -> legacy 50/50 split (basis on half_A,
    # linhead + honest held-out covariance on half_B). Overridable per run with
    # `python run.py --same-data {0,1}`.
    "SAME_DATA_BASIS_WIFI": True,

    # ── Bootstrapped MLP basis ───────────────────────────────────
    "K": 160,                           # number of basis MLPs (linear-head dim is K+1)
    "MLP_HIDDEN": [32, 32, 16],
    "BASIS_EPOCHS": 600,
    "BASIS_LR": 1e-3,
    "BASIS_LR_SCHEDULE": "cosine",      # "constant" or "cosine" (cosine -> eta_min = lr/100)
    "BASIS_WEIGHT_DECAY": 1e-4,
    "BASIS_BATCH_SIZE": 4096,           # None for full-batch
    "BASIS_VAL_FRAC": 0.10,             # held-out fraction of half_A used for val BCE/AUC tracking
    "BASIS_REF_OVERSAMPLE": 4,          # ref pool size per member = OVERSAMPLE × |bootstrap|

    # ── Linear head fit (held-out half) ──────────────────────────
    "N_REF_LINHEAD": None,              # None -> match held-out data size
    "LINHEAD_MAX_ITER": 500,
    "LINHEAD_RIDGE": 10,                # L2 ridge on wifi BCE loss (Sean: lambda=10 for K=128)

    # ── Covariance estimators ────────────────────────────────────
    "SANDWICH_RIDGE_REL": 1e-8,
    "BOOTSTRAP_B": 200,
    "BOOTSTRAP_LBFGS_MAX_ITER": 200,

    # ── Misc ─────────────────────────────────────────────────────
    # "auto" picks cuda when available, else cpu. With wider MLPs and 600
    # epochs over K=64 members, GPU is meaningfully faster; CPU still works.
    "DEVICE": "auto",

    # ── Coverage test ────────────────────────────────────────────
    "COVERAGE_N_PSEUDOEXP": 100,
    "COVERAGE_M_SIR": 200000,

    # ── Classifier GoF ───────────────────────────────────────────
    "GOF_M_PERT": 80,
    "GOF_PERT_SIGMA": 0.10,
    "GOF_LAM_RIDGE_PERT": 1e-3,
    "GOF_MAX_ITER": 500,
    "GOF_TOL": 1e-9,
    "GOF_N_TOYS": 100,
    "GOF_TOY_OVERSAMPLE": 10,
    # GOF_N_REF = None -> N_ref = N_data (1:1). This is CORRECT at 1:1 but does NOT
    # implement the paper's reference oversampling (NPLM-style ~5:1). Do not just
    # raise it: the BCE in classifier_gof.py is summed, so N_ref != N_data biases t.
    # Fix + plan tracked in ../OPEN_PROBLEMS.md ("Reference oversampling ...").
    "GOF_N_REF": None,
}


def make_run_name(cfg):
    h_tag = "x".join(str(h) for h in cfg["MLP_HIDDEN"])
    du_tag = "samedata" if cfg.get("SAME_DATA_BASIS_WIFI", False) else "split"
    return (f"wbb_{cfg['benchmark']}_K{cfg['K']}_H{h_tag}_"
            f"Ntr{cfg['N_train']}_Nte{cfg['N_test']}_s{cfg['seed']}_{du_tag}")
