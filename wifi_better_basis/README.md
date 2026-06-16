# wifi_better_basis

Self-contained pipeline for fitting a wifi-reweighted density model
`p̂(x) = r̂(x) q(x) / Z` and quantifying uncertainty on it:

- **`r̂(x) = w · features(x)`**, where `features(x)` is a frozen bank of
  bootstrap-trained MLP basis functions and `w` is a linear head fit by
  BCE against samples from a Gaussian reference `q`.
- Three covariance estimates for `ŵ`: sandwich, naïve `H⁻¹`, and bootstrap.
- Frequentist coverage test on the first-moment functional `⟨x_0⟩` via
  fresh-DGP pseudoexperiments.
- Classifier goodness-of-fit (NPLM-style) with three variants
  (`constrained`, `free`, `frozen`) and toy-MC calibration.

## Setup

```bash
pip install -r requirements.txt
```

Tested with PyTorch ≥ 2.0. A GPU helps for basis training but is not
required — set `DEVICE` in `config.py` to `"cpu"`, `"cuda"`, or `"auto"`.

## Layout

```
.
├── run.py                  # main pipeline: data → basis → linear head → Σ_w → plots
├── coverage.py             # frequentist coverage on ⟨x_0⟩
├── run_gof.py              # classifier GoF (constrained / free / frozen)
├── plot_gof.py             # post-hoc GoF figures from saved results
├── plot_marginals.py       # marginal plots vs analytic / DGP
├── submit_slurm.py         # cluster submission helper (optional)
│
├── config.py               # all hyperparameters live here; edit + re-run
├── benchmarks.py           # toy DGPs (2d_gaussian, 2d_gmm_skew)
├── reference.py            # Gaussian reference q(x)
├── basis.py                # MLP basis: training, evaluation, bootstrap
├── wifi_train.py           # linear head BCE fit + sandwich/bootstrap Σ_w
├── classifier_gof.py       # NPLM-style test statistic + toy calibration
├── wald_gof.py             # Wald χ² helper
│
├── data/                   # cached benchmark draws (auto-populated by run.py)
│   └── <benchmark>_Ntrain<N>_Ntest<N>_seed<s>/
│       ├── data_train.npy
│       ├── data_test.npy
│       └── data_config.json
└── runs/                   # one folder per run, written by run.py
    └── <run_name>/
        ├── wifi_config.json
        ├── w_hat.npy
        ├── Sigma_w_{sandwich,naive_inv_hessian,bootstrap}.npy
        ├── basis/basis_<k>.pt           # K frozen MLPs
        ├── basis_diag.json              # per-member train/val BCE/AUC + SVD rank
        ├── marginals.png
        ├── coverage_*.{npy,png,json}    # from coverage.py
        └── gof_{constrained,free,frozen}_*.{npy,npz,json}  # from run_gof.py
```

## Running

The whole pipeline lives in the same directory. Just `cd` in and run:

```bash
cd wifi_better_basis_share

# 1. Fit basis + linear head + covariances (~minutes on GPU, longer on CPU)
python run.py --name my_run

# 2. Frequentist coverage on ⟨x_0⟩ (pseudoexperiments)
python coverage.py --name my_run

# 3. Classifier GoF (all three variants, with toy MC calibration)
python run_gof.py --name my_run
python run_gof.py --name my_run --force      # ignore variant cache

# 4. GoF diagnostic plots from cached results
python plot_gof.py --name my_run
```

Each step is independent; `coverage.py`, `run_gof.py`, and `plot_gof.py`
just read the artifacts written by `run.py`.

## Configuring

All knobs live in `config.py`'s `CONFIG` dict. Common ones:

| Key | Meaning |
|---|---|
| `benchmark` | DGP key (`2d_gaussian`, `2d_gmm_skew`) |
| `seed`, `N_train`, `N_test` | Data |
| `K`, `MLP_HIDDEN`, `BASIS_EPOCHS` | Basis capacity / training |
| `BASIS_LR_SCHEDULE` | `"constant"` or `"cosine"` |
| `BASIS_VAL_FRAC` | Held-out slice of half_A for per-member BCE/AUC |
| `BOOTSTRAP_B` | Number of bootstrap replicates for Σ_w |
| `GOF_N_TOYS`, `GOF_M_PERT` | GoF toy count + perturbation basis size |
| `GOF_FREE_COV_INFLATE` | Covariance multiplier for the `free` variant |
| `DEVICE` | `"auto"` / `"cuda"` / `"cpu"` |

Editing `CONFIG` is the normal way to change a run. The auto-generated
run name is derived from the config; pass `--name` to override.

To add a benchmark, write `generate_<name>(N, seed) -> ndarray` in
`benchmarks.py` and register it under `BENCHMARKS` (optionally add a
1-D marginal under `MARGINALS` for `plot_marginals.py`).

## Reading a run

Quick orientation when you open a `runs/<name>/` folder:

- `wifi_summary.json` — top-level scalars: w_hat, Σ_w diagonals, Wald
  test stats, basis BCE/AUC summary, F_data effective rank.
- `basis_diag.json` — per-member training diagnostics + SVD effective
  rank of the feature matrix. Effective rank ≪ K+1 means the basis
  members are highly correlated.
- `marginals.png` — fitted vs DGP marginals.
- `coverage_pulls.png` — pseudoexperiment pulls; 1σ/2σ coverage should
  hit ~68% / 95% if Σ_w is well-calibrated.
- `gof_<variant>_results.json` — observed test stat, toy distribution
  summary, and p-value for each GoF variant.
- `gof_diagnostics.png`, `gof_calibration.png` — GoF figures from
  `plot_gof.py`.

## Sample run

`runs/wifi_better_basis_smaller_nets/` is a complete reference run on
`2d_gmm_skew` with `N_train=200000, N_test=20000, seed=42`. The
matching cached data shard ships under
`data/2d_gmm_skew_Ntrain200000_Ntest20000_seed42/`. Inspect those
artifacts before running anything to know what a healthy run looks
like.

## Notes

- All run/data paths are resolved relative to the package directory
  (the folder containing this README). You can move the whole tree.
- `submit_slurm.py` is included for completeness; the SLURM defaults
  (partitions, conda env name, email) are MIT/Harvard-specific and
  will need to be edited for your cluster.
- There are no tests / no lint; this is a research pipeline.
