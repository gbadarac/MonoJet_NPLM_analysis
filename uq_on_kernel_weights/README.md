# HIKER Hessian UQ

Hessian-based uncertainty quantification for the HIKER (Hierarchical Ensemble
of Gaussian Kernels) density estimator, with goodness-of-fit testing via
learned likelihood ratios (NPLM framework).

## Structure

```
hiker_hessian_uq/
  code/
    hiker.py              -- Multi-layer HIKER model, ensemble builder, sampling
    hessian_uq.py         -- Hessian computation, covariance inversion, predictive variance
    gof.py                -- Likelihood-ratio GoF test (with Gaussian constraint)
    gof_plots.py          -- Shared GoF plotting functions (ratio to data/true/HIKER)
    benchmarks.py         -- Registry of data distributions (2d_gaussian, 2d_gmm_skew, ...)
    generate_data.py      -- Step 1: generate train/test data (shared across runs)
    train_and_uq.py       -- Step 2: train HIKER (single or ensemble), Hessian, marginals
    plot_marginals.py     -- Step 2b: marginal density plots + ratio + covariance matrix
    coverage.py           -- Step 3: bootstrap coverage test for first-moment observable
    run_gof.py            -- Step 4: GoF test (3 variants) with toy-based calibration
    plot_gof.py           -- Step 4b: re-plot all GoF diagnostics from saved results
    run_pipeline.py       -- Pipeline runner: configure and run steps locally
    submit_slurm.py       -- Submit pipeline steps to SLURM cluster
  data/                   -- Shared data directory (one folder per benchmark/N/seed)
  runs/                   -- Model runs (one folder per configuration)
```

## Usage

### Local execution

Edit `CONFIG` in `code/run_pipeline.py`, then:

```bash
cd code
python run_pipeline.py                              # full pipeline
python run_pipeline.py --steps data train plots     # specific steps
python run_pipeline.py --steps plots                # re-plot without retraining
python run_pipeline.py --steps gof --force          # re-run GoF (skip cache)
python run_pipeline.py --steps plot_gof             # re-plot GoF diagnostics
python run_pipeline.py --dry-run                    # see what would run
python run_pipeline.py --name my_run                # custom folder name
```

### Run on existing folder

```bash
# Via pipeline --name
python run_pipeline.py --steps gof --name my_existing_run_folder_name

# Or directly with env var
HIKER_OUT_DIR=/path/to/runs/my_run python run_gof.py
HIKER_OUT_DIR=/path/to/runs/my_run python run_gof.py --force
```

### SLURM cluster

```bash
cd code
python submit_slurm.py --steps data train plots     # submit as one job
python submit_slurm.py --steps gof --force           # submit GoF step
python submit_slurm.py --steps train --time 8:00:00 --mem 32000
python submit_slurm.py --steps data --no-gpu         # CPU-only job
python submit_slurm.py --steps train --dry-run       # preview batch script
```

## Pipeline steps

| Step | Script | What it does |
|------|--------|-------------|
| `data` | `generate_data.py` | Generate train/test data. Saved in shared `data/` dir; skips if already exists |
| `train` | `train_and_uq.py` | Train HIKER (single or ensemble), compute Hessian covariance, precompute marginals. Plots: training loss (with linear fit), coefficient evolution (raw + normalized by Z), centroid evolution, eigenspectrum |
| `plots` | `plot_marginals.py` | Marginal density plots with +/-1,2 sigma bands, ratio to true, covariance/correlation matrix. Shows per-seed curves for ensemble |
| `coverage` | `coverage.py` | Bootstrap coverage test for first-moment observable |
| `gof` | `run_gof.py` | GoF LRT in 3 variants (constrained/free/frozen), then toy calibration. Each GoF config gets its own subfolder. Skips already-computed variants (use `--force` to re-run). GoF subfolder name includes norm/nonorm tag |
| `plot_gof` | `plot_gof.py` | Re-generate all GoF plots from saved model weights. Iterates over all `gof_*` subdirectories |

## Configuration

All hyperparameters are set in the `CONFIG` dict in `code/run_pipeline.py`:

### Data
| Key | Default | Description |
|-----|---------|-------------|
| `benchmark` | `"2d_gaussian"` | Distribution name (see `benchmarks.py`) |
| `seed` | `42` | Random seed |
| `N_train` | `100000` | Total training data generated |
| `N_test` | `100000` | Test set size |

### HIKER model
| Key | Default | Description |
|-----|---------|-------------|
| `M_KERNELS` | `50` | Kernels per layer. `int` for single layer, `list` for multi-layer |
| `KERNEL_SIGMA` | `0.10` | Final kernel width per layer. Follows same int/list convention |
| `WIDTH_INIT` | _(omit)_ | Optional: starting width for annealing (anneals down to `KERNEL_SIGMA`) |
| `LR` | `1e-4` | Learning rate |
| `EPOCHS` | `100000` | Epochs per layer (int or list) |
| `PATIENCE` | `5000` | Print/eval interval |
| `BATCH_SIZE` | `None` | Mini-batch size (`None` = full batch) |
| `TRAIN_MODE` | `"sequential"` | `"sequential"` (layer-by-layer) or `"joint"` (all at once) |
| `EARLY_STOPPING` | `None` | Stop after N evals without improvement (`None` = off). Works in both sequential and joint modes |
| `TRAIN_CENTROIDS` | `True` | Optimise kernel centres |
| `USE_NORM_KERNEL` | `True` | `True` = normalization kernel gauge, `False` = post-hoc + projection |
| `TRAIN_NORM_CENTRE` | `False` | Train the norm kernel centre (ignored if `USE_NORM_KERNEL=False`) |
| `COEFFS_CLIP` | `100.0` | Clamp weights to +/-clip (`None` = no clipping) |
| `N_ENSEMBLE` | `1` | Number of bootstrap models. `1` = single model, `>1` = ensemble averaging |
| `N_TRAIN_PER_MODEL` | `None` | Training points per model (`None` = use all `N_train`). For ensemble: bootstrap sample size |

### Coverage
| Key | Default | Description |
|-----|---------|-------------|
| `N_BOOTSTRAP` | `50` | Number of bootstrap resamples |

### GoF test
| Key | Default | Description |
|-----|---------|-------------|
| `GOF_LABEL` | _(auto)_ | Custom subfolder name. Auto-generated from hyperparameters + norm/nonorm tag if omitted |
| `N_TEST_GOF` | `20000` | Test points per LRT (subsampled from test set for observed; fresh samples for toys) |
| `N_KERNELS_NUM` | `80` | Number of perturbation kernels in numerator |
| `KERNEL_WIDTH_NUM` | `0.10` | Width of perturbation kernels |
| `CLIP_NUM` | `0.2` | Clipping for perturbation coefficients |
| `EPOCHS_DEN` | `30000` | Training epochs for denominator |
| `EPOCHS_NUM` | `80000` | Training epochs for numerator |
| `LR_DEN` / `LR_NUM` | `5e-5` | Learning rates for GoF training |
| `PATIENCE_GOF` | `2000` | Eval interval for GoF training |
| `N_TOYS` | `30` | Number of calibration toys for empirical p-value |

## Available benchmarks

| Name | Description | Dimensions |
|------|-------------|------------|
| `2d_gaussian` | N([-0.5, 0.6], diag(0.25^2, 0.4^2)) | 2 |
| `2d_gmm_skew` | Bimodal GMM in x0 + skew-normal in x1 | 2 |

Add new benchmarks in `code/benchmarks.py`: write a generator function and register
it in `BENCHMARKS` (and optionally `MARGINALS` for analytic true marginals).

## Ensemble mode

When `N_ENSEMBLE > 1`, the `train` step:

1. Trains N models on bootstrap resamples (each in `seed{i}/` subdirectory with its own plots and Hessian)
2. Collects all kernels from all models into one ensemble HIKER
3. Handles normalization depending on per-seed `USE_NORM_KERNEL`:
   - **`True`**: per-seed norm kernels are stored separately (`ensemble_info.npz`). Each seed's norm weight `w_norm_s = 1/N - sum(w_s)` keeps the contribution self-normalizing. Covariance is block-diagonal scaled by `1/N^2`
   - **`False`**: weights rescaled by `1/(N*Z_s)`. Post-hoc normalization. Covariance scaled by `1/(N*Z_s)^2`
4. Saves ensemble `model.pt`, `covariance.npy`, and `ensemble_info.npz` at the run root

Marginal plots show per-seed curves (thin gray) overlaid with the ensemble average
(blue with Hessian uncertainty bands). Downstream scripts (coverage, GoF) use the
ensemble model transparently.

Use `N_TRAIN_PER_MODEL` to control how many points each model sees:
```python
"N_train": 100000,
"N_ENSEMBLE": 5,
"N_TRAIN_PER_MODEL": 20000,  # each model bootstraps 20k from 100k
```

## Gauge freedom

HIKER's density is a sum of Gaussian kernels with trainable amplitudes. The NLL is
invariant under a global shift of all amplitudes, creating a flat direction in the
loss landscape and a rank-deficient Hessian. Two approaches are supported:

### Normalization kernel (`USE_NORM_KERNEL=True`, default)

A single global normalization kernel with derived weight `w_norm = 1 - sum(w)`:
```
f(x) = sum_l sum_i w_i^(l) K_i^(l)(x) + w_norm * K_norm(x)
```
The Hessian w.r.t. the free weights is full-rank and directly invertible.
The norm kernel centre is initialised at the dataset mean (frozen by default).

### Post-hoc normalization (`USE_NORM_KERNEL=False`)

No norm kernel. Training uses the unnormalized density `g(x) = sum w_i K_i(x)`.
The normalized density `f(x) = g(x) / Z` (`Z = sum w`) is used for evaluation, UQ, and the GoF test.
The Hessian is computed on the normalized NLL and has one flat direction `(1,...,1)`
which is projected out before inversion.

Note: training uses the unnormalized NLL (`-log(g + eps)`) but the Hessian is computed
on the normalized NLL (`-log(g/Z)`). These coincide up to the gauge direction, which
is projected out.

## GoF test

### Density in the GoF model

The GoF always uses the **normalized** density for both numerator and denominator:
- **Denominator**: `p(x) = g(x) / Z` (or `g(x)` with norm kernel)
- **Numerator**: `p(x) = g(x) / Z + perturbation(x)` (perturbation added after normalization)

This ensures both are on the same scale. The perturbation magnitude (controlled by
`CLIP_NUM`) is relative to the normalized density, independent of Z.

### Variants

| Variant | Suffix | Weights | Constraint | Purpose |
|---------|--------|---------|------------|---------|
| Constrained | _(none)_ | Trainable | Gaussian from Hessian cov | Default: proper UQ |
| Free | `_free` | Trainable | None | Check if constraint is too tight |
| Frozen | `_frozen` | Fixed at w_hat | N/A | Check perturbation-only sensitivity |

### Diagnostic plots per variant

Each variant produces (per dimension):
- `gof_marginals_x{dim}` — binned ratios to test data with Poisson + epistemic errors
- `gof_vs_true_x{dim}` — continuous ratios to analytic true density
- `gof_vs_hiker_x{dim}` — continuous model ratios to HIKER, binned data/HIKER ratio

Plus: `gof_loss_curves`, `gof_weight_pulls`, `gof_coeffs_evolution`

### Calibration

The test statistic is calibrated empirically by running `N_TOYS` pseudo-experiments:
each toy samples fresh data from the fitted HIKER (auto-choosing direct mixture
vs hit-or-miss sampling based on weight signs), runs the full LRT, and collects `t_toy`.
The empirical p-value is the fraction of toys with `t_toy >= t_obs`.

## Output structure

```
data/<benchmark>_Ntrain<N>_Ntest<N>_seed<S>/
    data_train.npy, data_test.npy, data_config.json

runs/<config_name>/
    pipeline_config.json
    train_config.json, data_config.json
    model.pt, covariance.npy, hessian.npy
    ensemble_info.npz               -- (ensemble only) per-seed norm kernels + boundaries
    marginals.npz
    training_loss.png
    training_coeffs.png, training_coeffs_normalized.png
    training_centroids.png
    eigenspectrum.png
    marginal_x0.png, marginal_x1.png
    covariance_matrix.png
    coverage_results.json, coverage_pulls.png
    gof_<label>/
        gof_config.json
        gof_results.json, gof_results_free.json, gof_results_frozen.json
        gof_model_den.pt, gof_model_num.pt, ...  (weights only, ~20KB each)
        gof_loss_hist.npz           -- loss + weight + perturbation coefficient histories
        gof_loss_curves.png, gof_weight_pulls.png, gof_coeffs_evolution.png
        gof_marginals_x0.png, gof_vs_true_x0.png, gof_vs_hiker_x0.png, ...
        gof_calibration.png
    logs/                           -- SLURM job logs and batch scripts
    seed0/, seed1/, ...             -- (ensemble only) per-seed model outputs
```
