# Uncertainty-Aware Density Estimation and GoF with Normalizing Flows and Sparker Kernels

End-to-end pipeline for:
1) distributional modeling with Normalizing Flows,
2) frequentist uncertainty with $w_i f_i$ ensembles
   ([Benevedes & Thaler, 2025](https://arxiv.org/abs/2506.00113)),
3) a learned one-sample goodness-of-fit likelihood–ratio test with calibration.

---

## Table of contents

1. Overview
2. Repository layout
3. Environment
4. Generate Data
5. Step-by-step pipeline 
   1. Step 1 — Train NF ensemble
   2. Step 2 — Fit $w$ and propagate uncertainty
   3. Step 3 — Learned one-sample GoF likelihood–ratio test

---

## 1. Overview

The pipeline has three components.

1. **Density estimation with Normalizing Flows**  
   Ensemble of normalizing flows trained on bootstrap replicas and independent initializations to capture data and optimization variability.

2. **Frequentist UQ with $w_i f_i$ ensembles**  
   Weighted mixture $\hat f(x)=\sum_i \hat w_i f_i(x)$. Weights come from penalized MLE with a normalization constraint. Covariance of $\hat w$ via a sandwich estimator, propagated to a pointwise band $\hat f \pm \sigma_{\hat f}$.

3. **Learned one-sample GoF likelihood–ratio test**  
   Uncertainty on $\hat w$ is propagated inside the test; null $H_\phi$ uses the ensemble with $w\sim\mathcal N(\hat w,\mathrm{Cov}(\hat w))$, the alternative adds a small Gaussian–mixture correction.

---

## 2. Repository layout

```text
MonoJet_NPLM_analysis/
├─ data/                                  # Shared inputs; self-reproducing generators
│  ├─ 2d_gmm_toymodel/                    # 2D GMM+skew "heavy tail" target + generator
│  └─ 4d_embeddings/                      # 4D embedding datasets (JetClass)
├─ Train_Ensembles/                       # Orchestration for NF ensembles (arrays, logs)
│  └─ Train_Models/                       # Train individual ensemble members
│     ├─ Normalizing_Flows/               # NF training backends
│     │  ├─ nflows/                       # nflows-based NF training
│     │  └─ zuko/                         # zuko-based (Bayesian) flow training
│     └─ Sparker_kernels/                 # Kernel-based density estimation
├─ Uncertainty_Modeling/                  # Weight fitting (w_i), sandwich covariance, propagation
│  └─ wifi/                               # w_i f_i frequentist ensembles and covariance
│     └─ Fit_Weights/                     # Penalized MLE for w, sandwich covariance, propagation
│        ├─ fit_ensemble_weights.py       # Unified weight-fitting script (MODEL_TYPE toggle)
│        └─ submit_fit_weights.sh         # SLURM submission (set MODEL_TYPE=nf|kernels)
├─ LRT/                                   # One-sample learned LRT with uncertainty-aware reference
│  ├─ LRT.py                              # Unified LRT script (--model_type kernels|nf)
│  └─ submit_LRT_toys.sh                  # SLURM submission (set MODEL_TYPE=kernels|nf)
├─ shared/                                # Shared library used across pipeline stages
│  └─ Sparker_utils/                      # Kernel utilities (SPARKutils, LRTGOFutils, ENSEMBLEutils, …)
├─ envs/                                  # Conda/pip environment files
├─ NOTES.md                               # Design decisions and experimental observations
└─ README.md
```

---

## 3. Environment

Create the environments:
```bash
conda env create -f envs/nf_env.yml
conda env create -f envs/nplm_env.yml
conda env create -f envs/kernels_env.yml
```
Activate the right one for each step:
- Steps 1–2 (NF training, weight fitting):
```bash
conda activate nf_env
```
- Steps 1–2 (kernel training, weight fitting):
```bash
conda activate kernels_env
```
- Learned likelihood–ratio test (LRT):
```bash
conda activate nplm_env   # for NF backend
conda activate kernels_env  # for kernel backend
```

---

## 4. Generate Data

This step creates the **2D target distribution** used across the pipeline. It lives
alongside the data it produces, at seed 42.

**Where:** `data/2d_gmm_toymodel/`  
**Main script:** `generate_2d_gaussian_heavy_tail_target_data.py`  
**Plotting notebook:** `plot_2d_gaussian_heavy_tail_target.ipynb`

**Split convention**
- `data_train` — sample the density model / kernel ensemble is fit on
- `data_test`  — disjoint draw (seed + 1) the GoF/LRT is run against
- default **100,000 / 100,000** train/test at seed 42

**Run**
```bash
cd data/2d_gmm_toymodel
python generate_2d_gaussian_heavy_tail_target_data.py
```

This creates a cache dir alongside the script:
```text
data/2d_gmm_toymodel/2d_gmm_skew_Ntrain100000_Ntest100000_seed42/
├── data_train.npy
├── data_test.npy
└── data_config.json
```

---

## 5. Step-by-step pipeline

---

### 5.1 Step 1 — Train NF ensemble

You can train with two backends:

- `nflows/`
- `zuko/` — same interface as `nflows`, plus **Bayesian Flows** support (set `bayesian=True` in the launcher arguments).

Each NF backend folder contains:
- Training script: `EstimationNFnflows.py` (nflows) or `EstimationNFzuko.py` (zuko)
- SLURM submission script: `submit_array_NFnflows.sh` / `submit_array_NFzuko.sh`
- Launcher script: `run_submit_array_NFnflows.sh` / `run_submit_array_NFzuko.sh`

NF shared helpers (model building, sampling, plotting):
- `Train_Ensembles/Train_Models/utils_flows.py`

Kernel shared utilities live in `shared/Sparker_utils/` (used across training, weight fitting, hit-or-miss, and LRT).

#### Configure

1) **How many models to train (ensemble size)**  
Edit the SLURM array in the submission script:
```bash
#SBATCH --array=0-59   # trains 60 models: indices [0, 59]
```

2) **Architecture, data paths, output, seeds**  
Edit the launcher script to set:
- architecture: layers, blocks, hidden features, bins
- data paths: where Step 4 saved the target dataset
- ensemble size and seeds
- output directory for logs and checkpoints

3) **Bayesian Flows (zuko only)**  
Enable in the launcher arguments (use the exact flag name you defined), e.g.:
```bash
bayesian=True
```

#### Run
Activate the environment for Steps 1–2, move into your backend, and submit:
```bash
cd Train_Ensembles/Train_Models/Normalizing_Flows/nflows
sbatch run_submit_array_NFnflows.sh
```
Replace `nflows` with `zuko` if you use the zuko backend.

#### Outputs
After jobs finish you will find:
```text
Train_Ensembles/Train_Models/Normalizing_Flows/nflows/EstimationNFnflows_outputs/
├── <one folder per trained member>   # checkpoints 
└── f_i.pth                           # collects all members for downstream steps
```
If `f_i.pth` is missing but all single-model .pth files exist, gather them with:
```bash
Train_Ensembles/Train_Models/Normalizing_Flows/collect_all_models_into_ensemble.ipynb
```

#### Quick visual check
Plot model vs target marginals with the notebook:
```bash
Train_Ensembles/Train_Models/Normalizing_Flows/test_NFs_marginals.ipynb
```

#### Typical settings
- architecture: layers 4, blocks 16, hidden 128, bins 15
- batch size 512, learning rate 5e-6, early stopping patience 10
- ensemble size: 60 models

---

### 5.2 Step 2 — Fit $w$ and propagate uncertainty

Form the weighted mixture $\hat f(x)=\sum_i w_i f_i(x)$ by fitting the ensemble
weights $w$ with a penalized MLE under the normalization constraint. The step also
computes the weight covariance (sandwich estimator) and propagates it
to a pointwise predictive band $\hat f \pm \sigma_{\hat f}$.

#### Where
```bash 
MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Fit_Weights/
```
#### Python script
- `fit_ensemble_weights.py` — unified script for both NF and kernel ensembles

#### Utilities
- `Uncertainty_Modeling/wifi/utils_NF_wifi.py` — NF ensemble helpers (density eval, marginal plots, covariance propagation)
- `Uncertainty_Modeling/wifi/utils_kernel_wifi.py` — kernel ensemble helpers (same interface)

#### Submission script
- `submit_fit_weights.sh` — set `MODEL_TYPE=nf|kernels` and `NDIM` at the top

#### Outputs 
- `results_fit_weights_NF/`
- `results_fit_weights_kernel/`
Each output folder contains the fitted weights `w_i_fitted.npy`, covariance matrix `cov_w.npy`, logs, and diagnostics.

#### Configure

1. Open `submit_fit_weights.sh` and set:
   - `MODEL_TYPE`: `nf` or `kernels`
   - `NDIM`: number of dimensions (e.g. `2` or `4`)
   - For `MODEL_TYPE=nf`, optionally set `DATASET` (e.g. `2d_gaussian`)
   - Paths (`TRIAL_DIR`, `DATA_PATH`) are set automatically from the toggles above
2. Submit
```bash 
cd Uncertainty_Modeling/wifi/Fit_Weights
sbatch submit_fit_weights.sh
```

The fitted weights `w_i_fitted.npy` are saved directly in the output folder and are ready for downstream steps. If optimization fails on some runs, check logs for non-finite losses and retry.

---

### 5.3 Step 3 — Learned one-sample GoF likelihood–ratio test

This step provides a one-sample goodness-of-fit hypothesis test based on a learned likelihood ratio. We report a Z score that quantifies how compatible the generated sample is with the target distribution.

#### Common setup 
- REF = density model, for example the fitted ensemble from Step 5.2, or a single NF.
- DATA = target distribution sample

#### Test statistic, high level 
Let $p_{\mathrm{REF}}$ be the reference density and $H_{\boldsymbol{\phi}}$ a flexible alternative that reduces to
$p_{\mathrm{REF}}$ when $\boldsymbol{\phi}=0$.
We learn the alternative from DATA and form a likelihood–ratio statistic
$ T = -2\log\lambda = 2\big[\ell(\hat{\boldsymbol{\phi}}) - \ell(\boldsymbol{\phi}{=}0)\big]$, 
where $\ell$ is the log likelihood on the DATA sample.
We calibrate the null distribution of $T$ with pseudo-experiments drawn from the reference,
convert the resulting $p$-value into a $Z$ score, and use $Z$ to summarize compatibility.

#### Idea 
- The reference enters in analytical form as an evaluable pdf $p_{\mathrm{REF}}(x)$, obtained from the ensemble with the $\hat{\boldsymbol w}$ uncertainty integrated as in Step 5.2.
- The alternative adds a small, learnable correction $f(x,\boldsymbol{\phi})$ that integrates to zero so the result remains a valid density:
$p(x \mid H_{\boldsymbol{\phi}}) = p_{\mathrm{REF}}(x) + f(x,\boldsymbol{\phi}),
\qquad \int f(x,\boldsymbol{\phi})dx = 0$.
- We fit $\boldsymbol{\phi}$ on DATA, build $T$, calibrate $T$ with toys drawn from $p_{\mathrm{REF}}$, then report $Z$.

Both backends (NF and kernels) are handled by a single unified script under `LRT/`:

#### Scripts
- Python script: `LRT/LRT.py` — handles both backends via `--model_type kernels|nf`
- Submission script: `LRT/submit_LRT_toys.sh` — set `MODEL_TYPE` and `CALIBRATION` at the top

#### Configure
Open `submit_LRT_toys.sh` and set:
- `MODEL_TYPE`: `kernels` or `nf`
- `CALIBRATION`: `1` = null toys (SIR + calibration pool), `0` = observed (target data)
- `FIX_WIFI_WEIGHTS` / `FREE_WIFI_WEIGHTS`: optional flags to fix or free the wifi weights during the LRT
- All paths (`W_PATH`, `W_COV_PATH`, `CALIB_DATA`, `TARGET_DATA`, and model-specific paths) are set automatically from the `MODEL_TYPE` toggle
- SLURM array for number of toys, e.g. `#SBATCH --array=0-99` for 100 toys

#### Run
```bash
cd LRT
# Edit MODEL_TYPE and CALIBRATION at the top of submit_LRT_toys.sh, then:
sbatch submit_LRT_toys.sh
```
Run twice: once with `CALIBRATION=1` (null toys) and once with `CALIBRATION=0` (observed).

#### Outputs
Results are written under `LRT/results/<run_tag>/`.
- `calibration/` (`CALIBRATION=1`) and `test/` (`CALIBRATION=0`) subfolders
- Inside each, one folder per toy `seed{N}/` containing:
  - `seed{N}_T.npy` — test-statistic value
  - `seed{N}_coeffs.npy`, `seed{N}_kernel_centers.npy` — numerator Gaussian kernel correction
  - `seed{N}_den_weights.npy`, `seed{N}_num_weights.npy`, `seed{N}_init_weights.npy` — WiFi weight diagnostics

#### Analyse results (no SLURM needed — runs locally in a few seconds)

**Main results** — T distribution, Z score, p-value, weight diagnostics:
```bash
conda activate kernels_env   # use kernels_env for both NF and kernel results — nf_env has a scipy/numpy conflict
python LRT/analyse_LRT_output.py \
    --results_dir LRT/results/<run_tag> \
    --dof 100 \
    --clip_tau 0.0005 \
    [--w_cov_path <path/to/cov_w.npy>]   # optional: adds normalised weight-pull plot
```
Produces `T_distribution.pdf/png`, `weight_shifts.pdf/png`, `chi2_quantile_table.txt` (and optionally `weight_pulls.pdf/png`) under `LRT/results/<run_tag>/plots/`.
