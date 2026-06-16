# Uncertainty-Aware Density Estimation and GoF with Normalizing Flows, Sparker Kernels, and Learned Classifier Basis

End-to-end pipeline for:
1) distributional modeling with Normalizing Flows,
2) frequentist uncertainty with $w_i f_i$ ensembles
   ([Benevedes & Thaler, 2025](https://arxiv.org/abs/2506.00113)),
3) coverage validation on observables,
4) learned likelihood–ratio hypothesis tests with calibration.

---

## Table of contents

1. Overview
2. Repository layout
3. Environment
4. Generate Data
5. Step-by-step pipeline 
   1. Step 1 — Train NF ensemble
   2. Step 2 — Fit $w$ and propagate uncertainty
   3. Step 3 — Coverage test
   4. Step 4 — Learned likelihood–ratio tests (one-sample GoF and two-sample)

---

## 1. Overview

The pipeline has four components.

1. **Density estimation with Normalizing Flows**  
   Ensemble of normalizing flows trained on bootstrap replicas and independent initializations to capture data and optimization variability.

2. **Frequentist UQ with $w_i f_i$ ensembles**  
   Weighted mixture $\hat f(x)=\sum_i \hat w_i f_i(x)$. Weights come from penalized MLE with a normalization constraint. Covariance of $\hat w$ via a sandwich estimator, propagated to a pointwise band $\hat f \pm \sigma_{\hat f}$.

3. **Coverage validation**  
   Check if a known observable (here, the first moment) falls within the predicted uncertainty band at the chosen confidence level.

4. **Learned likelihood–ratio tests**  
   **One-sample GoF:** uncertainty on $\hat w$ is propagated inside the test; null $H_\phi$ uses the ensemble with $w\sim\mathcal N(\hat w,\mathrm{Cov}(\hat w))$, the alternative adds a small Gaussian–mixture correction.  
   **Two-sample (NPLM):** both REF and DATA enter as samples; the ratio is learned and calibrated with toys or permutations.

---

## 2. Repository layout

```text
MonoJet_NPLM_analysis/
├─ Train_Ensembles/                       # Orchestration for NF ensembles (arrays, logs)
│  ├─ Generate_Data/                      # Prepare training splits and ensemble datasets
│  └─ Train_Models/                       # Train individual ensemble members
│     ├─ Normalizing_Flows/               # NF training backends
│     │  ├─ nflows/                       # nflows-based NF training
│     │  └─ zuko/                         # zuko-based (Bayesian) flow training
│     └─ Sparker_kernels/                 # Kernel-based density estimation
├─ Uncertainty_Modeling/                  # Weight fitting (w_i), sandwich covariance, propagation
│  └─ wifi/                               # w_i f_i frequentist ensembles and covariance
│     ├─ Coverage_Check/                  # Coverage studies on observables — not maintained/up-to-date
│     └─ Fit_Weights/                     # Penalized MLE for w, sandwich covariance, propagation
│        ├─ fit_ensemble_weights.py       # Unified weight-fitting script (MODEL_TYPE toggle)
│        └─ submit_fit_weights.sh         # SLURM submission (set MODEL_TYPE=nf|kernels)
├─ Generate_Ensemble_Samples/             # Accept–reject sampling for LRT and NPLM reference draws
│  ├─ Normalizing_Flows/                  # Hit-or-miss MC for NF ensemble
│  └─ Sparker_kernels/                    # Hit-or-miss MC for kernel ensemble
├─ LRT/                                   # One-sample learned LRT with uncertainty-aware reference
│  ├─ LRT.py                              # Unified LRT script (--model_type kernels|nf)
│  └─ submit_LRT_toys.sh                  # SLURM submission (set MODEL_TYPE=kernels|nf)
├─ NPLM/                                  # Two-sample LRT (NPLM)
│  ├─ NPLM-embedding/                     # Scripts and utilities for two-sample tests
│  ├─ NPLM_NF_ensemble/                   # Results: ensemble as reference
│  └─ NPLM_NF_one_model/                  # Results: single NF as reference
├─ shared/                                # Shared library used across pipeline stages
│  └─ Sparker_utils/                      # Kernel utilities (SPARKutils, LRTGOFutils, ENSEMBLEutils, …)
├─ wifi_better_basis/                     # Classifier-based GoF (Sean's approach, separate pipeline)
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
- Steps 1–3 (NF training, weight fitting, coverage):
```bash
conda activate nf_env
```
- Steps 1–3 (kernel training, weight fitting):
```bash
conda activate kernels_env
```
- Learned likelihood–ratio tests (LRT, NPLM):
```bash
conda activate nplm_env   # for NF backend
conda activate kernels_env  # for kernel backend
```

---

## 4. Generate Data

This step creates the **2D target distribution** used across the pipeline.

The script also **saves the analytic first moment** of the target, which is required later for the coverage test.

**Where:** `Train_Ensembles/Generate_Data/`  
**Main script:** e.g. `generate_2d_gaussian_heavy_tail_target_data.py`  
**Plotting notebook:** e.g. `plot_2d_gaussian_heavy_tail_target.ipynb`

**Event counts**
- For training the NF ensemble (“statistical power”): **100,000** target events
- For the statistical test (GoF calibration, toys): **500,000** target events

**Run**
```bash
cd Train_Ensembles/Generate_Data
python generate_2d_gaussian_heavy_tail_target_data.py
```

This will create outputs under something like:
```text
Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/
├── 100k_2d_gaussian_heavy_tail_target_set.npy
├── 500k_2d_gaussian_heavy_tail_target_set.npy
└── <file with analytic first moment used for coverage>
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
Activate the environment for Steps 1–3, move into your backend, and submit:
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

### 5.3 Step 3 — Coverage test

> **Note:** The scripts in `Coverage_Check/` are not maintained and may not reflect the current pipeline. Use as a reference only.

This step checks whether the propagated uncertainty from the fitted ensemble
adequately covers a known observable (here, the **first moment**) at a chosen
confidence level. At a high level, we run **multiple pseudo-experiments**. For each experiment:
1. Generate a fresh target dataset inside the script (e.g., **200,000** events) to
   achieve a **2× oversampling** with respect to the NF ensemble’s statistical
   power (**100,000** events).
2. Recompute the first moment estimate $\hat{\mu}$ from the fitted ensemble.
3. Compare against the analytic truth $\mu^\star$ using
   $$|\hat{\mu}-\mu^\star| < z_{1-\alpha/2}\,\sigma_{\hat{\mu}},$$
   where $\sigma_{\hat{\mu}}$ is the propagated uncertainty from the weight
   covariance (sandwich estimator).  
   We repeat this **300 times** and report pass/fail per feature and the overall
   coverage rate.

#### Prerequisites

You need the following artifacts from previous steps:
- **Step 5.1:** `f_i.pth` (the collected ensemble)
- **Step 5.2:** fitted weights and their covariance (saved in your `results_fit_weights_*` dir)
- **Step 4:** analytic first moment file $\mu^\star$
- **(New)** a file containing **sampled ensemble means** for each experiment (generated in A below)

#### A) Generate sampled ensemble means

##### Where
```text
MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/generate_sampled_means/
```

##### Scripts
- `generate_sampled_means.py`
- `submit_generate_sampled_means.sh`

##### Configure
In submit_generate_sampled_means.sh edit: 
- `TRIAL_DIR`: the directory where the Step 5.2 weight-fitting results are stored
- `ARCH_CONFIG_DIR`: the directory where the Step 5.1 ensemble models (and architecture config) are stored

##### Run
```bash
cd Uncertainty_Modeling/wifi/Coverage_Check/generate_sampled_means
sbatch submit_generate_sampled_means.sh
```

##### Outputs
```text
Uncertainty_Modeling/wifi/Coverage_Check/generate_sampled_means/results_generated_sampled_means/
└── <means_file.npy>   # first-moment estimates from the ensemble
```
You will pass the path to <means_file.npy> into the coverage step as MU_I_FILE.
To verify that the sampled means were generated correctly, use `check_sampled_means.ipynb` and confirm:
- Array shape is (n_models, n_features)
- Feature-wise means and standard deviations look reasonable
- Feature-wise minimum and maximum values are within expected ranges.
- No `NaNs` or `infs` values are present 

#### B) Run the coverage check

##### Where
```text
MonoJet_NPLM_analysis/Uncertainty_Modeling/wifi/Coverage_Check/
```

##### Scripts
- e.g. `coverage_check_2d.py` (for a 2d target distribution)
- `submit_coverage_check.sh`

##### Configure
Open `submit_coverage_check.sh` and set:
- `TRIAL_DIR` — directory that contains your `f_i.pth` from Step 5.1
- `N_SAMPLED`: number of target events to sample inside the coverage script (e.g., 200000 for a 2× oversampling relative to the 100000 used to train the NFs)
- `MU_TARGET_PATH`: path to the file saved in Step 4 with the analytic first moment
- `MU_I_FILE`: path to the <means_file.npy> produced in step A (this must point to .../Coverage_Check/generate_sampled_means/results_generated_sampled_means/<means_file.npy>)

##### Run
```bash
cd Uncertainty_Modeling/wifi/Coverage_Check
sbatch submit_coverage_check.sh
```

##### Outputs 
Results are written under:
```bash
Uncertainty_Modeling/wifi/Coverage_Check/coverage_outputs/
```
Typical contents include:
- Per-experiment pass/fail (per feature) indicating whether
$|\hat{\mu}-\mu^\star| < z_{1-\alpha/2} \sigma_{\hat{\mu}}$ held.
- Aggregate coverage summary (e.g., error bars, saved $\hat{\mu}$ vectors, propagated uncertainties,...).

##### Final diagnostic 
After the coverage check completes, open `Uncertainty_Modeling/wifi/Coverage_Check/coverage_calc.ipynb` and run all cells. The notebook reads from coverage_outputs/ and reports per feature and overall coverage, adds binomial confidence intervals, and produces quick sanity-check plots in the `coverage_plots/` folder. Use it to verify that the empirical coverage matches the target level within statistical uncertainty.

---

### 5.4 Step 4 — Learned-likelihood ratio tests 

This step provides two complementary hypothesis tests based on a learned likelihood ratio. In both cases we report a Z score that quantifies how compatible the generated sample is with the target distribution.

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

#### What we use here
1) One sample GoF learned LRT: 
Propagates the uncertainty on $\hat{w}$ directly into the GoF test, then compares the ensemble with the target distribution.
Use the fitted weights $\hat w$ and their covariance $\mathrm{Cov}(\hat w)$ from Step 5.2.
2) Two sample learned LRT (e.g. NPLM):
Compares the ensemble (or a single NF) with the target distribution when both REF and DATA are provided as samples.

### 1) One sample GoF learned LRT

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

**Per-toy numerator kernel diagnostics** (optional, secondary):
```bash
bash LRT/run_plot_lrt_num_kernels.sh
```
Produces per-seed 2D scatter of kernel centers and 1D marginal overlays. Edit paths in the script for your run tag and model type before running.

### 2) Two sample learned LRT (NPLM)

#### Idea 
- Both REF and DATA enter as samples.
- We first generate REF samples from the ensemble using hit-or-miss Monte Carlo.
- We then run the learned LRT where a function $t_{\theta}(x)$ is trained to approximate the log density ratio (e.g., $t_{\theta}(x)\approx \log \tfrac{p_{\text{DATA}}(x)}{p_{\text{REF}}(x)}$).
A sample-wise statistic such as
$S = \sum_{n} t_{\theta}(x_n)$
is aggregated, calibrated with toys or permutations, and converted into a $Z$ score.

#### A) Generate REF samples by hit-or-miss MC

##### Scripts 
Two backends are available under `Generate_Ensemble_Samples/`:

**Normalizing Flows backend** — `Generate_Ensemble_Samples/Normalizing_Flows/`
- Python script: `generate_hit_or_miss_NFs.py`
- Submission script: `submit_generate_hit_or_miss_NFs.sh`

**Sparker kernels backend** — `Generate_Ensemble_Samples/Sparker_kernels/`
- Python script: `generate_hit_or_miss_Sparker.py`
- Submission script: `submit_generate_hit_or_miss_Sparker.sh`

##### Configure 
All paths are passed as arguments — edit the submission script only, not the python script:

- **NF backend** (`submit_generate_hit_or_miss_NFs.sh`): set `TRIAL_DIR` (NF ensemble dir with `f_i.pth`), `W_PATH` (fitted weights), `OUT_DIR`, `NGENERATE` (events per task), `TAIL_BOUND`.
- **Kernel backend** (`submit_generate_hit_or_miss_Sparker.sh`): set `ENSEMBLE_DIR`, `W_PATH`, `OUT_DIR`, `NGENERATE`, `BOUNDS` (hit-or-miss box, e.g. `-1.5 0.5 0.5 4.5`).
- Set the SLURM array size, e.g. `#SBATCH --array=0-199`. With `NGENERATE=5000` and 200 tasks this yields 1M proposals. As a rule of thumb, generate about twice the number of accepted events you will need for the test.
- Output files are saved automatically as `seed{i}.npy` in a subfolder named after the wifi weights run.

##### Run 
```bash
# Normalizing Flows backend
cd Generate_Ensemble_Samples/Normalizing_Flows
sbatch submit_generate_hit_or_miss_NFs.sh

# Sparker kernels backend
cd Generate_Ensemble_Samples/Sparker_kernels
sbatch submit_generate_hit_or_miss_Sparker.sh
```

##### Sanity checks 
- Inspect a single file with `check_hit_or_miss.ipynb` (NF) or `check_hit_or_miss_Sparker.ipynb` (kernels).
- Merge many shards into one array with `concatenate.ipynb` (NF backend).

#### B) Run two sample test

##### Scripts
In `NPLM/NPLM-embedding/` one can find: 
- python scripts: 
   - `toy_ensemble.py`: REF is the ensemble (samples generated in A))
   - `toy.py`: REF is a single NF model
- submission scripts: `submit_toy_ensemble.sh`, `submit_toy.sh`
- utils scripts: `FLKutils.py` (ensemble), `FLKutils_model.py` (single NF), `SampleUtils.py`

##### Configure
In `toy_ensemble.py` or `toy.py`, set:
- `data_path`: DATA samples (target)
- `reference_path`: REF samples from hit-or-miss (step A)

In `submit_toy_ensemble.sh` or `submit_toy.sh`, set args:
- -d <int>: number of DATA events, e.g. 100000
- -r <int>: number of REF events, e.g. 500000
- -t <int>: number of toys, e.g. 100
- -M <int>: $M ≈ \sqrt(d+r)$
- -c {True,False}: CALIBRATION=True or False
Example: 
```text
-d 100000 -r 500000 -t 100 -M 2400 -c ${CALIBRATION}
```
Note that for each (d,r) pair, run both CALIBRATION=True  and CALIBRATION=False. To study p-value vs sample size, sweep multiple (d,r) pairs (update `M` accordingly). 

##### Run 
```bash 
cd NPLM/NPLM-embedding
# Ensemble as REF
sbatch submit_toy_ensemble.sh
# Single NF as REF
sbatch submit_toy.sh
```

#### Outputs 
`NPLM/NPLM_NF_ensemble/` and `NPLM/NPLM_NF_one_model/`, each containing:
   - `calibration/` (runs with -c True)
   - `comparison/` (runs with -c False)
   For every (d,r,c) combination:
      - `plots/`:  per-feature REF/DATA ratio plots and quick diagnostics
      - `.h5`: test-statistic payload used to derive p-values and Z

To produce the probability vs test-statistic curves and summary tables, run:
```text 
NPLM/NPLM-embedding/analyse_output.ipynb
```

#### Interpreting Z score
- Z near zero indicates that REF and DATA are statistically compatible at the chosen level.
- Large Z indicates a discrepancy.


