# Uncertainty-Aware Density Estimation and GoF with Normalizing Flows

End-to-end pipeline for
1) distributional modeling with Normalizing Flows,
2) frequentist uncertainty with $w_i f_i$ ensembles
   ([Benevedes & Thaler, 2025](https://arxiv.org/abs/2506.00113)),
3) coverage validation on observables,
4) one-sample learned likelihood–ratio GoF with calibration.

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
   4. Step 4 — One-sample GoF (learned LRT)

---

## 1. Overview

The pipeline has four components.

1. **Density estimation with Normalizing Flows**  
   Ensemble of normalizing flows trained on bootstrap replicas and independent initializations to capture data and optimization variability.

2. **Frequentist UQ with $w_i f_i$ ensembles**  
   Weighted mixture $\hat f(x)=\sum_i \hat w_i f_i(x)$. Weights come from penalized MLE with a normalization constraint. Covariance of $\hat w$ via a sandwich estimator, propagated to a pointwise band $\hat f \pm \sigma_{\hat f}$.

3. **Coverage validation**  
   Check if a known observable (here, the first moment) falls within the predicted uncertainty band at the chosen confidence level.

4. **One-sample learned LRT GoF**  
   Null $H_\phi$: ensemble with Gaussian prior $w\sim\mathcal N(\hat w,\mathrm{Cov}(\hat w))$.  
   Alternative $H_{\phi,a}$: ensemble plus a Gaussian–mixture expansion.  
   Calibrate the test statistic with pseudo-experiments from $\hat f$.

---

## 2. Repository layout

```text
MonoJet_NPLM_analysis/
├─ Generate_Gaussian_Controlled_Toy/      # 2D Gaussian toy (controlled cross-check dataset)
├─ Generate_Ensemble_Data_Hit_or_Miss_MC/ # Accept–reject sampling for toys/reference draws
├─ Normalizing_Flows/                     # Experiments with NFs
├─ Train_Ensembles/                       # Orchestration for NF ensembles (arrays, logs)
│  ├─ Generate_Data/                      # Prepare training splits and ensemble datasets
│  └─ Train_Models/                       # Train individual ensemble members
│     ├─ nflows/                          # nflows-based NF training
│     └─ zuko/                            # zuko-based (Bayesian) flow training
├─ Uncertainty_Modeling/                  # Weight fitting (w_i), sandwich covariance, propagation
│  ├─ BayesianFlows/                      # Bayesian-flow UQ utilities and experiments
│  └─ wifi/                               # w_i f_i frequentist ensembles and covariance
│     ├─ Coverage_Check/                  # Coverage studies on observables (e.g., first moment)
│     └─ Fit_Weights/                     # Penalized MLE for w, sandwich covariance, propagation
├─ LRT_with_unc/                          # One-sample learned LRT with uncertainty-aware reference
├─ NPLM/                                  # Two-sample LRT helpers (if needed)
├─ Grid_Search/                           # NF architecture experiments
├─ envs/                                  # Conda/pip environment files
└─ README.md
```

---

## 3. Environment

Create the environments:
```bash
conda env create -f envs/nf_env.yml
conda env create -f envs/nplm_env.yml
```
Activate the right one for each step:
- Steps 1, 2, 3: training, weight fitting, coverage
```bash
conda activate nf_env
```
- Learned likelihood–ratio test
```bash
conda activate nplm_env
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

Each backend folder contains:
- Training script, e.g. `EstimationNFnflows.py`
- SLURM submission script, e.g. `submit_array_NFnflows.sh`
- Launcher script, e.g. `run_submit_array_NFnflows.sh`
- `utils_flows.py` with shared helpers for model building, sampling, plotting

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
cd Train_Ensembles/Train_Models/nflows
sbatch run_submit_array_NFnflows.sh
```
Replace nflows with zuko if you use the zuko backend.

#### Outputs
After jobs finish you will find:
```text
Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/
├── <one folder per trained member>   # checkpoints 
└── f_i.pth                           # collects all members for downstream steps
```
If `f_i.pth` is missing but all single-model .pth files exist, then use the notebook below to gather the trained members into a single ensemble file:
```bash
Train_Ensembles/Train_Models/<backend>/collect_all_models_into_ensemble.ipynb
```

#### Quick visual check
Plot model vs target marginals with the notebook:
```bash
Train_Ensembles/Train_Models/nflows/test_NFs_marginals.ipynb
```
Plotting utilities live in `utils_flows.py` in both backends.

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
#### Python scripts
- `fit_NF_ensemble_weights_2d.py`
- `fit_NF_ensemble_weights_4d.py`
- `fit_toy_weights.py`

#### Utilities
`utils_wifi.py` — shared helpers used by all weight-fitting scripts

#### Submission scripts
- `submit_fit_NF_ensemble_weights.sh`
- `submit_fit_toy_weights.sh`

#### Outputs 
- `results_fit_weights_NF/`
- `results_fit_weights_gaussian_toy/`

These folders contain the fitted weights `w_i_fitted.npy`, covariance matrix `cov_w.npy`, logs, and diagnostics. 

#### Configure

1. Edit the submission script
Open `submit_fit_NF_ensemble_weights.sh` and set:
- `trial_dir` — directory that contains your `f_i.pth` from Step 5.1
- `data_path` — path to the 100000 target events file from Step 4

2. Submit
```bash 
cd Uncertainty_Modeling/wifi/Fit_Weights
sbatch submit_fit_NF_ensemble_weights.sh
```

#### Recovering the final weights 

The fitted weights are later used to perform hit-or-miss Monte Carlo over the ensemble for the learned likelihood-ratio GoF test.

You can obtain the final fitted weights in two ways:
- From the output files saved in `results_fit_weights_*`
- With the notebook: `Uncertainty_Modeling/wifi/Fit_Weights/w_i_final_file.ipynb`

Note: If optimization fails on some runs, check logs for non-finite losses and retry.
---

### 5.3 Step 3 — Coverage test

This step checks whether the propagated uncertainty from the fitted ensemble
adequately covers a known observable (here, the **first moment**) at a chosen
confidence level.

At a high level, we run **multiple pseudo-experiments**. For each experiment:
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
- `TRIAL_DIR` — the directory where the Step 5.2 weight-fitting results are stored
- `ARCH_CONFIG_DIR` — the directory where the Step 5.1 ensemble models (and architecture config) are stored

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
- `N_SAMPLED` — number of target events to sample inside the coverage script (e.g., 200000 for a 2× oversampling relative to the 100000 used to train the NFs)
- `MU_TARGET_PATH` — path to the file saved in Step 4 with the analytic first moment
- `MU_I_FILE` — path to the <means_file.npy> produced in step A (this must point to .../Coverage_Check/generate_sampled_means/results_generated_sampled_means/<means_file.npy>)

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

#### Test statistic - high level 
Let $(p_{\mathrm{REF}}$ be the reference density and $H_{\boldsymbol{\phi}}$ a flexible alternative that reduces to
$p_{\mathrm{REF}}$ when $\boldsymbol{\phi}=0$.
We learn the alternative from \text{DATA} and form a likelihood–ratio statistic
$T \;=\; -2\log\lambda \;=\; 2\big[\ell(\hat{\boldsymbol{\phi}}) - \ell(\boldsymbol{\phi}{=}0)\big]$
where $\ell$ is the log-likelihood on the DATA sample.
We calibrate the null distribution of $T$ with pseudo-experiments drawn from the reference,
convert the resulting $p$-value into a $Z$ score, and use $Z$ to summarize compatibility.

#### What we use here

1) One sample GoF learned LRT: 
Used to propagate the uncertainty from $\hat{w}$ directly inside the GoF test. 
Use the fitted weights $\hat{w}$ and their covariance $Cov(\hat{w})$ from Step 5.2.

2) Two sample learned LRT (e.g. NPLM):
Used to compare the ensemble and e.g. a single NF against the target distribution with both sides provided as samples.

#### 1) One sample GoF learned LRT

##### Idea 

##### Scripts 
in `LRT_with_unc/`you can find: 
-
-


##### Run 

##### Outputs 


#### 2) Two sample learned LRT (NPLM)

##### Idea 


##### A) Generate REF samples by hit-or-miss MC

###### Scripts 
in `Generate_Ensemble_Data_Hit_or_Miss_MC/`:
- python script:`generate_hit_or_miss.py`
- submission script: `submit_generate_hit_or_miss.sh`

###### Configure 
- in `generate_hit_or_miss.py`: 
   - `trial_dir`: directory where the fitted weights $\hat{w}$ are stored, example: `Uncertainty_Modeling/wifi/Fit_Weights/results_fit_weights_NF/N_100000_dim_2_seeds_60_4_16_128_15_bimodal_gaussian_heavy_tail`
   - `data_path`: file where the target distribution samples are stored, example: `Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/100k_2d_gaussian_heavy_tail_target_set.npy`
   - arch_config_path: file where the architecture of the NFs is stored, example: `Train_Ensembles/Train_Models/nflows/EstimationNFnflows_outputs/2_dim/2d_bimodal_gaussian_heavy_tail/N_100000_dim_2_seeds_60_4_16_128_15/architecture_config.json`
   - `subdir`: desired output subfolder name
   - `w_i_fitted`: load from the final weights saved in trial_dir (np.load from w_i_fitted.npy)
   - `N_events`: events per array task, for example 5000
   - Output file naming is set here, for example:
   ```bash 
   np.save(os.path.join(out_dir,
    f"ensemble_generated_samples_4_16_128_15_bimodal_gaussian_heavy_tail_seed_{seed}.npy"),
    samples.cpu().numpy())
   ```
   
- in `submit_generate_hit_or_miss.sh`:
   - Set the array size, for example `#SBATCH --array=0-199`. With N_events=5000 this yields one million proposals. As a rule of thumb, generate about twice the number of accepted events you will need for the test.

###### Run 
```bash 
cd Generate_Ensemble_Data_Hit_or_Miss_MC
sbatch submit_generate_hit_or_miss.sh
```

###### Sanity checks 
- Inspect a single file with `check_hit_or_miss.ipynb`.
- Merge many shards into one array with `concatenate.ipynb`.

##### B) Run two sample test

###### Scripts 

###### Configure

###### Utils 

###### Run 

###### Outputs 


##### Interpreting Z-score

- Z near zero indicates that REF and DATA are statistically compatible at the chosen level.
- Large positive or negative Z indicates a discrepancy.


