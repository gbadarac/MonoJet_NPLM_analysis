# Uncertainty-Aware Density Estimation and GoF with Normalizing Flows

End-to-end pipeline for
1) distributional modeling with Normalizing Flows,
2) frequentist uncertainty with \(w_i f_i\) ensembles
   ([Benevedes & Thaler, 2025](https://arxiv.org/abs/2506.00113)),
3) coverage validation on observables,
4) one-sample learned likelihood–ratio GoF with calibration.

---

## Table of contents

1. Overview
2. Repository layout
3. Environment
4. Generate Data
5. Step-by-step usage
   1. Step 1 — Train NF ensemble
   2. Step 2 — Fit \(w\) and propagate uncertainty
   3. Step 3 — Coverage test
   4. Step 4 — One-sample GoF (learned LRT)

---

## 1. Overview

The pipeline has four components.

1. **Density estimation with Normalizing Flows**  
   Ensemble of normalizing flows trained on bootstrap replicas and independent initializations to capture data and optimization variability.

2. **Frequentist UQ with \(w_i f_i\) ensembles**  
   Weighted mixture \(\hat f(x)=\sum_i \hat w_i f_i(x)\). Weights come from penalized MLE with a normalization constraint. Covariance of \(\hat w\) via a sandwich estimator, propagated to a pointwise band \(\hat f \pm \sigma_{\hat f}\).

3. **Coverage validation**  
   Check if a known observable (here, the first moment) falls within the predicted uncertainty band at the chosen confidence level.

4. **One-sample learned LRT GoF**  
   Null \(H_\phi\): ensemble with Gaussian prior \(w\sim\mathcal N(\hat w,\mathrm{Cov}(\hat w))\).  
   Alternative \(H_{\phi,a}\): ensemble plus a Gaussian-mixture expansion.  
   Calibrate the test statistic with pseudo-experiments from \(\hat f\).

---

## 2. Repository layout

```text
MonoJet_NPLM_analysis/
├─ Generate_Gaussian_Toy/                 # 2D Gaussian toy (controlled cross-check dataset)
├─ Generate_Ensemble_Data_Hit_or_Miss_MC/ # Accept–reject sampling for toys/reference draws
├─ Normalizing_Flows/                     # Flow components, transforms, model builders
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
├─ Grid_Search/                           # Architecture experiments
├─ envs/                                  # Conda/pip environment files
└─ README.md

---
## 3. Environment

Create the environments with:
```bash
conda env create -f envs/nf_env.yml
conda env create -f envs/nplm_env.yml
```
Activate the right one for each step:
# Steps 1, 2, 3: training, weight fitting, coverage
conda activate nf_env

# Learned likelihood–ratio test
conda activate nplm_env

---

## 4. Generate Data

This step creates the **2D target distribution** used across the pipeline: a mixture where
- **Feature 1** is a bi-modal Gaussian mixture, and  
- **Feature 2** is skewed with heavy tails.

The script also **saves the analytic first moment** of the target, which is required later for the coverage test.

**Where:** `Train_Ensembles/Generate_Data/`  
**Main script:** `generate_2d_gaussian_heavy_tail_target_data.py`  
**Plotting notebook:** `plot_2d_gaussian_heavy_tail_target.ipynb`

**Event counts**
- For training the NF ensemble (“statistical power”): **100,000** target events
- For the statistical test (GoF calibration, toys): **500,000** target events

**Run**
```bash
cd Train_Ensembles/Generate_Data
python generate_2d_gaussian_heavy_tail_target_data.py
```

This will create outputs under something like:
Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim/
├── 100k_2d_gaussian_heavy_tail_target_set.npy
├── 500k_2d_gaussian_heavy_tail_target_set.npy
└── <file with analytic first moment used for coverage>

---