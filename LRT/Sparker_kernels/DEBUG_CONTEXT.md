We are continuing the LRT kernel overfitting investigation.

Current technical context:

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Analysis:
Let me chronologically analyze the conversation to build a comprehensive technical report.

**Initial Phase: Code Review**
- User asked me to look at LRT.py and LRTGOFutils_v2.py before asking specific questions
- I read both files via an Explore agent

**Problem Statement**
- User showed log file LRT_toys-28638_0.out with T = 151965.847737, which is too large
- User asked for help understanding why

**Key files read:**
1. `/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/LRT.py`
2. `/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_utils/LRTGOFutils_v2.py`
3. `/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/results/logs/LRT_toys-28638_0.out`

**Bug 1: Sign of aux in test statistic**

The loss function in LRTGOFutils_v2.py:
```python
def loss(self, x):
    aux = self.log_auxiliary_term()  # log P(w | prior), negative
    out = -torch.log(p).sum()
    if self.train_weights:
        out = out - aux.sum()  # = -sum log p - log P(w)
    return out
```

This minimizes `-sum log p - log P(w)`, which maximizes `sum log p + log P(w)` (MAP).

But the test statistic in LRT.py used:
```python
T_tensor = (num_log_data.sum() - aux_num) - (den_log_data.sum() - aux_den)
```

This is `(sum log p_num - log P(w_num)) - (sum log p_den - log P(w_den))`, which is INCONSISTENT with the MAP training objective. The correct T should use `+ aux` not `- aux`:
```python
T_tensor = (num_log_data.sum() + aux_num) - (den_log_data.sum() + aux_den)
```

The quantitative effect: T_code = T_correct + 2*(log P(w_den) - log P(w_num)). Since the numerator weights deviate more from the prior, T_code > T_correct by ~37k.

**Bug 2: Asymmetric training epochs**
- `epochs_delta = 20000` for DEN
- `epochs_tau = 200000` for NUM
- At ep 20k: DEN = -1.133e+05, NUM = -1.164e+05 (gap = 3100 nats)
- At ep 200k: NUM = -2.284e+05 (gap inflated to 115k)

**Fixes applied:**
1. Sign fix in LRT.py: `-aux` → `+aux` in T computation and per-event array
2. Sign fix in LRTGOFutils_v2.py: `loglik()` sign
3. epochs_delta: 20000 → 200000

**Results progression:**
- Original T (sign bug, 20k DEN): 151,965
- After sign fix (20k DEN): 115,117
- After sign fix + equal epochs (200k DEN): 23,344

**Key clarification from user:**
- CALIBRATION=0 = 1-sample test: target data vs ensemble (analytic background)
- CALIBRATION=1 = 2-sample test: ensemble vs ensemble (not implemented yet)
- Both "heavy tail gaussian" and "bimodal gaussian" are 2D features of the same target distribution
- DEN should retrain weights on target data (confirmed by user)
- T should be ~χ²(100) ≈ 100 for null hypothesis

**Remaining issue: kernel overfitting**
- T = 23,344 remains because `lambda_L2_numerator = 0`
- 100 narrow kernels (σ = 0.08) placed at data points with no regularization
- Each kernel provides ~233 nats/param improvement vs expected ~0.5 nats/param under H0
- DEN rate at 200k: ~40 nats/epoch → nearly converged
- NUM rate at 200k: ~140 nats/epoch → still learning via kernels
- This is a statistical/hyperparameter problem, not a code bug

**Next steps needed:**
- Set `lambda_L2_numerator > 0` to regularize kernels
- Implement CALIBRATION=1 to empirically calibrate T distribution under H0
- Tune λ until calibration toys give T ~ χ²(100)

Summary:
1. Primary Request and Intent:
   The user asked for help debugging why the LRT (Likelihood Ratio Test) test statistic T was producing anomalously large values (T = 151,965) in their NPLM (New Physics Learning Machine) pipeline. The user also requested a structured technical summary of the full debugging session to preserve context across chats.

2. Key Technical Concepts:
   - **LRT (Likelihood Ratio Test)**: Test statistic T = log L_num − log L_den comparing a flexible numerator model to a constrained denominator model
   - **MAP (Maximum A Posteriori) estimation**: Training minimizes `−sum log p − log P(w | prior)`, equivalently maximizing `sum log p + log P(w | prior)`
   - **TAU model**: PyTorch neural network combining a weighted ensemble of Gaussian kernel models with optional extra Gaussian kernels
   - **WiFi weights**: Pre-fitted weights for combining 60 ensemble members. Both DEN and NUM retrain these weights on target data (profile likelihood ratio test)
   - **Auxiliary term**: `aux = log P(w | prior)` from a 60-dimensional MultivariateNormal prior on ensemble weights
   - **GaussianKernelLayer**: 100 RBF kernels placed at first 100 data points, σ = 0.08, mean-centred coefficients clipped to [−0.1, 0.1]
   - **1-sample test** (CALIBRATION=0): Target data evaluated against ensemble as analytic background. T should follow χ²(100) ≈ 100 under H0 (null hypothesis: data from same distribution as ensemble)
   - **2-sample test** (CALIBRATION=1): Would compare ensemble-sampled data to ensemble. Not yet implemented.
   - **Wilks' theorem**: Under H0 with converged models, T ~ χ²(DOF) where DOF = number of extra parameters in numerator (100 kernels here)
   - **Kernel overfitting**: Extra kernels with no L2 regularization fit finite-sample fluctuations, inflating T far beyond χ²(100)

3. Files and Code Sections:

   - **`/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/LRT.py`** — Main LRT pipeline script
     - Loads 60-member Gaussian kernel ensemble from `ensemble_dir`
     - Loads pre-fitted WiFi weights (`w_path`, `w_cov_path`)
     - Evaluates ensemble probabilities on target data: `model_probs (N, 60)`, `model_norm_probs (N, 1)`
     - Instantiates DEN (ensemble weights only, `train_net=False`) and NUM (weights + 100 kernels, `train_net=True`)
     - Trains both models via `lrt.train_loop()`, then computes T

     **Bug 1 — sign of aux (FIXED):**
     ```python
     # BEFORE (wrong):
     T_tensor = (num_log_data.sum() - aux_num) - (den_log_data.sum() - aux_den)
     test = (num_log_data - den_log_data) + ((aux_den - aux_num) / N)

     # AFTER (correct — lines 447–453):
     # Training maximizes MAP = sum log p(x) + log P(w | prior)
     # Consistent LRT: T = [sum num_log_data + aux_num] - [sum den_log_data + aux_den]
     T_tensor = (num_log_data.sum() + aux_num) - (den_log_data.sum() + aux_den)
     test = (num_log_data - den_log_data) + ((aux_num - aux_den) / N)
     ```

     **Bug 2 — asymmetric training epochs (FIXED):**
     ```python
     # BEFORE:
     epochs_tau   = 200000
     epochs_delta = 20000   # DEN only 1/10th the training of NUM

     # AFTER (line 79):
     epochs_tau   = 200000
     epochs_delta = 200000  # equal training time
     ```

     **Remaining hyperparameter (not yet changed):**
     ```python
     lambda_L2_numerator = 0   # ← core issue causing kernel overfitting
     kernel_width_numerator = 0.08  # narrow kernels placed at data points
     n_kernels_numerator = 100
     clip_tau = 0.1  # coefficient clipping range
     ```

   - **`/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Train_Models/Sparker_utils/LRTGOFutils_v2.py`** — Core model library
     - `GaussianKernelLayer`: RBF kernels with mean-centred, clipped coefficients
     - `TAU`: Main model; `call()` returns `ensemble (N,1)` when `train_net=False`, or `(ensemble (N,1), net_out (N,))` when `train_net=True`
     - `loss()`: minimises `−sum log p − log P(w)` (MAP negative log-posterior)
     - `loglik()`: **FIXED** — was returning `sum log p − log P(w)` (wrong sign), now returns `sum log p + log P(w) = −loss` (correct)
     - `log_auxiliary_term()`: returns `log P(w | prior)` from MultivariateNormal, always negative

     ```python
     # loss() — correct MAP objective (unchanged):
     def loss(self, x):
         aux = self.log_auxiliary_term()       # log P(w), negative
         out = -torch.log(p).sum()             # positive
         if self.train_weights:
             out = out - aux.sum()             # = -sum log p - log P(w) ✓
         return out

     # loglik() — FIXED (line 203):
     # BEFORE: out = out - aux.sum()   # gave sum log p - log P(w) ← wrong
     # AFTER:
     if self.train_weights:
         out = out + aux.sum()                 # = sum log p + log P(w) = -loss ✓
     ```

   - **Log files read:**
     - `LRT_toys-28638_0.out`: Original run, `epochs_delta=20000`, sign bug → **T = 151,965**
     - `LRT_toys-38573_0.out`: After sign fix only, `epochs_delta=20000` → **T = 115,117**
     - `LRT_toys-39576_0.out`: After sign fix + `epochs_delta=200000` → **T = 23,344**

   - **`/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/LRT/Sparker_kernels/submit_LRT_toys.sh`** — SLURM submission script
     - Confirms CALIBRATION=0 mode: uses `TARGET_DATA = 500k_2d_gaussian_heavy_tail_target_set.npy`
     - CALIBRATION=1 path (`CALIB_DATA`) exists in script but is commented out; corresponding code not yet implemented

4. Errors and Fixes:

   - **Bug 1 — Wrong sign of auxiliary term in T (inflated T by ~37k)**:
     - Root cause: Training maximises `sum log p + log P(w)` (MAP), but test statistic used `sum log p − log P(w)`. The discrepancy is `T_code = T_correct + 2*(log P(w_den) − log P(w_num))`. Since the numerator weights deviate more from the prior, `T_code > T_correct`.
     - Fix: Changed `−aux_num`, `−aux_den` to `+aux_num`, `+aux_den` in `LRT.py` T computation, and flipped the per-event aux spread. Also fixed `loglik()` in `LRTGOFutils_v2.py`.
     - Confirmed: T dropped from 151,965 → 115,117 = `loss_den_final − loss_num_final` exactly as predicted.

   - **Bug 2 — Asymmetric training epochs (inflated T by ~92k)**:
     - Root cause: DEN trained 20k epochs, NUM trained 200k epochs. At epoch 20k both models had nearly identical losses (gap = 3,100 nats). The remaining 180k epochs let NUM's kernels keep learning while DEN was frozen, creating a 115k-nat gap.
     - Fix: Set `epochs_delta = 200000`. This was applied twice — first application was overwritten by user edits to LRT.py, requiring reapplication.
     - Confirmed: After equal training, T = 23,344. DEN final loss: −2.051e+05; NUM final loss: −2.284e+05.

   - **Note on fix reversion**: When the user edited LRT.py to update the `run_tag` string (adding `_claude` and `_claude_v2` suffixes), the `epochs_delta = 200000` change was reverted to 20000. It had to be re-applied a second time.

5. Problem Solving:

   **Solved:**
   - Sign bug in `T` computation: T reduced by ~37k
   - Asymmetric training: T reduced from 115k → 23k
   - Clarified test design: CALIBRATION=0 is a 1-sample profile LRT where both DEN and NUM retrain WiFi weights on target data (confirmed correct by user)

   **Ongoing / Not yet solved:**
   - T = 23,344 remains far above the expected χ²(100) ≈ 100 under H0
   - Root cause: `lambda_L2_numerator = 0` gives 100 narrow kernels (σ = 0.08) unlimited freedom to fit finite-sample fluctuations
   - Evidence: At ep 200k, DEN improves at ~40 nats/epoch (nearly plateaued), NUM still improves at ~140 nats/epoch (kernels still learning)
   - Each kernel delivers ~233 nats/param improvement; Wilks' theorem predicts ~0.5 nats/param under H0
   - This is now a **statistical/hyperparameter problem, not a code bug**: the code is mathematically correct, but λ = 0 allows overfitting
   - Proper calibration requires implementing CALIBRATION=1 (2-sample null toys from ensemble)

6. All User Messages:
   - "hey, can you give a look at LRT.py, LRTGOFutils_v2.py and then I'll ask you specific questions"
   - "so basically if you look at the LRT-toys28638_0.out file, you can see that the tests statistics value is huge, and it shouldn't be like that. I am trying to figure out why is that happening, can you help me?"
   - "yes please tell me how to implement these 2 fixes"
   - "well the problem wasn't solved.... look at LRT_toys-38573_0.out"
   - "okay yeah if you look at submit_LRT_toys.sh, basically right now we are training in CALIBRATION=0 mode, which is comparing the ensemble with the target distribution data, where this is a 1-sample test and hence the ensemble enters as analytic term. the CALIBRATION=1 mode would be comparing ensemble to ensemble, where it would be needed to sample from the ensemble as well (but i don't have the code for that yet)"
   - (Answer to question about DEN design): "DEN should retrain weights on target data (current code)"
   - "okay if you look at LRT_toys-39576_0.out, the test statistics decreased, but it is still very high compared to the values we are expecting. Is it possible something is still not correct in the LRT.py?"
   - "Please summarize this entire debugging session in a structured technical report..." [current message]

7. Pending Tasks:
   - Determine and apply an appropriate value for `lambda_L2_numerator` to regularise the kernel network and reduce T to ~χ²(100) under H0
   - Implement CALIBRATION=1 mode (2-sample test: generate samples from ensemble and compare to ensemble) — this is needed to empirically calibrate T and find the right λ value
   - Run calibration toys to verify T distribution converges to χ²(100) with chosen λ

8. Current Work:
   All code fixes have been applied. The last log analysed was `LRT_toys-39576_0.out` showing T = 23,344 after sign fix + equal 200k epochs for both DEN and NUM. The user asked whether something is still wrong in `LRT.py`. The conclusion reached is that the code is now **mathematically correct**; the remaining large T is due to `lambda_L2_numerator = 0` causing kernel overfitting, which is a hyperparameter issue requiring calibration rather than a code bug.

   Current state of LRT.py (key lines):
   ```python
   # Line 78-83 — hyperparameters:
   epochs_tau   = 200000   # NUM training epochs
   epochs_delta = 200000   # DEN training epochs (fixed)
   kernel_width_numerator = 0.08
   lambda_L2_numerator = 0   # ← needs to be > 0 to control overfitting

   # Lines 447–453 — T computation (fixed):
   T_tensor = (num_log_data.sum() + aux_num) - (den_log_data.sum() + aux_den)
   T = float(T_tensor.detach().cpu().item())
   test = (num_log_data - den_log_data) + ((aux_num - aux_den) / N)
   ```

9. Optional Next Step:
   The user asked: "Is it possible something is still not correct in the LRT.py?" The answer is no — the code is now correct. The next concrete step is to set `lambda_L2_numerator` to a non-zero value and run calibration toys (CALIBRATION=1, once implemented) to find the λ that brings T into the χ²(100) regime. As a first experiment, try `lambda_L2_numerator = 1e3` or `1e4` in a test run and observe how much T decreases.

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /t3home/gbadarac/.claude/projects/-work-gbadarac-MonoJet-NPLM-MonoJet-NPLM-analysis/27a8a166-41f6-4453-9677-0526de3d1e3b.jsonl
Please continue the conversation from where we left off without asking the user any further questions. Continue with the last task that you were asked to work on.