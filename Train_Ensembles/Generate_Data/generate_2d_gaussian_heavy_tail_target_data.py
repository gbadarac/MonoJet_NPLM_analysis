import numpy as np, os

# Repro & paths
np.random.seed(1234)
output_dir = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis/Train_Ensembles/Generate_Data/saved_generated_target_data/2_dim"
os.makedirs(output_dir, exist_ok=True)
training_file = os.path.join(output_dir, "500k_2d_gaussian_heavy_tail_target_set.npy")

N = 500000

# ---- Feature 1: Bi-modal Gaussian mixture (clearly bimodal) ----
wG = 0.50
mu_a, sig_a = -0.70, 0.12
mu_b, sig_b = -0.30, 0.12
n1_a = np.random.binomial(N, wG)
x1 = np.concatenate([
    np.random.normal(mu_a, sig_a, n1_a),
    np.random.normal(mu_b, sig_b, N - n1_a),
])
E_x1 = wG * mu_a + (1.0 - wG) * mu_b  # analytic mean (pre-scaling)

## ---- Feature 2: Skew-Normal (strong skew, simple & analytic) ----
def sample_skewnorm(n, loc, scale, alpha):
    alpha = float(alpha)
    delta = alpha / np.sqrt(1.0 + alpha**2)
    z0 = np.random.randn(n); z1 = np.random.randn(n)
    xstd = delta * np.abs(z0) + np.sqrt(1.0 - delta**2) * z1
    return loc + scale * xstd

loc2, scale2, alpha2 = 1.0, 0.75, 8.0
x2 = sample_skewnorm(N, loc2, scale2, alpha2)

# analytic mean (pre-scaling)
delta = alpha2 / np.sqrt(1.0 + alpha2**2)
E_x2 = loc2 + scale2 * delta * np.sqrt(2.0 / np.pi)

# Stack & save
X = np.column_stack([x1, x2]).astype('float32')
np.save(training_file, X)

mu_target = np.array([E_x1, E_x2], dtype=np.float32)
np.save(os.path.join(output_dir, "mu_2d_gaussian_heavy_tail_target.npy"), mu_target)
print("sample mean:", X.mean(axis=0))
print("analytic mu_target:", mu_target)


