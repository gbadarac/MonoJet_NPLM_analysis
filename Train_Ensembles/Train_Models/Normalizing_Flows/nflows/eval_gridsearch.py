#!/usr/bin/env python
"""
Rank the NF architecture grid search by held-out (seed-42 test) NLL and plot
data-vs-model marginals for each config.

For each <gridsearch>/L*_h*_bins*/ dir it reads architecture_config.json +
model_000/model.pth, evaluates NLL on data_test.npy, and compares to the true
2D-toymodel density's cross-entropy floor (the best any estimator can do).
Outputs go to <gridsearch>/eval/.
"""
import os, sys, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
sys.path.insert(0, os.path.join(REPO, "Train_Ensembles", "Train_Models"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_flows import make_flow

NFLOWS_DIR = os.path.join(REPO, "Train_Ensembles", "Train_Models", "Normalizing_Flows", "nflows")
GRID_BASE = os.path.join(NFLOWS_DIR, "EstimationNFnflows_outputs", "2_dim",
                         "2d_bimodal_gaussian_heavy_tail", "gridsearch")
DATA_DIR = os.path.join(REPO, "data", "2d_gmm_toymodel",
                        "2d_gmm_skew_Ntrain100000_Ntest100000_seed42")
EVAL_DIR = os.path.join(GRID_BASE, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- data + true-density floor (features independent: bimodal gauss x skew-normal) ---
Xtr = np.load(os.path.join(DATA_DIR, "data_train.npy")).astype("float32")
Xte = np.load(os.path.join(DATA_DIR, "data_test.npy")).astype("float32")
# scipy is broken in nf_env (numpy/scipy ABI mismatch) -> compute the true
# toymodel density with numpy + torch.erf instead of scipy.stats.
def _norm_pdf(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

def _skewnorm_pdf(x, a, loc, scale):
    z = ((x - loc) / scale).astype(np.float64)
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1.0 + torch.erf(torch.from_numpy(a * z / np.sqrt(2.0))).numpy())
    return (2.0 / scale) * phi * Phi

ftrue = ((0.5 * _norm_pdf(Xte[:, 0], -0.70, 0.12) + 0.5 * _norm_pdf(Xte[:, 0], -0.30, 0.12))
         * _skewnorm_pdf(Xte[:, 1], 8.0, 1.0, 0.75))
FLOOR = float(-np.log(ftrue + 1e-300).mean())
print(f"true-density NLL floor (test) = {FLOOR:.5f}\n")


def nll_on(flow, X):
    xt = torch.from_numpy(X).to(device)
    out = []
    with torch.no_grad():
        for j in range(0, len(xt), 10000):
            out.append(flow.log_prob(xt[j:j + 10000]).cpu())
    return float(-torch.cat(out).mean())


rows = []
for d in sorted(glob.glob(os.path.join(GRID_BASE, "L*"))):
    cfg_p = os.path.join(d, "architecture_config.json")
    mdl_p = os.path.join(d, "model_000", "model.pth")
    tag = os.path.basename(d)
    if not (os.path.exists(cfg_p) and os.path.exists(mdl_p)):
        print(f"[skip] incomplete: {tag}")
        continue
    arch = json.load(open(cfg_p))
    kw = {k: v for k, v in arch.items() if k != "backend"}
    flow = make_flow(**kw).to(device).eval()
    flow.load_state_dict(torch.load(mdl_p, map_location=device))

    nll_te = nll_on(flow, Xte)
    nll_tr = nll_on(flow, Xtr)
    rows.append(dict(tag=tag, layers=arch["num_layers"], hidden=arch["hidden_features"],
                     bins=arch["num_bins"], blocks=arch["num_blocks"],
                     nll_test=nll_te, nll_train=nll_tr, gap=nll_te - FLOOR))

    # marginals: model samples vs test data
    with torch.no_grad():
        xs = flow.sample(50000).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(2):
        lo, hi = np.quantile(Xte[:, k], [0.001, 0.999])
        b = np.linspace(lo, hi, 80)
        ax[k].hist(Xte[:, k], bins=b, density=True, histtype="step", lw=2, label="data (test)")
        ax[k].hist(xs[:, k], bins=b, density=True, histtype="step", lw=2, label="NF model")
        ax[k].set_xlabel(f"Feature {k+1}"); ax[k].legend()
    fig.suptitle(f"{tag}   test NLL={nll_te:.4f}  (floor {FLOOR:.4f}, gap {nll_te-FLOOR:+.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, f"marginals_{tag}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag:28s}  test NLL {nll_te:.5f}  (gap {nll_te-FLOOR:+.5f})")

if not rows:
    print("No completed configs found. Did the array job finish?")
    sys.exit(1)

rows.sort(key=lambda r: r["nll_test"])

# --- ranking table ---
lines = [f"true-density NLL floor (test) = {FLOOR:.5f}", "",
         f"{'rank':>4}  {'architecture':28s}  {'test_NLL':>9}  {'gap_to_floor':>12}  {'train_NLL':>9}"]
for i, r in enumerate(rows, 1):
    lines.append(f"{i:>4}  {r['tag']:28s}  {r['nll_test']:>9.5f}  {r['gap']:>+12.5f}  {r['nll_train']:>9.5f}")
table = "\n".join(lines)
print("\n" + table)
with open(os.path.join(EVAL_DIR, "ranking.txt"), "w") as f:
    f.write(table + "\n")

# --- summary bar chart ---
fig, ax = plt.subplots(figsize=(11, 5))
tags = [r["tag"] for r in rows]
ax.bar(range(len(rows)), [r["nll_test"] for r in rows], color="C0")
ax.axhline(FLOOR, color="r", ls="--", label=f"true-density floor {FLOOR:.4f}")
ax.set_xticks(range(len(rows))); ax.set_xticklabels(tags, rotation=90)
ax.set_ylabel("held-out test NLL"); ax.set_title("NF architecture grid search (lower = better fit)")
ax.legend(); ax.set_ylim(FLOOR - 0.02, max(r["nll_test"] for r in rows) + 0.02)
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, "summary_nll.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

print(f"\nBest: {rows[0]['tag']}  (test NLL {rows[0]['nll_test']:.5f}, gap {rows[0]['gap']:+.5f})")
print(f"Wrote ranking + plots to {EVAL_DIR}")
