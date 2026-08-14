#!/usr/bin/env python
"""
Plot data-vs-model marginals + held-out test NLL for each NF member (model_*/)
in a trial dir. Reads architecture_config.json + model_*/model.pth, evaluates on
the seed-42 data_test.npy, and saves marginals_<model>.png into the trial dir.

scipy-free (nf_env's scipy is broken): the true-density floor is computed with
numpy + torch.erf. Works for a single member or a full ensemble dir.

Usage:
    python plot_nf_marginals.py --trial_dir <.../N_..._seeds_M_...>
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/work/gbadarac/MonoJet_NPLM/MonoJet_NPLM_analysis"
sys.path.insert(0, os.path.join(REPO, "Train_Ensembles", "Train_Models"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_flows import make_flow

DATA_DIR = os.path.join(REPO, "data", "2d_gmm_toymodel",
                        "2d_gmm_skew_Ntrain100000_Ntest100000_seed42")

ap = argparse.ArgumentParser()
ap.add_argument("--trial_dir", required=True)
ap.add_argument("--n_sample", type=int, default=50000)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xte = np.load(os.path.join(DATA_DIR, "data_test.npy")).astype("float32")


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

arch = json.load(open(os.path.join(args.trial_dir, "architecture_config.json")))
kw = {k: v for k, v in arch.items() if k != "backend"}

members = sorted(glob.glob(os.path.join(args.trial_dir, "model_*", "model.pth")))
if not members:
    print(f"no model_*/model.pth under {args.trial_dir}")
    sys.exit(1)
print(f"arch={kw}\nfloor(test NLL)={FLOOR:.5f}  members={len(members)}")

for mp in members:
    tag = os.path.basename(os.path.dirname(mp))
    flow = make_flow(**kw).to(device).eval()
    flow.load_state_dict(torch.load(mp, map_location=device))
    xt = torch.from_numpy(Xte).to(device)
    with torch.no_grad():
        out = [flow.log_prob(xt[j:j + 10000]).cpu() for j in range(0, len(xt), 10000)]
        nll = float(-torch.cat(out).mean())
        xs = flow.sample(args.n_sample).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(2):
        lo, hi = np.quantile(Xte[:, k], [0.001, 0.999])
        b = np.linspace(lo, hi, 80)
        ax[k].hist(Xte[:, k], bins=b, density=True, histtype="step", lw=2, label="data (test)")
        ax[k].hist(xs[:, k], bins=b, density=True, histtype="step", lw=2, label="NF model")
        ax[k].set_xlabel(f"Feature {k+1}")
        ax[k].legend()
    fig.suptitle(f"{os.path.basename(args.trial_dir)} / {tag}   "
                 f"test NLL={nll:.4f}  (floor {FLOOR:.4f}, gap {nll-FLOOR:+.4f})")
    fig.tight_layout()
    outp = os.path.join(os.path.dirname(mp), f"marginals_{tag}.png")  # inside model_NNN/
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag}: test NLL {nll:.5f}  gap {nll-FLOOR:+.5f}  -> {outp}")

print("done")
