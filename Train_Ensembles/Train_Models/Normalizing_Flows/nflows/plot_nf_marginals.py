#!/usr/bin/env python
"""
Plot data-vs-model marginals + held-out test NLL for each NF member (model_*/)
in a trial dir. Reads architecture_config.json + model_*/model.pth, evaluates on
the seed-42 data_test.npy, and saves marginals_<model>.png into each model dir.

Annotated by held-out test NLL only (lower = better): real embeddings have no
closed-form density, so there is no analytic floor to compare to. Works for a
single member or a full ensemble dir, in any dimensionality (num_features is read
from architecture_config.json and one panel is drawn per feature).

Usage:
    python plot_nf_marginals.py --trial_dir <.../N_..._seeds_M_...> \
        [--data_dir <dir with data_test.npy>] [--n_sample 50000]
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

DEFAULT_DATA_DIR = os.path.join(REPO, "data", "4d_embeddings",
                                "4d_embedding_qcd_Ntrain100000_Ntest100000_seed42")

ap = argparse.ArgumentParser()
ap.add_argument("--trial_dir", required=True)
ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                help="dir containing data_test.npy (default: 4D JetClass QCD embedding)")
ap.add_argument("--n_sample", type=int, default=50000)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xte = np.load(os.path.join(args.data_dir, "data_test.npy")).astype("float32")

arch = json.load(open(os.path.join(args.trial_dir, "architecture_config.json")))
kw = {k: v for k, v in arch.items() if k != "backend"}
D = int(arch["num_features"])

members = sorted(glob.glob(os.path.join(args.trial_dir, "model_*", "model.pth")))
if not members:
    print(f"no model_*/model.pth under {args.trial_dir}")
    sys.exit(1)
print(f"arch={kw}\nmembers={len(members)}  data={args.data_dir}")

ncols = int(np.ceil(np.sqrt(D)))
nrows = int(np.ceil(D / ncols))

for mp in members:
    tag = os.path.basename(os.path.dirname(mp))
    flow = make_flow(**kw).to(device).eval()
    flow.load_state_dict(torch.load(mp, map_location=device))
    xt = torch.from_numpy(Xte).to(device)
    with torch.no_grad():
        out = [flow.log_prob(xt[j:j + 10000]).cpu() for j in range(0, len(xt), 10000)]
        nll = float(-torch.cat(out).mean())
        xs = flow.sample(args.n_sample).cpu().numpy()
    fig, ax = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    ax = ax.ravel()
    for k in range(D):
        lo, hi = np.quantile(Xte[:, k], [0.001, 0.999])
        b = np.linspace(lo, hi, 80)
        ax[k].hist(Xte[:, k], bins=b, density=True, histtype="step", lw=2, label="data (test)")
        ax[k].hist(xs[:, k], bins=b, density=True, histtype="step", lw=2, label="NF model")
        ax[k].set_xlabel(f"Feature {k+1}")
        ax[k].legend()
    for k in range(D, len(ax)):
        ax[k].axis("off")
    fig.suptitle(f"{os.path.basename(args.trial_dir)} / {tag}   test NLL={nll:.4f}")
    fig.tight_layout()
    outp = os.path.join(os.path.dirname(mp), f"marginals_{tag}.png")  # inside model_NNN/
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag}: test NLL {nll:.5f}  -> {outp}")

print("done")
