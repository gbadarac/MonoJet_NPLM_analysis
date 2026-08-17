#!/usr/bin/env python
"""
Plot train-vs-val NLL loss curves for each NF member (model_*/) in a trial dir.
Reads model_*/losses.npz (written by EstimationNFnflows.py) and saves
loss_<model>.png into each model_NNN/ dir. Marks the kept checkpoint (best-val
epoch) and prints the final train/val gap so overtraining is visible at a glance.
Mirrors plot_nf_marginals.py: works for a single member or a full ensemble dir.

Usage:
    python plot_nf_loss.py --trial_dir <.../N_..._seeds_M_...>
"""
import os, sys, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--trial_dir", required=True)
args = ap.parse_args()

members = sorted(glob.glob(os.path.join(args.trial_dir, "model_*", "losses.npz")))
if not members:
    print(f"no model_*/losses.npz under {args.trial_dir}\n"
          "(only runs trained with the loss-logging EstimationNFnflows.py have it)")
    sys.exit(1)
print(f"members={len(members)}")

for lp in members:
    tag = os.path.basename(os.path.dirname(lp))          # model_NNN
    d = np.load(lp)
    tr, va = d["train"], d["val"]
    be = int(d["best_epoch"]) if "best_epoch" in d else int(np.argmin(va))
    ep = np.arange(len(tr))
    gap = float(va[be] - tr[be])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ep, tr, lw=2, label="train NLL")
    ax.plot(ep, va, lw=2, label="val NLL")
    ax.axvline(be, ls="--", color="k", alpha=0.6, label=f"kept checkpoint (epoch {be})")
    ax.scatter([be], [va[be]], color="k", zorder=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.set_title(f"{os.path.basename(args.trial_dir)} / {tag}   "
                 f"best val={va[be]:.4f}  train={tr[be]:.4f}  gap {gap:+.4f}")
    ax.legend()
    fig.tight_layout()
    outp = os.path.join(os.path.dirname(lp), f"loss_{tag}.png")   # inside model_NNN/
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {tag}: best val {va[be]:.5f}  train {tr[be]:.5f}  gap {gap:+.5f}  "
          f"epochs={len(tr)}  -> {outp}")

print("done")
