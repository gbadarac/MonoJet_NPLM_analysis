#!/usr/bin/env python
"""
Single-NF-member ("ensemblecomponents1") marginal plot WITHOUT uncertainties.

This is the M=1 companion to fit_ensemble_weights.py: one ensemble member, weight
fixed to 1, so there are no free weights and no weight-covariance -> the marginal
band collapses to zero width (i.e. the plot shows the target vs the single model's
marginal, with no uncertainty band). It reuses the exact same plotting functions as
the ensemble fit (plot_ensemble_marginals_2d for 2D, plot_ensemble_marginals_nd for
D>=3) so the style matches; the only difference is weights=[1.0] and cov_w=zeros((0,0))
-> the uncertainty band collapses to zero width.

Outputs, under <out_dir>:
    w_i_fitted.npy         # [1.0]
    cov_w.npy              # shape (0,0)
    loss_history.npy       # [ -mean log f_seed(x) ] (single value, for reference)
    wifi_ensemble_plots/ensemble_marginals.png  (both features, one figure) + marginal_*_data.npz

Usage:
    python plot_single_seed_marginal.py \
        --trial_dir <.../N_..._seeds_128_...> --seed 100 \
        --data_path <.../data_train.npy> --out_dir <...ensemblecomponents1>
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")

THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Train_Ensembles" / "Train_Models"))

from utils_flows import make_flow
from Uncertainty_Modeling.wifi.utils_NF_wifi import (
    plot_ensemble_marginals_2d,
    plot_ensemble_marginals_nd,
)

ap = argparse.ArgumentParser()
ap.add_argument("--trial_dir", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--data_path", required=True)
ap.add_argument("--out_dir", required=True)
args = ap.parse_args()

out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
trial_dir = Path(args.trial_dir).resolve()

with open(trial_dir / "architecture_config.json") as f:
    arch = json.load(f)
flow_kwargs = {k: v for k, v in arch.items() if k != "backend"}

mp = trial_dir / f"model_{args.seed:03d}" / "model.pth"
if not mp.exists():
    raise FileNotFoundError(f"seed {args.seed} -> {mp} not found")

data_np = np.load(args.data_path)
ndim = data_np.shape[1]
x_data = torch.from_numpy(np.ascontiguousarray(data_np, dtype=np.float32))

flow = make_flow(**flow_kwargs).to("cpu").float().eval()
flow.load_state_dict(torch.load(str(mp), map_location="cpu"))
print(f"Loaded single member seed={args.seed} from {mp}", flush=True)

# reference NLL of this member on the data (single-value "loss history")
with torch.no_grad():
    chunks = [flow.log_prob(x_data[j:j + 5000]) for j in range(0, len(x_data), 5000)]
    nll = float(-torch.cat(chunks).mean())
print(f"single-member NLL on data = {nll:.6f}", flush=True)

# weight fixed to 1, no free params -> no covariance -> zero-width band
w_final_t = torch.ones(1, dtype=torch.float64)
cov_np = np.zeros((0, 0), dtype=np.float64)
np.save(out_dir / "w_i_fitted.npy", np.array([1.0]))
np.save(out_dir / "cov_w.npy", cov_np)
np.save(out_dir / "loss_history.npy", np.array([nll], dtype=np.float32))

wifi_plots_dir = out_dir / "wifi_ensemble_plots"
wifi_plots_dir.mkdir(exist_ok=True)
feature_names = [f"Feature {i+1}" for i in range(ndim)]

if ndim == 2:
    # exact grid marginal (unbiased) — 2D only
    plot_ensemble_marginals_2d(
        [flow], x_data, w_final_t, cov_np, feature_names, str(wifi_plots_dir),
        n_components=1,
    )
else:
    # any D >= 3: Monte-Carlo marginal (matches fit_ensemble_weights.py). Single member
    # + cov_w=(0,0) -> the uncertainty band degenerates to zero width.
    plot_ensemble_marginals_nd(
        [flow], x_data, w_final_t, cov_np, feature_names, str(wifi_plots_dir),
    )
print(f"Marginal plot (no uncertainty band) saved to {wifi_plots_dir}", flush=True)
print("Done.", flush=True)
