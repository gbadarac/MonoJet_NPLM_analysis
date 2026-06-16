#!/usr/bin/env python
"""
Unified WiFi weight fitting for kernel and NF ensembles.

Always optimises M-1 free parameters (last weight = 1 - sum(others)).
Covariance output is (M-1)×(M-1) sandwich estimator for both model types.
Outputs: w_i_fitted.npy (M,), cov_w.npy (M-1, M-1), loss_history.npy
Plots saved to wifi_ensemble_plots/ (marginals) and plots/ (diagnostics).
"""
import sys, gc, json, argparse, ctypes
from pathlib import Path
import numpy as np
import torch
from torch.autograd.functional import hessian as torch_hessian
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def _np2t(arr, dtype=torch.float32):
    """numpy → tensor, safe for numpy 2.x + old torch (ctypes fallback)."""
    arr_c = np.ascontiguousarray(arr, dtype=np.float32 if dtype == torch.float32 else np.float64)
    try:
        return torch.from_numpy(arr_c).to(dtype)
    except TypeError:
        t = torch.empty(arr_c.shape, dtype=dtype)
        ctypes.memmove(t.data_ptr(), arr_c.ctypes.data_as(ctypes.c_void_p), arr_c.nbytes)
        return t


def _t2np(t, dtype=np.float64):
    """tensor → numpy, safe for numpy 2.x + old torch (ctypes fallback)."""
    t_c = t.to(torch.float64 if dtype == np.float64 else torch.float32).cpu().contiguous()
    arr = np.empty(t_c.shape, dtype=dtype)
    ctypes.memmove(arr.ctypes.data_as(ctypes.c_void_p), t_c.data_ptr(), arr.nbytes)
    return arr

# ── Repo paths ────────────────────────────────────────────────────────────────
THIS_DIR  = Path(__file__).resolve().parent   # .../Fit_Weights
REPO_ROOT = THIS_DIR.parents[2]               # .../MonoJet_NPLM_analysis
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "shared" / "Sparker_utils"))
sys.path.insert(0, str(REPO_ROOT / "Train_Ensembles" / "Train_Models"))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Unified WiFi weight fitter (kernels or NF)."
)
parser.add_argument("--model_type", required=True, choices=["kernels", "nf"])
parser.add_argument("--data_path",  required=True,
                    help="Path to .npy target data (N, d).")
parser.add_argument("--out_dir",    required=True,
                    help="Directory where outputs are written.")
parser.add_argument("--n_wifi_components", type=int, required=True,
                    help="Number of ensemble members M.")
# Kernels-specific
parser.add_argument("--folder_path", default=None,
                    help="[kernels] Dir with config.json + seed*/ subdirs.")
# NF-specific
parser.add_argument("--trial_dir", default=None,
                    help="[nf] Dir with f_i.pth + architecture_config.json.")
# Optimiser
parser.add_argument("--epochs",   type=int,   default=2000)
parser.add_argument("--patience", type=int,   default=10,
                    help="Stop after this many log intervals with no improvement.")
parser.add_argument("--lr",       type=float, default=0.1)
parser.add_argument("--no_plots", action="store_true")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
M = args.n_wifi_components

# ── Load data ─────────────────────────────────────────────────────────────────
data_np = np.load(args.data_path)
ndim = data_np.shape[1]
print(f"Data shape: {data_np.shape}", flush=True)

# ── Load ensemble and evaluate model_probs (N, M) ─────────────────────────────
if args.model_type == "kernels":
    if args.folder_path is None:
        raise ValueError("--folder_path required for model_type=kernels")
    from kernel_wifi_ensemble_utils import build_wifi_ensemble
    from Uncertainty_Modeling.wifi.utils_kernel_wifi import (
        plot_ensemble_marginals_2d_kernel,
        plot_final_marginals_and_ratio,
    )
    ensemble, _ = build_wifi_ensemble(
        folder_path=Path(args.folder_path).resolve(),
        n_wifi_components=M,
        train_centroids=False, train_coeffs=False,
        train_widths=False, train_weights=True,
        weights_activation=None,
    )
    ensemble = ensemble.to(device=torch.device("cpu"), dtype=torch.float64)
    data_t = _np2t(data_np, dtype=torch.float64)
    print("Evaluating kernel ensemble on data ...", flush=True)
    with torch.no_grad():
        model_probs = torch.clamp_min(
            ensemble.member_probs(data_t).double(), 0.0
        ).cpu()   # (N, M)

elif args.model_type == "nf":
    if args.trial_dir is None:
        raise ValueError("--trial_dir required for model_type=nf")
    from utils_flows import make_flow
    from Uncertainty_Modeling.wifi.utils_NF_wifi import (
        plot_ensemble_marginals_2d,
        plot_ensemble_marginals_4d,
    )
    trial_dir = Path(args.trial_dir).resolve()
    with open(trial_dir / "architecture_config.json") as f:
        arch = json.load(f)
    flow_kwargs = {k: v for k, v in arch.items() if k != "backend"}
    f_i_statedicts = torch.load(str(trial_dir / "f_i.pth"), map_location="cpu")
    device_nf = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_t = _np2t(data_np, dtype=torch.float32)
    print(f"Evaluating {M} NF models on data ...", flush=True)
    rows = []
    for i, sd in enumerate(f_i_statedicts[:M]):
        flow = make_flow(**flow_kwargs).to(device_nf).float().eval()
        flow.load_state_dict(sd)
        chunks = []
        with torch.no_grad():
            for j in range(0, len(data_t), 5000):
                xb = data_t[j:j + 5000].to(device_nf)
                chunks.append(torch.exp(flow.log_prob(xb)).cpu().double())
        rows.append(torch.cat(chunks))
        del flow; gc.collect()
        if device_nf.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  model {i+1}/{M}", flush=True)
    model_probs = torch.clamp(torch.stack(rows, dim=1), min=0).double()  # (N, M)

print(
    f"model_probs: shape={tuple(model_probs.shape)}  "
    f"min={float(model_probs.min()):.3e}  max={float(model_probs.max()):.3e}  "
    f"frac<=0={float((model_probs <= 0).double().mean()):.4f}",
    flush=True,
)

# ── Optimise M-1 free weights via Adam ────────────────────────────────────────
device_fit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mp = model_probs.to(device_fit)


def build_w(u: torch.Tensor) -> torch.Tensor:
    return torch.cat([u, (1.0 - u.sum()).view(1)])


def nll(u: torch.Tensor) -> torch.Tensor:
    p = (mp * build_w(u)).sum(1)
    return -torch.log(torch.clamp(p, 1e-300)).mean()


u = torch.nn.Parameter(
    torch.full((M - 1,), 1.0 / M, dtype=torch.float64, device=device_fit)
)
opt = torch.optim.Adam([u], lr=args.lr)

log_every = max(1, args.epochs // 40)
best, bad = float("inf"), 0
loss_hist = []

for ep in range(1, args.epochs + 1):
    opt.zero_grad(set_to_none=True)
    nll(u).backward()
    opt.step()
    if ep % log_every == 0:
        with torch.no_grad():
            cur = float(nll(u))
        gnorm = float(u.grad.norm()) if u.grad is not None else float("nan")
        print(f"ep {ep:6d}  loss {cur:.6e}  |g| {gnorm:.2e}", flush=True)
        loss_hist.append(cur)
        if cur < best:
            best, bad = cur, 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at ep {ep}, best={best:.6e}")
                break

with torch.no_grad():
    w_final_t = build_w(u.detach()).cpu()   # (M,) tensor

w_final = _t2np(w_final_t)
np.save(out_dir / "w_i_fitted.npy", w_final)
np.save(out_dir / "loss_history.npy", np.array(loss_hist, dtype=np.float32))
print(f"w_i_fitted.npy saved  sum={float(w_final_t.sum()):.8f}", flush=True)

# ── Sandwich covariance → (M-1)×(M-1) ────────────────────────────────────────
mp_cpu = model_probs.cpu()
w_t    = w_final_t.to(torch.float64)
f      = (mp_cpu * w_t).sum(1)
q01    = float(torch.quantile(f, 0.001))
eps    = max(1e-300, q01 * 0.1)
f_safe = torch.clamp(f, min=eps)

G  = -(mp_cpu[:, :-1] - mp_cpu[:, -1:]) / f_safe.unsqueeze(1)  # (N, M-1)
Gc = G - G.mean(0, keepdim=True)
N  = mp_cpu.shape[0]
U  = (Gc.T @ Gc) / N

def nll_cpu(u_vec: torch.Tensor) -> torch.Tensor:
    p = (mp_cpu * build_w(u_vec)).sum(1)
    return -torch.log(torch.clamp(p, 1e-300)).mean()


u_h = w_final_t[:-1].clone().to(torch.float64).requires_grad_(True)
H   = torch_hessian(nll_cpu, u_h).detach()
H   = 0.5 * (H + H.T)

eigs = torch.linalg.eigvalsh(H)
print(f"Hessian eigvals: min={eigs.min():.3e}  max={eigs.max():.3e}", flush=True)

ridge = 1e-10 + 1e-8 * float(eigs.abs().max())
V     = torch.linalg.solve(
    H + ridge * torch.eye(M - 1, dtype=torch.float64),
    torch.eye(M - 1, dtype=torch.float64),
)
cov_u = (V @ U @ V.T) / N

ce, Ue = torch.linalg.eigh(cov_u)
cov_u  = Ue @ torch.diag(torch.clamp(ce, 1e-18)) @ Ue.T
cov_np = _t2np(cov_u.detach())

np.save(out_dir / "cov_w.npy", cov_np)
cov_diag = cov_u.diagonal()
print(
    f"cov_w.npy saved  shape={tuple(cov_u.shape)}  "
    f"diag [{float(cov_diag.min()):.3e}, {float(cov_diag.max()):.3e}]",
    flush=True,
)

# ── Plots ─────────────────────────────────────────────────────────────────────
if not args.no_plots:
    wifi_plots_dir = out_dir / "wifi_ensemble_plots"
    wifi_plots_dir.mkdir(exist_ok=True)
    diag_plots_dir = out_dir / "plots"
    diag_plots_dir.mkdir(exist_ok=True)

    # Simple diagnostics
    fig, ax = plt.subplots()
    ax.plot(loss_hist)
    ax.set_xlabel("log interval"); ax.set_ylabel("NLL")
    ax.set_title("WiFi weight fitting — loss history")
    fig.savefig(diag_plots_dir / "loss_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(range(M), w_final)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Model index"); ax.set_ylabel("Weight")
    ax.set_title("Final WiFi weights")
    fig.savefig(diag_plots_dir / "final_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(range(M - 1), _t2np(cov_u.diagonal().sqrt()))
    ax.set_xlabel("Free weight index"); ax.set_ylabel(r"$\sigma(w_i)$")
    ax.set_title("Weight uncertainties (sqrt of cov diagonal)")
    fig.savefig(diag_plots_dir / "weight_uncertainties.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Model-type specific marginal plots
    if args.model_type == "kernels":
        # Copy final weights back into ensemble object for plotting
        with torch.no_grad():
            ensemble.weights.copy_(
                _np2t(w_final, dtype=ensemble.weights.dtype).to(
                    device=torch.device("cpu")
                )
            )

        # Grid for 2D ratio plot
        pad = 0.05
        x0_lo, x0_hi = float(data_t[:, 0].min()), float(data_t[:, 0].max())
        x1_lo, x1_hi = float(data_t[:, 1].min()), float(data_t[:, 1].max())
        x0_pad = pad * (x0_hi - x0_lo + 1e-12)
        x1_pad = pad * (x1_hi - x1_lo + 1e-12)
        x0g = torch.arange(x0_lo - x0_pad, x0_hi + x0_pad, 0.01, dtype=torch.float64)
        x1g = torch.arange(x1_lo - x1_pad, x1_hi + x1_pad, 0.005, dtype=torch.float64)
        X0, X1 = torch.meshgrid(x0g, x1g, indexing="xy")
        grid = torch.stack([X0.flatten(), X1.flatten()], dim=1)
        with torch.no_grad():
            Y = ensemble(grid)

        feature_names = [f"Feature {i+1}" for i in range(ndim)]
        plot_final_marginals_and_ratio(
            ensemble, data_t, grid, x0g, x1g, Y,
            outdir=str(wifi_plots_dir), tag="final",
        )
        plot_ensemble_marginals_2d_kernel(
            kernel_models=ensemble.ensemble,
            x_data=data_t.detach().cpu(),
            weights=ensemble.weights.detach().cpu(),
            cov_w=cov_np,
            feature_names=feature_names,
            outdir=str(wifi_plots_dir),
            bins=40,
        )

    elif args.model_type == "nf":
        # Reload flow models for plotting
        print("Reloading NF models for marginal plots ...", flush=True)
        f_i_models = []
        for sd in f_i_statedicts[:M]:
            flow = make_flow(**flow_kwargs).to("cpu").float().eval()
            flow.load_state_dict(sd)
            f_i_models.append(flow)

        x_data_plot = data_t.to("cpu")
        feature_names = [f"Feature {i+1}" for i in range(ndim)]

        if ndim == 2:
            plot_ensemble_marginals_2d(
                f_i_models, x_data_plot, w_final_t, cov_np,
                feature_names, str(wifi_plots_dir),
            )
        elif ndim == 4:
            plot_ensemble_marginals_4d(
                f_i_models, x_data_plot, w_final_t, cov_np,
                feature_names, str(wifi_plots_dir),
            )

    print(f"Plots saved to {wifi_plots_dir} and {diag_plots_dir}", flush=True)

print("Done.", flush=True)
