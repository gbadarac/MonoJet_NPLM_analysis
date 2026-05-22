"""
analyse_LRT_output.py
---------------------
Unified analysis script for LRT results — works for both Sparker kernels and
normalizing flow model types. Reads the standard output format produced by LRT.py:

    <results_dir>/<run_tag>/
        calibration/seed{N}/seed{N}_T.npy
        test/seed{N}/seed{N}_T.npy
        calibration/seed{N}/seed{N}_den_weights.npy   (optional)
        calibration/seed{N}/seed{N}_coeffs.npy        (optional, kernels only)

Produces:
    <out_dir>/T_distribution.png         — null + test T distributions vs chi2
    <out_dir>/chi2_quantile_table.txt    — empirical vs chi2 quantile comparison
    <out_dir>/weight_shifts.png          — Δw = w_den - w_init per weight component
    <out_dir>/n_at_clip.png              — kernel clip saturation histogram (if coeffs exist)

Usage:
    python analyse_LRT_output.py \\
        --results_dir /path/to/LRT/results/SparKer32_Ntest100000_... \\
        [--out_dir /path/to/plots/] \\
        [--scale_t 1.0] \\
        [--dof 100] \\
        [--clip_tau 0.0005] \\
        [--xmin 0] [--xmax 200] \\
        [--title "My run"]
"""

import os, glob, argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, beta
from scipy.optimize import minimize_scalar

plt.rcParams["font.family"] = "serif"
plt.style.use("classic")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, required=True,
                    help="Path to the run-tag directory (contains calibration/ and test/).")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for plots (default: <results_dir>/plots/).")
parser.add_argument("--scale_t", type=float, default=1.0,
                    help="Multiply T by this factor before plotting (use 2.0 for 2T convention).")
parser.add_argument("--dof", type=int, default=100,
                    help="Nominal degrees of freedom (number of kernels in NUM).")
parser.add_argument("--clip_tau", type=float, default=0.0005,
                    help="Kernel coefficient clip threshold (for saturation diagnostic).")
parser.add_argument("--xmin", type=float, default=None)
parser.add_argument("--xmax", type=float, default=None)
parser.add_argument("--ymax", type=float, default=None)
parser.add_argument("--title", type=str, default=None,
                    help="Plot title (default: last two components of results_dir).")
args = parser.parse_args()

results_dir = args.results_dir.rstrip("/")
out_dir = args.out_dir or os.path.join(results_dir, "plots")
os.makedirs(out_dir, exist_ok=True)

title = args.title or "/".join(results_dir.split("/")[-2:])
label = "2T" if args.scale_t == 2.0 else "T"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def collect_T(base_dir, mode, scale=1.0):
    """Scan base_dir/<mode>/seed*/seed*_T.npy, return 1-D array."""
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, mode, "seed*")))
    values, bad = [], []
    for sd in seed_dirs:
        name = os.path.basename(sd)
        fs = glob.glob(os.path.join(sd, f"{name}_T.npy"))
        if not fs:
            bad.append((sd, "no T.npy"))
            continue
        try:
            v = float(np.load(fs[0])) * scale
            if np.isfinite(v):
                values.append(v)
            else:
                bad.append((sd, f"non-finite T={v}"))
        except Exception as e:
            bad.append((sd, str(e)))
    if bad:
        print(f"  [{mode}] {len(bad)} seeds skipped:")
        for p, r in bad:
            print(f"    {os.path.basename(p)}: {r}")
    return np.array(values, dtype=float)


def empirical_Z(T_null, T_obs, alpha=0.32):
    """One-sided empirical Z with Clopper-Pearson interval."""
    B = T_null.size
    k = int(np.count_nonzero(T_null >= T_obs))
    p_hat = (k + 1) / (B + 1)
    p_lo = beta.ppf(alpha / 2, k + 1, B - k + 1) if k < B else 0.0
    p_hi = beta.ppf(1 - alpha / 2, k + 1, B - k + 1) if k > 0 else 1.0
    Z     = -norm.ppf(p_hat)
    Z_lo  = -norm.ppf(min(p_hi, 1 - 1e-10))
    Z_hi  = -norm.ppf(max(p_lo, 1e-10))
    return Z, Z_lo, Z_hi, p_hat


def fit_dof_eff(t_calib):
    """Fit chi2(DOF_eff) to calibration T via quantile matching."""
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    eq = np.quantile(t_calib, qs)
    def res(dof):
        return np.sum((eq - chi2.ppf(qs, dof)) ** 2)
    r = minimize_scalar(res, bounds=(5, 500), method="bounded")
    return r.x


def fd_bins(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2 * iqr / (len(x) ** (1 / 3) + 1e-12)
    if h <= 0:
        return 10
    return max(5, int(np.ceil((x.max() - x.min()) / h)))

# -------------------------------------------------------------------
# Load T values
# -------------------------------------------------------------------
print(f"Results dir : {results_dir}")
print(f"Output dir  : {out_dir}")

t_calib = collect_T(results_dir, "calibration", scale=args.scale_t)
t_test  = collect_T(results_dir, "test",        scale=args.scale_t)

print(f"\nCalibration {label}: n={len(t_calib)}", end="")
if len(t_calib):
    print(f"  median={np.median(t_calib):.2f}  std={np.std(t_calib):.2f}", end="")
print()
if len(t_test):
    print(f"Test        {label}: n={len(t_test)}  median={np.median(t_test):.2f}  std={np.std(t_test):.2f}")
    Z, Zlo, Zhi, p = empirical_Z(t_calib, np.median(t_test))
    print(f"Empirical Z (median test vs calib null): {Z:.3f} +{Zhi-Z:.3f}/-{Z-Zlo:.3f}  p={p:.4f}")

if len(t_calib) == 0:
    print("No calibration toys found — exiting.")
    raise SystemExit(1)

# -------------------------------------------------------------------
# DOF_eff fit
# -------------------------------------------------------------------
DOF_eff = fit_dof_eff(t_calib)
print(f"\nNominal DOF   : {args.dof}")
print(f"Fitted DOF_eff: {DOF_eff:.1f}")

print(f"\nCalibration {label} vs chi2({args.dof}) quantiles:")
print(f"{'q':>6}  {'data':>8}  {'chi2':>8}  {'ratio':>7}")
for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
    d = float(np.quantile(t_calib, q))
    c = float(chi2.ppf(q, args.dof))
    print(f"{q:>6.2f}  {d:>8.2f}  {c:>8.2f}  {d/c:>7.3f}")

with open(os.path.join(out_dir, "chi2_quantile_table.txt"), "w") as fh:
    fh.write(f"DOF_nominal={args.dof}  DOF_eff={DOF_eff:.2f}\n")
    fh.write(f"{'q':>6}  {'data':>8}  {'chi2':>8}  {'ratio':>7}\n")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        d = float(np.quantile(t_calib, q))
        c = float(chi2.ppf(q, args.dof))
        fh.write(f"{q:>6.2f}  {d:>8.2f}  {c:>8.2f}  {d/c:>7.3f}\n")

# -------------------------------------------------------------------
# T distribution plot
# -------------------------------------------------------------------
t_all = np.concatenate([t_calib, t_test]) if len(t_test) else t_calib
xmin = args.xmin if args.xmin is not None else max(0.0, t_all.min() * 0.9)
xmax = args.xmax if args.xmax is not None else t_all.max() * 1.05

nbins = min(fd_bins(t_calib), 60)
bins  = np.linspace(xmin, xmax, nbins + 1)
bin_w = bins[1] - bins[0]
x_chi2 = np.linspace(xmin, xmax, 400)

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Calibration (null)
ax.hist(t_calib, bins=bins, density=True, color="#b0c4de", alpha=0.85,
        edgecolor="steelblue", linewidth=0.6, label=rf"Null ($H_0$, n={len(t_calib)})")

# Chi2 overlays
ax.plot(x_chi2, chi2.pdf(x_chi2, args.dof), "b--", lw=1.4,
        label=rf"$\chi^2({args.dof})$")
ax.plot(x_chi2, chi2.pdf(x_chi2, DOF_eff), "b-", lw=1.8,
        label=rf"$\chi^2({DOF_eff:.1f})$ fitted")

# Test (signal) if present
if len(t_test):
    ax.hist(t_test, bins=bins, density=True, color="#f4a460", alpha=0.75,
            edgecolor="#cc6600", linewidth=0.6,
            label=rf"Test ($H_1$ target, n={len(t_test)})")
    ax.axvline(np.median(t_test), color="#cc6600", lw=2, ls="--",
               label=rf"median test = {np.median(t_test):.1f}")

ax.axvline(np.median(t_calib), color="steelblue", lw=1.5, ls=":",
           label=rf"median null = {np.median(t_calib):.1f}")

if args.ymax:
    ax.set_ylim(0, args.ymax)
ax.set_xlim(xmin, xmax)
ax.set_xlabel(label, fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(title, fontsize=11)
ax.legend(fontsize=9, frameon=True, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "T_distribution.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_dir}/T_distribution.png")

# -------------------------------------------------------------------
# Weight shift diagnostics (if den_weights exist)
# -------------------------------------------------------------------
def collect_weight_deltas(base_dir, mode):
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, mode, "seed*")))
    deltas = []
    for sd in seed_dirs:
        name = os.path.basename(sd)
        f_den  = os.path.join(sd, f"{name}_den_weights.npy")
        f_init = os.path.join(sd, f"{name}_init_weights.npy")
        if os.path.exists(f_den) and os.path.exists(f_init):
            deltas.append(np.load(f_den).ravel() - np.load(f_init).ravel())
    return np.array(deltas) if deltas else None

dw_calib = collect_weight_deltas(results_dir, "calibration")
dw_test  = collect_weight_deltas(results_dir, "test")

if dw_calib is not None and dw_calib.shape[0] > 0:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    n_w = dw_calib.shape[1]
    idx = np.arange(n_w)
    ax.fill_between(idx,
                    np.percentile(dw_calib, 16, axis=0),
                    np.percentile(dw_calib, 84, axis=0),
                    alpha=0.35, color="steelblue", label="null 68% band")
    ax.plot(idx, np.median(dw_calib, axis=0), color="steelblue", lw=1.5, label="null median")
    if dw_test is not None and dw_test.shape[0] > 0:
        ax.fill_between(idx,
                        np.percentile(dw_test, 16, axis=0),
                        np.percentile(dw_test, 84, axis=0),
                        alpha=0.35, color="#f4a460", label="test 68% band")
        ax.plot(idx, np.median(dw_test, axis=0), color="#cc6600", lw=1.5, label="test median")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("WiFi weight index", fontsize=11)
    ax.set_ylabel(r"$\Delta w = w_\mathrm{den} - w_\mathrm{init}$", fontsize=11)
    ax.set_title(f"{title} — DEN weight shifts", fontsize=10)
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_shifts.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/weight_shifts.png")

# -------------------------------------------------------------------
# Clip saturation (kernels only — skipped if no coeffs.npy)
# -------------------------------------------------------------------
def collect_n_at_clip(base_dir, mode, clip_tau):
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, mode, "seed*")))
    counts = []
    for sd in seed_dirs:
        name = os.path.basename(sd)
        f = os.path.join(sd, f"{name}_coeffs.npy")
        if os.path.exists(f):
            coeffs = np.load(f).ravel()
            counts.append(int(np.sum(np.abs(coeffs) >= clip_tau * 0.999)))
    return np.array(counts) if counts else None

n_clip = collect_n_at_clip(results_dir, "calibration", args.clip_tau)

if n_clip is not None and len(n_clip) > 0:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bins_c = np.arange(n_clip.min() - 0.5, n_clip.max() + 1.5, 1)
    ax.hist(n_clip, bins=bins_c, color="#e186ed", alpha=0.7,
            edgecolor="#8a2be2", linewidth=0.8)
    ax.axvline(n_clip.mean(), color="#8a2be2", lw=2, ls="--",
               label=rf"mean $= {n_clip.mean():.1f}$")
    ax.axvline(args.dof - DOF_eff, color="steelblue", lw=2, ls=":",
               label=rf"$M - \mathrm{{DOF}}_\mathrm{{eff}} = {args.dof - DOF_eff:.1f}$")
    ax.set_xlabel(r"$n_\mathrm{at\_clip}$ per toy", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{title} — kernel clip saturation (null)", fontsize=10)
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "n_at_clip.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/n_at_clip.png")

print("\nDone.")
