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
    <out_dir>/T_distribution.{png,pdf}   — null + test T distributions vs chi2,
                                           two-panel layout with Z and p in side panel
    <out_dir>/chi2_quantile_table.txt    — empirical vs chi2 quantile comparison
    <out_dir>/weight_shifts.{png,pdf}    — Δw = w_den - w_init per weight component
    <out_dir>/n_at_clip.{png,pdf}        — kernel clip saturation histogram (if coeffs exist)

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
import matplotlib.font_manager as font_manager
from matplotlib.patches import Rectangle
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
parser.add_argument("--w_cov_path", type=str, default=None,
                    help="Path to w_cov .npy file (M-1 x M-1). If given, adds "
                         "normalized weight-pull plot Δw/sqrt(diag Σ_w).")
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
    """One-sided empirical Z with exact Clopper-Pearson interval on p."""
    T_null = np.asarray(T_null, float)
    B = T_null.size
    k = int(np.count_nonzero(T_null >= T_obs))
    p_hat = (k + 1) / (B + 1)
    if k == 0:
        p_lo = 0.0
        p_hi = beta.ppf(1 - alpha / 2, 1, B)
    elif k == B:
        p_lo = beta.ppf(alpha / 2, B, 1)
        p_hi = 1.0
    else:
        p_lo = beta.ppf(alpha / 2,     k,     B - k + 1)
        p_hi = beta.ppf(1 - alpha / 2, k + 1, B - k)
    Z      = norm.ppf(1 - p_hat)
    Z_plus  = norm.ppf(1 - max(1e-16, p_lo)) - Z
    Z_minus = Z - norm.ppf(1 - min(1 - 1e-16, p_hi))
    return Z, Z_plus, Z_minus, p_hat


def fit_dof_eff(t_calib):
    """Fit chi2(DOF_eff) to calibration T via quantile matching."""
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    eq = np.quantile(t_calib, qs)
    def res(dof):
        return np.sum((eq - chi2.ppf(qs, dof)) ** 2)
    return minimize_scalar(res, bounds=(5, 500), method="bounded").x


def fd_bins(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2 * iqr / (len(x) ** (1 / 3) + 1e-12)
    if h <= 0:
        return 10
    return max(5, int(np.ceil((x.max() - x.min()) / h)))


def save_fig(fig, out_dir, name):
    """Save figure as both high-DPI PNG (1200 dpi) and PDF."""
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(base + ".png", dpi=1200, bbox_inches="tight",
                pad_inches=0, facecolor="white")
    print(f"Saved: {base}.pdf")
    print(f"Saved: {base}.png")


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
    Z, Zp, Zm, p = empirical_Z(t_calib, np.median(t_test))
    print(f"Empirical Z (median test vs calib null): {Z:.3f} +{Zp:.3f}/-{Zm:.3f}  p={p:.4f}")

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
# T distribution plot — two-panel layout matching old notebooks
# -------------------------------------------------------------------
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams['patch.linewidth'] = 0.0

t_all = np.concatenate([t_calib, t_test]) if len(t_test) else t_calib
xmin = args.xmin
xmax = args.xmax
if xmin is None or xmax is None:
    dmin, dmax = t_all.min(), t_all.max()
    span = dmax - dmin
    pad  = 0.05 * span if span > 0 else 1.0
    if xmin is None:
        xmin = max(0.0, dmin - pad)
    if xmax is None:
        xmax = dmax + pad

nbins = min(fd_bins(t_calib), 60)
bins  = np.linspace(xmin, xmax, nbins + 1)
bin_w = bins[1] - bins[0]
xcenters = 0.5 * (bins[1:] + bins[:-1])

fig = plt.figure(figsize=(12, 9))
fig.patch.set_facecolor("white")
ax  = fig.add_axes([0.10, 0.12, 0.62, 0.78])
axp = fig.add_axes([0.76, 0.12, 0.20, 0.78])
axp.axis("off")

# --- calibration histogram (null) ---
h1 = ax.hist(
    t_calib,
    weights=np.ones_like(t_calib) / (len(t_calib) * bin_w),
    color="#e186ed", alpha=0.5, bins=bins,
    label=r"REF (calibration, $H_0$)",
    edgecolor="none", linewidth=0,
)
err1 = np.sqrt(h1[0] / (len(t_calib) * bin_w))
ax.errorbar(xcenters, h1[0], yerr=err1,
            color="#8a2be2", marker="o", ls="", alpha=0.6,
            markersize=5, capsize=2, elinewidth=0.8)

# --- chi2 overlays (nominal dashed, fitted solid) ---
x_chi2 = np.linspace(max(0.1, xmin), xmax, 500)
ax.plot(x_chi2, chi2.pdf(x_chi2, args.dof), "k--", lw=1.4,
        label=rf"$\chi^2({args.dof})$ nominal")
ax.plot(x_chi2, chi2.pdf(x_chi2, DOF_eff), "k-", lw=1.8,
        label=rf"$\chi^2({DOF_eff:.1f})$ fitted")

# --- test histogram ---
p_emp = Z_val = Z_p = Z_m = np.nan
if len(t_test):
    h2 = ax.hist(
        t_test,
        weights=np.ones_like(t_test) / (len(t_test) * bin_w),
        color="#68aedc", alpha=0.5, bins=bins,
        label=r"DATA (test, target)",
        edgecolor="none", linewidth=0,
    )
    err2 = np.sqrt(h2[0] / (len(t_test) * bin_w))
    ax.errorbar(xcenters, h2[0], yerr=err2,
                color="#004c99", marker="o", ls="", alpha=0.6,
                markersize=5, capsize=2, elinewidth=0.8)
    T_obs = float(np.median(t_test))
    Z_val, Z_p, Z_m, p_emp = empirical_Z(t_calib, T_obs)

# --- side panel ---
panel_fp = font_manager.FontProperties(family="serif", size=20)
SW, GAP = 0.032, 0.012

def _header(y, color_hex, text):
    axp.add_patch(Rectangle((0.03, y - SW / 2), SW, SW,
                             transform=axp.transAxes,
                             facecolor=color_hex, edgecolor="none", alpha=0.5))
    axp.text(0.03 + SW + GAP, y, text,
             va="center", ha="left", fontproperties=panel_fp, color="black")

axp.text(0.03, 0.98, rf"$N_{{\mathrm{{calib}}}} = {len(t_calib)}$",
         va="top", ha="left", fontproperties=panel_fp)
if len(t_test):
    axp.text(0.03, 0.92, rf"$N_{{\mathrm{{test}}}} = {len(t_test)}$",
             va="top", ha="left", fontproperties=panel_fp)

Y_REF_HDR = 0.82
_header(Y_REF_HDR, "#e186ed", r"REF ($H_0$)")
axp.text(0.03 + SW + GAP, Y_REF_HDR - 0.065,
         rf"median $= {np.median(t_calib):.1f}$" + "\n"
         + rf"std $= {np.std(t_calib):.1f}$",
         va="top", ha="left", fontproperties=panel_fp)

if len(t_test):
    Y_DATA_HDR = 0.60
    _header(Y_DATA_HDR, "#68aedc", "DATA (target)")
    axp.text(0.03 + SW + GAP, Y_DATA_HDR - 0.065,
             rf"median $= {np.median(t_test):.1f}$" + "\n"
             + rf"std $= {np.std(t_test):.1f}$",
             va="top", ha="left", fontproperties=panel_fp)
    axp.text(0.03 + SW + GAP, Y_DATA_HDR - 0.195,
             rf"p-value $= {p_emp:.4f}$" + "\n"
             + rf"emp. $Z = {Z_val:.2f}\,^{{+{Z_p:.2f}}}_{{-{Z_m:.2f}}}$",
             va="top", ha="left", fontproperties=panel_fp)

legend_fp = font_manager.FontProperties(family="serif", size=16.5)
ax.legend(ncol=1, loc="upper right", prop=legend_fp, frameon=False,
          handlelength=1.8, borderpad=0.3, labelspacing=0.3)
xlabel = r"$2t$" if args.scale_t == 2.0 else r"$t$"
ax.set_xlabel(xlabel, fontsize=32, fontname="serif")
ax.set_ylabel("Probability density", fontsize=28, fontname="serif")
ax.set_xlim(xmin, xmax)
if args.ymax:
    ax.set_ylim(0, args.ymax)
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
if title:
    ax.set_title(title, fontsize=24, fontname="serif", pad=14)

save_fig(fig, out_dir, "T_distribution")
plt.close(fig)

# -------------------------------------------------------------------
# Weight shift diagnostics (den and num vs init)
# -------------------------------------------------------------------
def collect_weight_arrays(base_dir, mode):
    """Return (dw_den, dw_num) arrays of shape (n_seeds, n_weights), or None."""
    seed_dirs = sorted(glob.glob(os.path.join(base_dir, mode, "seed*")))
    dw_den, dw_num = [], []
    for sd in seed_dirs:
        name = os.path.basename(sd)
        f_init = os.path.join(sd, f"{name}_init_weights.npy")
        f_den  = os.path.join(sd, f"{name}_den_weights.npy")
        f_num  = os.path.join(sd, f"{name}_num_weights.npy")
        if os.path.exists(f_init) and os.path.exists(f_den):
            w_init = np.load(f_init).ravel()
            dw_den.append(np.load(f_den).ravel() - w_init)
            if os.path.exists(f_num):
                dw_num.append(np.load(f_num).ravel() - w_init)
    dw_den = np.array(dw_den) if dw_den else None
    dw_num = np.array(dw_num) if dw_num else None
    return dw_den, dw_num

dw_den_calib, dw_num_calib = collect_weight_arrays(results_dir, "calibration")
dw_den_test,  dw_num_test  = collect_weight_arrays(results_dir, "test")

if dw_den_calib is not None and dw_den_calib.shape[0] > 0:
    n_w = dw_den_calib.shape[1]
    idx = np.arange(n_w)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_w), 5), sharey=False)
    fig.patch.set_facecolor("white")

    for ax, (dw_calib_arr, dw_test_arr), ylabel, panel_title in zip(
        axes,
        [(dw_den_calib, dw_den_test), (dw_num_calib, dw_num_test)],
        [r"$\Delta w = w_\mathrm{den} - w_\mathrm{init}$",
         r"$\Delta w = w_\mathrm{num} - w_\mathrm{init}$"],
        ["DEN weight shifts", "NUM weight shifts"],
    ):
        ax.set_facecolor("white")
        if dw_calib_arr is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        ax.fill_between(idx,
                        np.percentile(dw_calib_arr, 16, axis=0),
                        np.percentile(dw_calib_arr, 84, axis=0),
                        alpha=0.35, color="#e186ed", label=r"null 68% band")
        ax.plot(idx, np.median(dw_calib_arr, axis=0), color="#8a2be2", lw=1.5,
                label="null median")
        if dw_test_arr is not None and dw_test_arr.shape[0] > 0:
            ax.fill_between(idx,
                            np.percentile(dw_test_arr, 16, axis=0),
                            np.percentile(dw_test_arr, 84, axis=0),
                            alpha=0.35, color="#68aedc", label="test 68% band")
            ax.plot(idx, np.median(dw_test_arr, axis=0), color="#004c99", lw=1.5,
                    label="test median")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("WiFi weight index", fontsize=14, fontname="serif")
        ax.set_ylabel(ylabel, fontsize=14, fontname="serif")
        ax.set_title(f"{title} — {panel_title}", fontsize=12, fontname="serif")
        ax.legend(frameon=False,
                  prop=font_manager.FontProperties(family="serif", size=11))

    fig.tight_layout()
    save_fig(fig, out_dir, "weight_shifts")
    plt.close(fig)

# -------------------------------------------------------------------
# Normalised weight pulls Δw / sqrt(diag Σ_w)  (optional, needs --w_cov_path)
# -------------------------------------------------------------------
if args.w_cov_path and os.path.exists(args.w_cov_path):
    Sigma_w = np.load(args.w_cov_path)
    if Sigma_w.ndim == 2:
        sigma_w = np.sqrt(np.diag(Sigma_w).clip(min=1e-30))
    else:
        sigma_w = np.sqrt(Sigma_w.clip(min=1e-30))

    if dw_den_calib is not None and dw_den_calib.shape[1] == sigma_w.shape[0]:
        n_w = dw_den_calib.shape[1]
        idx = np.arange(n_w)

        fig, axes = plt.subplots(1, 2, figsize=(max(14, n_w), 5), sharey=True)
        fig.patch.set_facecolor("white")

        for ax, (dw_calib_arr, dw_test_arr), panel_title in zip(
            axes,
            [(dw_den_calib, dw_den_test), (dw_num_calib, dw_num_test)],
            ["DEN pulls", "NUM pulls"],
        ):
            ax.set_facecolor("white")
            if dw_calib_arr is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            pulls_calib = dw_calib_arr / sigma_w[np.newaxis, :]
            ax.fill_between(idx,
                            np.percentile(pulls_calib, 16, axis=0),
                            np.percentile(pulls_calib, 84, axis=0),
                            alpha=0.35, color="#e186ed", label=r"null 68% band")
            ax.plot(idx, np.median(pulls_calib, axis=0), color="#8a2be2", lw=1.5,
                    label="null median")

            if dw_test_arr is not None and dw_test_arr.shape[0] > 0:
                pulls_test = dw_test_arr / sigma_w[np.newaxis, :]
                ax.fill_between(idx,
                                np.percentile(pulls_test, 16, axis=0),
                                np.percentile(pulls_test, 84, axis=0),
                                alpha=0.35, color="#68aedc", label="test 68% band")
                ax.plot(idx, np.median(pulls_test, axis=0), color="#004c99", lw=1.5,
                        label="test median")

            for level in [1.0, -1.0, 2.0, -2.0]:
                ax.axhline(level, color="gray",
                           lw=1.2 if abs(level) == 1 else 0.6,
                           ls="--" if abs(level) == 1 else ":",
                           alpha=0.7)
            ax.axhline(0, color="k", lw=0.8, ls="-")
            ax.set_xlabel("WiFi weight index", fontsize=14, fontname="serif")
            ax.set_ylabel(r"$\Delta w\,/\,\sqrt{\mathrm{diag}\,\Sigma_w}$",
                          fontsize=14, fontname="serif")
            ax.set_title(f"{title} — {panel_title}", fontsize=12, fontname="serif")
            ax.legend(frameon=False,
                      prop=font_manager.FontProperties(family="serif", size=11))

        fig.tight_layout()
        save_fig(fig, out_dir, "weight_pulls")
        plt.close(fig)
    else:
        print(f"  Skipping pulls plot: weight dim {dw_den_calib.shape[1] if dw_den_calib is not None else 'N/A'}"
              f" ≠ sigma_w dim {sigma_w.shape[0]}")
elif args.w_cov_path:
    print(f"  w_cov_path not found: {args.w_cov_path}")

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
            counts.append(int(np.sum(np.abs(coeffs) >= args.clip_tau * 0.999)))
    return np.array(counts) if counts else None

n_clip = collect_n_at_clip(results_dir, "calibration", args.clip_tau)

if n_clip is not None and len(n_clip) > 0:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bins_c = np.arange(n_clip.min() - 0.5, n_clip.max() + 1.5, 1)
    ax.hist(n_clip, bins=bins_c, color="#e186ed", alpha=0.7,
            edgecolor="#8a2be2", linewidth=0.8)
    ax.axvline(n_clip.mean(), color="#8a2be2", lw=2, ls="--",
               label=rf"mean $= {n_clip.mean():.1f}$")
    ax.axvline(args.dof - DOF_eff, color="steelblue", lw=2, ls=":",
               label=rf"$M - \mathrm{{DOF}}_\mathrm{{eff}} = {args.dof - DOF_eff:.1f}$")
    ax.set_xlabel(r"$n_\mathrm{at\_clip}$ per toy", fontsize=14, fontname="serif")
    ax.set_ylabel("Toys", fontsize=14, fontname="serif")
    ax.set_title(rf"{title} — kernel clip saturation (null, clip$={args.clip_tau}$)",
                 fontsize=12, fontname="serif")
    legend_fp = font_manager.FontProperties(family="serif", size=12)
    ax.legend(prop=legend_fp, frameon=False)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir, "n_at_clip")
    plt.close(fig)

print("\nDone.")
