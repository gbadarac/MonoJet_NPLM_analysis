"""
plot_Tdist_vs_ntest.py
----------------------
Grid of LRT test-statistic (2t) distributions: null (calibration) vs observed
(test), one row per null mode (composite / point) and one column per N_test.

A companion to plot_pvalue_vs_ntest.py — SAME discover -> parse -> plot pattern
over the run-tag dirs LRT.py writes, but instead of collapsing each run to a
single p-value it shows the FULL 2t distributions side by side, so you can watch
the observed statistic walk away from the null as N_test grows. This is a NEW,
standalone plot:
  * LRT.py is UNCHANGED — this reads only the seed*_T.npy files it already writes.
  * analyse_LRT_output.py is UNCHANGED — that stays the single-run diagnostic
    (one N_test, one mode); this is the cross-N_test overview.

Layout (mirrors the reference multi-panel figure):
    rows    = null modes present, composite (top) then point (bottom)
    columns = N_test, ascending left -> right
    each cell:
        * null toys (calibration) : filled histogram, coloured by mode
        * observed (test)         : black step histogram
        * chi2 reference          : dashed; fitted chi2(DOF_eff) per cell, or the
                                    nominal chi2(--dof) if given (fixed across cells)
        * title: "<mode> . N_test=<n> . null <mean>+-<std> . obs z <score>"

The per-cell "obs z" is a STANDARDISED score, (median observed - mean null) / std
null. It is NOT the empirical-tail Z used by analyse_LRT_output.py: that one
saturates at ~1/(B+1) (Z_max ~ 2.3 for B=100 null toys), so it cannot resolve the
strongly-separated large-N_test cells. The standardised score keeps resolving
them (and matches how the reference figure reports "obs z").

Usage:
  python plot_Tdist_vs_ntest.py \
     --base LRT/results/kernels/2d_ensemble \
     [--nens 128] [--dof 64] [--min_ntest 2000] \
     [--out LRT/results/kernels/2d_ensemble/figs/Tdist_grid_vs_ntest.pdf] \
     [--title "2d_gmm kernels"]
"""
import os, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import minimize_scalar

# Row order + colours. Group convention (see plot_pvalue_vs_ntest.py): constrained
# w = COMPOSITE null (weight nuisances profiled); frozen w = POINT null (plug-in
# at w_hat). The two fills are the SAME pink/blue analyse_LRT_output.py uses
# (blue #68aedc = its DATA hist, pink #e186ed = its REF/null hist); here we map
# composite = blue (keeping the composite=blue convention of plot_pvalue_vs_ntest.py)
# and point = pink.
MODE_ORDER = ["composite", "point"]
MODE_FILL  = {"composite": "#68aedc", "point": "#e186ed"}
OBS_INK    = "#2b2b2b"
CHI2_INK   = "#7f7f7f"


def collect_T(run_dir, mode):
    """Load all finite seed*_T.npy under run_dir/<mode>/seed*/ into a 1-D array."""
    out = []
    for sd in sorted(glob.glob(os.path.join(run_dir, mode, "seed*"))):
        f = os.path.join(sd, os.path.basename(sd) + "_T.npy")
        if os.path.exists(f):
            v = float(np.load(f))
            if np.isfinite(v):
                out.append(v)
    return np.array(out, dtype=float)


def fit_dof_eff(t_calib):
    """Fit chi2(DOF_eff) to the null via quantile matching (as analyse_LRT_output.py)."""
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    eq = np.quantile(t_calib, qs)
    return float(minimize_scalar(lambda d: np.sum((eq - chi2.ppf(qs, d)) ** 2),
                                 bounds=(5, 500), method="bounded").x)


def parse_tag(name):
    """Parse (Nens, Ntest, mode) from a run-tag folder name; None if it isn't one.
    Identical rules to plot_pvalue_vs_ntest.parse_tag (incl. legacy names)."""
    m_n = re.search(r"Nens(\d+)", name)
    m_t = re.search(r"Ntest(\d+)", name)
    if not (m_n and m_t):
        return None
    name = name.rstrip("/")
    if name.endswith("composite_null") or name.endswith("constrained"):
        mode = "composite"
    elif "point_null" in name or "frozen_weights" in name:
        mode = "point"
    else:
        return None
    return int(m_n.group(1)), int(m_t.group(1)), mode


def fmt_ntest(n):
    """2000 -> '2k', 100000 -> '100k', 100 -> '100'."""
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def cell_edges(t_null, t_obs, cap=55, floor=25):
    """Shared bin edges over the combined null+observed range of ONE cell.

    Unlike analyse_LRT_output.py (which bins null and test separately), this plot
    deliberately puts both on the SAME edges per cell: the whole point is to see
    the observed distribution drift off the null on a common axis."""
    x = np.concatenate([t_null, t_obs]) if len(t_obs) else t_null
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.linspace(lo, lo + 1.0, floor + 1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2.0 * iqr / (len(x) ** (1 / 3) + 1e-12)
    nb = max(floor, min(cap, int(np.ceil((hi - lo) / h)))) if h > 0 else floor
    return np.linspace(lo, hi, nb + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True,
                    help="One or more dirs whose immediate subdirs are run-tag dirs.")
    ap.add_argument("--nens", type=int, default=None,
                    help="Ensemble size to plot (one grid = one Nens). Default: the "
                         "Nens with the most runs found.")
    ap.add_argument("--dof", type=int, default=None,
                    help="Nominal chi2 DOF for a FIXED reference curve across all cells "
                         "(like the reference figure). Default: fitted chi2(DOF_eff) "
                         "per cell.")
    ap.add_argument("--ntest", nargs="+", type=int, default=None,
                    help="Only show these N_test columns (in the given order). Default: "
                         "every N_test found on disk, ascending. Use this to pick a clean "
                         "subset when the full sweep has too many columns, e.g. "
                         "--ntest 200 2000 20000 200000.")
    ap.add_argument("--min_ntest", type=int, default=0,
                    help="Skip runs with Ntest below this (drops debug runs).")
    ap.add_argument("--min_toys", type=int, default=5,
                    help="Skip a cell if it has fewer than this many null OR test toys.")
    ap.add_argument("--out", default=None,
                    help="Output path (.pdf; a .png sibling is also written). "
                         "Default: <first base>/figs/Tdist_grid_nens<Nens>_vs_ntest.pdf "
                         "(the Nens is in the name so per-ensemble grids don't overwrite "
                         "each other).")
    ap.add_argument("--title", default=None, help="Overall figure title override.")
    args = ap.parse_args()

    # ---- discover runs: (nens, mode, ntest) -> run_dir ----------------------
    found = {}                                   # (nens, mode, ntest) -> path
    nens_counts = {}
    for base in args.base:
        for run_dir in sorted(glob.glob(os.path.join(base, "*"))):
            if not os.path.isdir(run_dir):
                continue
            tag = parse_tag(os.path.basename(run_dir))
            if tag is None:
                continue
            nens, ntest, mode = tag
            if ntest < args.min_ntest:
                continue
            found[(nens, mode, ntest)] = run_dir
            nens_counts[nens] = nens_counts.get(nens, 0) + 1

    if not found:
        raise SystemExit("No usable run-tag dirs found under --base.")

    nens = args.nens if args.nens is not None else max(nens_counts, key=nens_counts.get)
    print(f"Ensemble size (Nens) : {nens}   "
          f"(available: {sorted(nens_counts)})")

    # rows = modes present for this Nens (composite top, point bottom); cols = Ntest
    modes = [m for m in MODE_ORDER
             if any(k[0] == nens and k[1] == m for k in found)]
    ntests = sorted({k[2] for k in found if k[0] == nens})
    if args.ntest is not None:                    # keep only requested, in given order
        have = set(ntests)
        missing = [n for n in args.ntest if n not in have]
        if missing:
            print(f"  [warn] requested N_test not found for Nens={nens}: {missing}")
        ntests = [n for n in args.ntest if n in have]
    if not modes or not ntests:
        raise SystemExit(f"No runs for Nens={nens}.")
    print(f"Rows (modes)         : {modes}")
    print(f"Cols (N_test)        : {ntests}")

    nrows, ncols = len(modes), len(ntests)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.3 * nrows),
                             squeeze=False)

    for r, mode in enumerate(modes):
        for c, ntest in enumerate(ntests):
            ax = axes[r][c]
            run_dir = found.get((nens, mode, ntest))
            if run_dir is None:
                ax.text(0.5, 0.5, "no run", ha="center", va="center",
                        transform=ax.transAxes, color=CHI2_INK, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            t_null = collect_T(run_dir, "calibration")
            t_obs  = collect_T(run_dir, "test")
            if len(t_null) < args.min_toys or len(t_obs) < args.min_toys:
                ax.text(0.5, 0.5, f"too few toys\n(null {len(t_null)}, obs {len(t_obs)})",
                        ha="center", va="center", transform=ax.transAxes,
                        color=CHI2_INK, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            mean_n, std_n = float(np.mean(t_null)), float(np.std(t_null))
            obs_z = (float(np.median(t_obs)) - mean_n) / std_n if std_n > 0 else np.nan

            edges = cell_edges(t_null, t_obs)
            ax.hist(t_null, bins=edges, density=True, color=MODE_FILL[mode],
                    alpha=0.55, edgecolor="none",
                    label=f"null toys (n={len(t_null):,})")
            ax.hist(t_obs, bins=edges, density=True, histtype="step",
                    color=OBS_INK, lw=1.4, label=f"observed (n={len(t_obs):,})")

            dof = args.dof if args.dof is not None else fit_dof_eff(t_null)
            xs = np.linspace(max(0.1, edges[0]), edges[-1], 400)
            ax.plot(xs, chi2.pdf(xs, dof), ls="--", color=CHI2_INK, lw=1.3,
                    label=rf"$\chi^2_{{{dof:.0f}}}$")

            ax.set_title(rf"{mode} $\cdot$ $N_{{\rm test}}$={fmt_ntest(ntest)} $\cdot$ "
                         rf"null {mean_n:.1f}$\pm${std_n:.1f} $\cdot$ obs z {obs_z:+.2f}",
                         fontsize=9)
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")
            ax.tick_params(labelsize=8)
            if r == nrows - 1:
                ax.set_xlabel(r"$2t$", fontsize=12)
            if c == 0:
                ax.set_ylabel("probability density", fontsize=10)
            print(f"  {mode:9s} Ntest={ntest:<7d}  null {mean_n:.1f}+-{std_n:.1f}  "
                  f"obs z {obs_z:+.2f}  (n_null={len(t_null)}, n_obs={len(t_obs)})")

    dim = None
    for base in args.base:
        m = re.search(r"(\d+)d", os.path.basename(base.rstrip("/")))
        if m:
            dim = f"{int(m.group(1))}D"
            break
    ens_str  = "single model" if nens == 1 else f"{nens}-component ensemble"
    suptitle = args.title or (
        f"{dim + ' ' if dim else ''}{ens_str} $\\cdot$ "
        r"null (calibration) vs observed (test) $2t$ across $N_{\rm test}$")
    fig.suptitle(suptitle, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or os.path.join(args.base[0], "figs",
                                   f"Tdist_grid_nens{nens}_vs_ntest.pdf")
    out_base = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_base + ".png", "and", out_base + ".pdf")


if __name__ == "__main__":
    main()
