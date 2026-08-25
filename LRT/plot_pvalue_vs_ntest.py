"""
plot_pvalue_vs_ntest.py
-----------------------
NPLM p-value vs N_test — frozen vs constrained (and single) — in the reference
group-standard style: a median p-value line per configuration with a single
light IQR (25-75%) toy-spread band and bootstrap 95% CI error bars on the
median, plus the nominal (p=0.5) and bank-floor (p=1/(B+1)) reference lines.

Design (mirrors the reference repo's per-run -> merge -> plot split):
  * LRT.py is UNCHANGED — this reads only the seed*_T.npy files it already writes.
  * analyse_LRT_output.py is UNCHANGED — that stays the single-run diagnostic;
    this is the cross-run scan aggregation + plot.

What it does:
  1. discovers run-tag dirs under the given --base dir(s),
  2. parses (Nens, Ntest, mode = frozen | constrained) from each folder name,
  3. per run computes, for each TEST toy i, the empirical p-value against the
     NULL toys:  p_i = (#{T_null >= T_test,i} + 1) / (B + 1)
     (identical estimator to analyse_LRT_output.py, just per-toy instead of the
     single median), then the {25, 50, 75} percentiles of {p_i} plus a
     bootstrap 95% CI of the median,
  4. plots one curve per (Nens, mode): median + a single light IQR band +
     median CI error bars; distinct markers (constrained=o, frozen=s,
     single=triangle); log x-axis; nominal p=0.5 and bank-floor reference lines.

NOTE on the median line: because the empirical p-value is a monotonic function
of T_obs, the median of the per-toy p-values equals the p-value of the median
test T — i.e. this p50 line is exactly the `p-value` number analyse_LRT_output.py
already prints per run. The bands are the extra piece (they need the full per-toy
p-value distribution, which the single-run script does not emit).

Caveat: the test toys are BOOTSTRAP resamples of one fixed target, so these bands
are a bootstrap spread (correlated draws) — an approximation to the true
experiment-to-experiment spread. Label them as such.

Usage:
  python plot_pvalue_vs_ntest.py \
     --base LRT/results/kernels/2d_ensemble LRT/results/kernels/2d_single_model \
     --out  LRT/results/figs/pvalue_vs_ntest.pdf \
     [--csv LRT/results/figs/pvalue_vs_ntest.csv] [--title "2d_gmm kernels"]
"""
import os, glob, re, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One light IQR band (25-75) + median + bootstrap CI of the median. No nested bands.
QS = [25.0, 50.0, 75.0]
N_BOOT = 2000
RNG = np.random.default_rng(20260825)

# Axis furniture (spines, reference lines, annotations) stays muted grey; the
# three data curves are blue / red / green (constrained / frozen / single).
INK, GRID, MUTED = "#33322e", "#c9c8c0", "#8a897f"
BLUE, RED, GREEN = "#2a78d6", "#d62728", "#2ca02c"
# mode -> (color, marker, legend label)
MODE_STYLE = {
    "constrained": (BLUE, "o", "constrained (uncertainty propagated)"),
    "frozen":      (RED,  "s", "frozen (fixed weights)"),
}


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


def pvalue_percentiles(run_dir, min_toys=5):
    """Return (q25, med, q75, ci_lo, ci_hi, n_test, B) or None if insufficient.

    ci_lo/ci_hi are a bootstrap 95% CI of the median p-value, resampling the
    test toys (an approximate spread — the test toys are correlated bootstrap
    resamples of one fixed target; see the module docstring)."""
    tc = collect_T(run_dir, "calibration")
    tt = collect_T(run_dir, "test")
    if len(tc) < min_toys or len(tt) < min_toys:
        return None
    B = len(tc)
    # per test toy: k_i = #{null >= T_test,i};  p_i = (k_i + 1)/(B + 1)
    k = (tc[None, :] >= tt[:, None]).sum(axis=1)          # (n_test,)
    p = (k + 1.0) / (B + 1.0)
    q25, med, q75 = np.percentile(p, QS)
    nt = len(p)
    boots = np.median(p[RNG.integers(0, nt, (N_BOOT, nt))], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return q25, med, q75, lo, hi, nt, B


def parse_tag(name):
    """Parse (Nens, Ntest, mode) from a run-tag folder name; None if it isn't one."""
    m_n = re.search(r"Nens(\d+)", name)
    m_t = re.search(r"Ntest(\d+)", name)
    if not (m_n and m_t):
        return None
    name = name.rstrip("/")
    if name.endswith("constrained"):
        mode = "constrained"
    elif "frozen_weights" in name:
        mode = "frozen"
    else:
        return None
    return int(m_n.group(1)), int(m_t.group(1)), mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True,
                    help="One or more dirs whose immediate subdirs are run-tag dirs "
                         "(e.g. .../2d_ensemble .../2d_single_model).")
    ap.add_argument("--out", default=None,
                    help="Output path. Default: <first --base>/figs/pvalue_vs_ntest.pdf "
                         "(figures live next to the runs they summarize).")
    ap.add_argument("--csv", default=None, help="Optional summary.csv dump.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--alpha_level", type=float, default=0.05)
    ap.add_argument("--nens", nargs="+", type=int, default=None,
                    help="Only include these ensemble sizes (the single model, Nens=1, is "
                         "always kept as the baseline). Default: all sizes found.")
    ap.add_argument("--min_ntest", type=int, default=0,
                    help="Skip runs with Ntest below this — drops off-scan/debug runs "
                         "(e.g. the Ntest=1000/M40 test).")
    args = ap.parse_args()
    if args.out is None:                 # figs/ default lives inside the primary base dir
        args.out = os.path.join(args.base[0], "figs", "pvalue_vs_ntest.pdf")

    # (Nens, mode) -> list of (Ntest, q25, med, q75, ci_lo, ci_hi)
    rows = {}
    Bs = []
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
            if args.nens is not None and nens != 1 and nens not in set(args.nens):
                continue
            res = pvalue_percentiles(run_dir)
            if res is None:
                continue
            q25, med, q75, lo, hi, n_test, B = res
            rows.setdefault((nens, mode), []).append((ntest, q25, med, q75, lo, hi))
            Bs.append(B)
            print(f"{mode:11s} Nens={nens:<4d} Ntest={ntest:<7d}  "
                  f"med={med:.3f}  IQR=[{q25:.3f},{q75:.3f}]  "
                  f"CI=[{lo:.3f},{hi:.3f}]  (n_test={n_test}, B={B})")

    if not rows:
        raise SystemExit("No usable runs found (need calibration/ + test/ T.npy under --base).")

    out_base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # --- ensemble curves (constrained / frozen): IQR band + median + boot CI ---
    for (nens, mode), pts in sorted(rows.items()):
        if nens == 1 or mode not in MODE_STYLE:
            continue
        pts = np.array(sorted(pts))                        # sort by Ntest
        x, q25, med, q75, lo, hi = (pts[:, 0], pts[:, 1], pts[:, 2],
                                    pts[:, 3], pts[:, 4], pts[:, 5])
        color, mk, lab = MODE_STYLE[mode]
        if len(x) > 1:
            ax.fill_between(x, q25, q75, color=color, alpha=0.16, lw=0)
        ax.plot(x, med, color=color, lw=2.3, marker=mk, ms=6.5, label=lab, zorder=3)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none",
                    ecolor=color, elinewidth=1.2, capsize=3, zorder=4)

    # --- single-model baseline: muted dashed line + its own light IQR band ---
    for (nens, mode), pts in sorted(rows.items()):
        if nens != 1:
            continue
        pts = np.array(sorted(pts))
        x, q25, med, q75 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        if len(x) > 1:
            ax.fill_between(x, q25, q75, color=GREEN, alpha=0.12, lw=0)
        ax.plot(x, med, color=GREEN, lw=1.6, ls=(0, (5, 3)), marker="^", ms=6,
                label="single model (no ensemble)", zorder=2)

    # --- reference lines: nominal p=0.5, bank floor p=1/(B+1), N_test=N_train ---
    Btyp = int(np.median(Bs)) if Bs else 100
    floor = 1.0 / (Btyp + 1)
    ax.axhline(0.5,   color=GRID, lw=1.0, ls=":",   zorder=0)
    ax.axhline(floor, color=GRID, lw=1.0, ls="--",  zorder=0)
    ax.axvline(1e5,   color=GRID, lw=1.0, ls="-.",  zorder=0)
    ax.text(2.05e5, 0.5,           " nominal",    color=MUTED, fontsize=8, va="center")
    ax.text(2.05e5, floor + 0.012, " bank floor", color=MUTED, fontsize=8, va="bottom")
    ax.text(1e5, 0.62, r"$N_{\rm test}=N_{\rm train}$", rotation=90,
            color=MUTED, fontsize=8, va="center", ha="right")

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlim(2.2e4, 2.3e5)
    ax.set_xticks([25000, 50000, 100000, 200000])
    ax.set_xticklabels(["25k", "50k", "100k", "200k"])
    ax.minorticks_off()
    ax.set_xlabel(r"$N_{\rm test}$", color=INK, fontsize=13)
    ax.set_ylabel(r"$p$-value   (median, 25–75%)", color=INK, fontsize=13)
    if args.title:
        ax.set_title(args.title, color=INK, fontsize=10.5)
    ax.tick_params(direction="in", which="both", colors=INK, top=True, right=True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        fig.savefig(out_base + ext, dpi=200, bbox_inches="tight")
    plt.close()
    print("wrote", out_base + ".png", "and", out_base + ".pdf")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["nens", "mode", "ntest", "p25", "p50", "p75", "ci_lo", "ci_hi"])
            for (nens, mode), pts in sorted(rows.items()):
                for row in sorted(pts):
                    w.writerow([nens, mode, *(f"{v:.6g}" for v in row)])
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
