"""t-distribution diagnostics for EVERY stored working point (no recomputation:
reads results/tstats.* produced by merge.py).

Outputs (in <run>/figs/):
  diagnostics_grid_<test>.pdf       ALL working points of a test type on one page
                                    (null bank vs chi2 with moment-matched + fitted dof
                                    + KS p, and every observed source overlaid:
                                    truth / null / power-eps)
  dof_vs_ntest.pdf                  fitted chi2 dof vs N_test across working points
  results/diagnostics_summary.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import config as C
from paperstyle import INK, GRID, MUTED, BLUE, ORANGE, style_ax, save_fig, ramp

def load(name):
    p = os.path.join(C.OUT_DIR, name)
    if os.path.exists(p + ".parquet"):
        return pd.read_parquet(p + ".parquet")
    return pd.read_pickle(p + ".pkl")

df = load("tstats")
os.makedirs(C.FIG_DIR, exist_ok=True)

SRC_STYLE = {"truth": (ORANGE, "-",  "data (truth)"),
             "null":  (MUTED, "--", "coverage (null-drawn)")}

def panel(ax, g, title):
    bank = g[g.source == "calib"].t.to_numpy()
    if len(bank) < 5:
        ax.set_axis_off(); return None
    dof_mm = float(bank.mean())
    try:
        dof_fit = float(stats.chi2.fit(np.maximum(bank, 1e-6), floc=0, fscale=1)[0])
    except Exception:
        dof_fit = dof_mm
    ks_p = float(stats.kstest(bank, stats.chi2(dof_fit).cdf).pvalue)
    alts = g[g.source != "calib"]
    hi = np.percentile(np.concatenate([bank, alts.t.to_numpy()]) if len(alts) else bank,
                       99.5) * 1.05
    grid = np.linspace(0, max(hi, dof_mm*2), 300)
    bins = np.linspace(0, grid[-1], max(12, min(40, len(bank)//8)))
    ax.hist(bank, bins=bins, density=True, alpha=.35, color=BLUE,
            edgecolor="none", label=f"null bank (n={len(bank)})")
    ax.plot(grid, stats.chi2(dof_mm).pdf(grid), color=INK, ls="--", lw=1.3,
            label=fr"$\chi^2$ dof=$\bar t$={dof_mm:.1f}")
    ax.plot(grid, stats.chi2(dof_fit).pdf(grid), color=INK, lw=1.0,
            label=fr"$\chi^2$ fit dof={dof_fit:.1f} (KS p={ks_p:.2f})")
    for src, ga in alts.groupby("source"):
        if src == "power":
            for eps, ge in ga.groupby("eps"):
                ax.hist(ge.t, bins=bins, density=True, histtype="step", lw=1.7,
                        label=fr"power $\epsilon$={eps:g} (n={len(ge)})")
        else:
            col, ls, lab = SRC_STYLE.get(src, ("C3", "-", src))
            ax.hist(ga.t, bins=bins, density=True, histtype="step", lw=1.7,
                    color=col, ls=ls, label=f"{lab} (n={len(ga)})")
    ax.set_xlabel("t"); ax.set_title(title, fontsize=9)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=6, labelcolor=INK)
    return dict(dof_mm=dof_mm, dof_fit=dof_fit, ks_p=ks_p, n_bank=len(bank))

summary = []
for tt in ["point", "comp"]:
    d = df[df.test_type == tt]
    if not len(d):
        continue
    groups = sorted(d.groupby(["N_fit", "K", "N_test"]).groups.keys())

    # --- everything on one page (grid); summary stats collected per panel ---
    ncol = max(1, len(set(nt for _, _, nt in groups)))
    nrow = int(np.ceil(len(groups)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6*ncol, 3.4*nrow), squeeze=False)
    for i, (N, K, nt) in enumerate(groups):
        g = d[(d.N_fit == N) & (d.K == K) & (d.N_test == nt)]
        info = panel(axes[i//ncol][i % ncol], g, f"N_fit={N}, K={K}, N_test={nt}")
        if info is not None:
            summary.append(dict(test_type=tt, N_fit=N, K=K, N_test=nt, **info))
    for i in range(len(groups), nrow*ncol):
        axes[i//ncol][i % ncol].set_axis_off()
    fig.suptitle(f"{tt} test — all working points", fontsize=12, color=INK)
    plt.tight_layout()
    save_fig(fig, C.FIG_DIR, f"diagnostics_grid_{tt}")
    plt.close(fig)
    print(f"  ({len(groups)} panels)")

sm = pd.DataFrame(summary)
if len(sm):
    sm.to_csv(os.path.join(C.OUT_DIR, "diagnostics_summary.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    keys = sorted(sm.groupby(["N_fit", "K"]).groups.keys())
    shades = {k: c for k, c in zip(keys, ramp("Blues", len(keys)))}
    for (tt, N, K), g in sm.groupby(["test_type", "N_fit", "K"]):
        g = g.sort_values("N_test")
        ax.plot(g.N_test, g.dof_fit, marker="o", ms=4, color=shades[(N, K)],
                ls=("-" if tt == "point" else "--"),
                label=f"{tt}, N_fit={N}, K={K}")
    ax.axhline(C.J_CENTERS, color=GRID, ls=":", lw=1)
    ax.text(sm.N_test.max(), C.J_CENTERS + 0.5, f"J = {C.J_CENTERS}", color=MUTED,
            fontsize=8, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\rm test}$")
    ax.set_ylabel(r"fitted $\chi^2$ dof of the null bank")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=7, labelcolor=INK)
    plt.tight_layout()
    save_fig(fig, C.FIG_DIR, "dof_vs_ntest")
    plt.close(fig)
    with pd.option_context("display.width", 140):
        print(sm.to_string(index=False))
