"""Paper-style figures from results/summary.csv + pvalues (run after merge.py)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from paperstyle import (INK, GRID, MUTED, BLUE, ORANGE, style_ax, save_fig,
                        fmt_n, ramp, nominal_lines, frameless_legend)

sm_all = pd.read_csv(os.path.join(C.OUT_DIR, "summary.csv"))
sm = sm_all[sm_all.source == "truth"]
os.makedirs(C.FIG_DIR, exist_ok=True)
pt = sm[sm.test_type == "point"]; cp = sm[sm.test_type == "comp"]

# ---- headline: one figure per (N_fit, K) with composite results ----
for (N, K), g in cp.groupby(["N_fit", "K"]):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    g = g.sort_values("N_test"); xs = g.N_test.to_numpy()
    ax.fill_between(xs, g.p25, g.p75, color=BLUE, alpha=0.16, lw=0)
    ax.plot(xs, g.p50, color=BLUE, lw=2.3, marker="o", ms=6.5,
            label="composite  (uncertainty propagated)", zorder=3)
    gp = pt[(pt.N_fit == N) & (pt.K == K)].sort_values("N_test")
    if len(gp):
        ax.fill_between(gp.N_test, gp.p25, gp.p75, color=ORANGE, alpha=0.16, lw=0)
        ax.plot(gp.N_test, gp.p50, color=ORANGE, lw=2.3, marker="s", ms=6.5,
                label="point null  (plug-in at $\\hat\\theta$)", zorder=3)
    nominal_lines(ax, xs[-1])
    ax.axvline(N, color=GRID, lw=1.0, ls="-.", zorder=0)
    ax.text(N*1.06, 0.95, "$N_{\\rm test}=N_{\\rm fit}$", color=MUTED, fontsize=8)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xticks(xs); ax.set_xticklabels([fmt_n(n) for n in xs], fontsize=8)
    ax.set_xlabel(r"$N_{\rm test}$")
    ax.set_ylabel("null $p$-value    (median, 25–75%)")
    ax.set_title(f"4D embeddings · GMM $K={K}$, "
                 f"$N_\\mathrm{{fit}}={fmt_n(N)}$", fontsize=9.5)
    style_ax(ax)
    frameless_legend(ax, loc="lower left")
    fig.tight_layout()
    save_fig(fig, C.FIG_DIR, f"point_vs_composite_N{N}_K{K}")
    plt.close(fig)

# ---- K sweep of the point null, one figure per N_fit (blue shades) ----
for N, gN in pt.groupby("N_fit"):
    Ks = sorted(gN.K.unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for c, K in zip(ramp("Blues", len(Ks)), Ks):
        g = gN[gN.K == K].sort_values("N_test")
        ax.plot(g.N_test, g.p50, color=c, lw=1.9, marker="o", ms=4.5, label=f"$K={K}$")
    nominal_lines(ax, gN.N_test.max(), quantiles=(0.5,))
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.axhline(C.ALPHA_LEVEL, color=GRID, lw=1.0, ls="--", zorder=0)
    ax.text(0.98, C.ALPHA_LEVEL + 0.012, f"$\\alpha$={C.ALPHA_LEVEL}",
            color=MUTED, fontsize=8, ha="right", va="bottom", transform=tr)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel(r"$N_{\rm test}$")
    ax.set_ylabel("null $p$-value  (median)")
    ax.set_title(f"point-null test vs capacity · $N_\\mathrm{{fit}}={fmt_n(N)}$",
                 fontsize=9.5)
    style_ax(ax)
    frameless_legend(ax, loc="lower left")
    fig.tight_layout()
    save_fig(fig, C.FIG_DIR, f"point_vs_K_N{N}")
    plt.close(fig)

# ---- power / coverage from per-observation p-values ----
def load_pvalues():
    p = os.path.join(C.OUT_DIR, "pvalues")
    if os.path.exists(p + ".parquet"): return pd.read_parquet(p + ".parquet")
    if os.path.exists(p + ".pkl"): return pd.read_pickle(p + ".pkl")
    return None

pv = load_pvalues()
if pv is not None and (pv.source == "power").any():
    for (N, K, nt), g in pv[pv.source == "power"].groupby(["N_fit", "K", "N_test"]):
        rej = g.groupby("eps").p.apply(lambda x: (x < C.ALPHA_LEVEL).mean()).sort_index()
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.plot(rej.index, rej.values, color=BLUE, lw=2.3, marker="o", ms=6.5,
                label="composite rejection rate")
        base = pv[(pv.source == "truth") & (pv.test_type == "comp") &
                  (pv.N_fit == N) & (pv.K == K) & (pv.N_test == nt)]
        if len(base):
            ax.axhline((base.p < C.ALPHA_LEVEL).mean(), color=MUTED, lw=1.4,
                       ls=(0, (5, 3)), zorder=2)
            ax.text(rej.index.max(), (base.p < C.ALPHA_LEVEL).mean() + 0.015,
                    "undistorted data", color=MUTED, fontsize=8, ha="right")
        import matplotlib.transforms as mtransforms
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.axhline(C.ALPHA_LEVEL, color=GRID, lw=1.0, ls=":")
        ax.text(0.98, C.ALPHA_LEVEL + 0.012, f"$\\alpha$={C.ALPHA_LEVEL}",
                color=MUTED, fontsize=8, ha="right", va="bottom", transform=tr)
        ax.set(xlabel=r"$\epsilon$ (coord-0 scale distortion)",
               ylabel="rejection rate", ylim=(-0.02, 1.02))
        ax.set_title(f"power · $K={K}$, $N_\\mathrm{{fit}}$={fmt_n(N)}, "
                     f"$N_\\mathrm{{test}}$={fmt_n(nt)}", fontsize=9.5)
        style_ax(ax)
        frameless_legend(ax)
        fig.tight_layout()
        save_fig(fig, C.FIG_DIR, f"power_N{N}_K{K}_T{int(nt)}")
        plt.close(fig)

if pv is not None and (pv.source == "null").any():
    from scipy import stats as _st
    for (N, K, nt), g in pv[pv.source == "null"].groupby(["N_fit", "K", "N_test"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.hist(g.p, bins=np.linspace(0, 1, 11), density=True, alpha=.35,
                color=BLUE, edgecolor="none")
        ax.axhline(1, color=INK, ls="--", lw=1.0)
        ax.set(xlabel="p", ylabel="density")
        ax.set_title(f"coverage · $K={K}$, $N_\\mathrm{{fit}}$={fmt_n(N)}, "
                     f"$N_\\mathrm{{test}}$={fmt_n(nt)} "
                     f"(KS unif. p={_st.kstest(g.p, 'uniform').pvalue:.2f})",
                     fontsize=9.5)
        style_ax(ax)
        fig.tight_layout()
        save_fig(fig, C.FIG_DIR, f"coverage_N{N}_K{K}_T{int(nt)}")
        plt.close(fig)
