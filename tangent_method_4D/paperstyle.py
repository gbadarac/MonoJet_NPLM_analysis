"""Shared paper plotting style (coworker template, plot_fresh.py)."""
import os
import numpy as np
import matplotlib.pyplot as plt

INK, GRID, MUTED = "#33322e", "#c9c8c0", "#8a897f"
BLUE, ORANGE = "#2a78d6", "#eb6834"

def style_ax(ax):
    ax.tick_params(direction="in", which="both", colors=INK, top=True, right=True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.minorticks_off()
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)

def save_fig(fig, fig_dir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"), dpi=200,
                    bbox_inches="tight")
    print(os.path.join(fig_dir, stem) + ".{png,pdf}")

def fmt_n(n):
    n = int(n)
    return f"{n//1_000_000}M" if n >= 1_000_000 else (f"{n//1000}k" if n >= 1000 else str(n))

def ramp(base_cmap, n, lo=0.45, hi=0.95):
    """n shades of a matplotlib colormap, light -> dark."""
    cm = plt.get_cmap(base_cmap)
    return [cm(x) for x in np.linspace(lo, hi, max(n, 1))]

def nominal_lines(ax, x_right=None, quantiles=(0.25, 0.5, 0.75)):
    # x_right kept for backward compat (ignored): labels are placed at 98% of the
    # axes width via a blended transform - robust to log scales and any x-limits.
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for q in quantiles:
        ax.axhline(q, color=GRID, lw=1.0, ls=":", zorder=0)
        ax.text(0.98, q + 0.012, f"nominal {int(q*100)}%", color=MUTED,
                fontsize=8, ha="right", va="bottom", zorder=1, transform=tr)

def frameless_legend(ax, **kw):
    kw.setdefault("frameon", False); kw.setdefault("labelcolor", INK)
    kw.setdefault("fontsize", 9)
    ax.legend(**kw)
