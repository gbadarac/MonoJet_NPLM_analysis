"""
Plot the classifier-GoF null distributions and optimisation diagnostics.

Two figures per run:
  1) gof_calibration.png — per-variant t_toy histograms with t_obs marked,
     empirical p-values annotated.
  2) gof_diagnostics.png — per-variant rows showing
       (a) loss curves (den + num) over L-BFGS evaluations,
       (b) toy convergence: ||∇L|| and n_eval distributions across toys,
       (c) weight pulls Δw/sqrt(diag Σ_w) for the observed fit
           (constrained variant — most informative; the frozen panel just
           notes that w ≡ ŵ).
       (d) histogram of perturbation coefficients b_num for the observed num fit.

Reads the per-variant JSONs, t_toy arrays, observed-fit diagnostics, and
toy diagnostic npz files written by run_gof.py.

Usage:
    python plot_gof.py --name <wifi_run_name>
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = PACKAGE_DIR
sys.path.insert(0, PACKAGE_DIR)

from classifier_gof import VARIANTS, GRAD_NORM_OK


def _plot_calibration(out_dir, run_name, results, toys):
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5 * len(VARIANTS), 4),
                             sharey=False)
    for ax, v in zip(axes, VARIANTS):
        t = toys[v]
        t_obs = results[v]["t_obs"]
        p_val = results[v]["p_value_addone"]

        lo = min(t.min(), t_obs) - 0.05 * (t.max() - t.min() + 1.0)
        hi = max(t.max(), t_obs) + 0.05 * (t.max() - t.min() + 1.0)
        bins = np.linspace(lo, hi, 30)
        ax.hist(t, bins=bins, density=True, color="C0", alpha=0.7,
                edgecolor="C0", label=f"toys (N={len(t)})")
        ax.axvline(t_obs, color="C3", lw=2.0,
                   label=f"$t_\\mathrm{{obs}}$={t_obs:.2f}")
        ax.axvline(np.median(t), color="k", ls="--", lw=1.0, alpha=0.6,
                   label=f"toy median={np.median(t):.2f}")
        ax.set_xlabel(r"$t = 2\,[L_\mathrm{den} - L_\mathrm{num}]$")
        ax.set_ylabel("density")
        ax.set_title(f"{v}  (p = {p_val:.3f})")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(f"wifi_better_basis classifier GoF: {run_name}", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "gof_calibration.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_diagnostics(out_dir, run_name):
    """One row per variant; columns: loss curves, ||∇|| dist, n_eval dist,
    weight pulls, perturbation-coefficient histogram."""
    n_cols = 5
    fig, axes = plt.subplots(len(VARIANTS), n_cols,
                             figsize=(4 * n_cols, 3.0 * len(VARIANTS)),
                             squeeze=False)

    # Need Sigma_w_sandwich for weight pulls
    Sigma_path = os.path.join(out_dir, "Sigma_w_sandwich.npy")
    Sigma_w = np.load(Sigma_path) if os.path.exists(Sigma_path) else None

    for r, v in enumerate(VARIANTS):
        obs_path = os.path.join(out_dir, f"gof_{v}_obs_diag.npz")
        toy_path = os.path.join(out_dir, f"gof_{v}_toy_diag.npz")
        if not (os.path.exists(obs_path) and os.path.exists(toy_path)):
            print(f"  missing diag arrays for variant '{v}' — skipping")
            continue
        obs = np.load(obs_path)
        toy = np.load(toy_path)

        # ── (a) Loss curves ──────────────────────────────────────
        ax = axes[r][0]
        d_hist = obs["den_loss_hist"]; n_hist = obs["num_loss_hist"]
        ax.plot(np.arange(len(d_hist)), d_hist, color="C0", lw=1.0,
                label=f"den (n={obs['den_n_eval']:.0f}, "
                      f"||∇||={obs['den_grad_norm']:.1e})")
        ax.plot(np.arange(len(n_hist)), n_hist, color="C3", lw=1.0,
                label=f"num (n={obs['num_n_eval']:.0f}, "
                      f"||∇||={obs['num_grad_norm']:.1e})")
        ax.set_xlabel("L-BFGS closure call")
        ax.set_ylabel(r"$L = \mathrm{BCE} + \mathrm{prior} + \mathrm{ridge}$")
        ax.set_title(f"{v}: observed-fit loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # ── (b) Toy ||∇L|| distributions ─────────────────────────
        ax = axes[r][1]
        gn_d = toy["den_grad_norm"]
        gn_n = toy["num_grad_norm"]
        # log-scale histogram, guarding zeros
        eps = 1e-30
        bins = np.logspace(
            np.log10(max(eps, min(gn_d.min(), gn_n.min(), eps))),
            np.log10(max(gn_d.max(), gn_n.max(), GRAD_NORM_OK) * 2 + eps),
            20,
        )
        ax.hist(np.clip(gn_d, eps, None), bins=bins, color="C0", alpha=0.6,
                edgecolor="C0", label="den")
        ax.hist(np.clip(gn_n, eps, None), bins=bins, color="C3", alpha=0.6,
                edgecolor="C3", label="num")
        ax.axvline(GRAD_NORM_OK, color="k", ls="--", lw=1.0,
                   label=f"OK threshold ({GRAD_NORM_OK:.0e})")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\|\nabla L\|_2$ at fit end")
        ax.set_ylabel("toy count")
        ax.set_title(f"{v}: toy gradient norms")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

        # ── (c) Toy n_eval distributions ─────────────────────────
        ax = axes[r][2]
        ne_d = toy["den_n_eval"]
        ne_n = toy["num_n_eval"]
        bins = np.linspace(0, max(ne_d.max(), ne_n.max()) + 1, 20)
        ax.hist(ne_d, bins=bins, color="C0", alpha=0.6, edgecolor="C0",
                label=f"den (median={np.median(ne_d):.0f})")
        ax.hist(ne_n, bins=bins, color="C3", alpha=0.6, edgecolor="C3",
                label=f"num (median={np.median(ne_n):.0f})")
        ax.set_xlabel("L-BFGS closure calls per toy fit")
        ax.set_ylabel("toy count")
        ax.set_title(f"{v}: toy n_eval")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # ── (d) Weight pulls (Δw / σ_w_diag) ────────────────────
        ax = axes[r][3]
        w_hat = obs["w_hat"]
        if Sigma_w is not None and v != "frozen":
            sw_diag = np.sqrt(np.diag(Sigma_w).clip(min=1e-30))
            dw_den = (obs["w_den"] - w_hat) / sw_diag
            dw_num = (obs["w_num"] - w_hat) / sw_diag
            idx = np.arange(w_hat.size)
            ax.bar(idx - 0.18, dw_den, width=0.35, color="C0",
                   label="den", alpha=0.85)
            ax.bar(idx + 0.18, dw_num, width=0.35, color="C3",
                   label="num", alpha=0.85)
            ax.axhline(+1, color="gray", ls=":", lw=1.0)
            ax.axhline(-1, color="gray", ls=":", lw=1.0)
            ax.set_xlabel("weight index")
            ax.set_ylabel(r"$(w - \hat w) / \sqrt{\mathrm{diag}\,\Sigma_w}$")
            ax.set_title(f"{v}: observed weight pulls")
            ax.legend(fontsize=7)
        else:
            if v == "frozen":
                ax.text(0.5, 0.5, "frozen variant\n($w \\equiv \\hat w$, no pulls)",
                        ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Σ_w_sandwich.npy not found",
                        ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.3)

        # ── (e) Numerator perturbation coefficients ─────────────
        ax = axes[r][4]
        b = obs["b_num"]
        if b.size > 0:
            ax.hist(b, bins=30, color="C3", alpha=0.7, edgecolor="C3")
            ax.axvline(0, color="k", ls="--", lw=1.0)
            ax.set_xlabel(r"$b_j$ (num)")
            ax.set_ylabel("count")
            rms = float(np.sqrt(np.mean(b ** 2)))
            ax.set_title(f"{v}: $b_\\mathrm{{num}}$ rms={rms:.3f}, "
                         f"max|b|={np.max(np.abs(b)):.3f}")
        else:
            ax.text(0.5, 0.5, "no perturbation coefficients\n(b_num is empty)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.3)

    fig.suptitle(f"wifi_better_basis GoF diagnostics: {run_name}", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "gof_diagnostics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    args = p.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "runs", args.name)

    results = {}
    toys = {}
    for v in VARIANTS:
        rp = os.path.join(out_dir, f"gof_{v}_results.json")
        tp = os.path.join(out_dir, f"gof_{v}_toys_t.npy")
        if not (os.path.exists(rp) and os.path.exists(tp)):
            print(f"missing artifacts for variant '{v}' — run run_gof.py first")
            return
        with open(rp) as f:
            results[v] = json.load(f)
        toys[v] = np.load(tp)

    # Print headline + convergence summary. The two variants now use
    # different toy DGPs (constrained: posterior-predictive at w_k;
    # frozen: plug-in at ŵ), so there's no inherent t-ordering between them.
    for v in VARIANTS:
        r = results[v]
        flags = []
        if not r.get("obs_den_converged", True):
            flags.append("obs den unconverged")
        if not r.get("obs_num_converged", True):
            flags.append("obs num unconverged")
        n_unc = r.get("toy_n_unconverged_den", 0) + r.get("toy_n_unconverged_num", 0)
        if n_unc:
            flags.append(f"{n_unc} toy fits unconverged")
        n_neg = r.get("n_toys_with_negative_t", 0)
        if n_neg:
            flags.append(f"{n_neg} toys with t<0")
        flag_str = "  WARN: " + "; ".join(flags) if flags else ""
        null_tag = r.get("null_kind", "")
        print(f"  t_obs [{v:<11s} | {null_tag}] = {r['t_obs']:+.3f}, "
              f"p = {r['p_value_addone']:.3f}{flag_str}")

    _plot_calibration(out_dir, args.name, results, toys)
    _plot_diagnostics(out_dir, args.name)


if __name__ == "__main__":
    main()
