# Falkon packages
from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss
# scipy packages
from scipy.spatial.distance import pdist
from scipy.stats import norm, chi2, rv_continuous, kstest

# plot packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "serif"
plt.style.use('classic')
font = font_manager.FontProperties(family='serif', size=20)

# others
import numpy as np
import os, time
import torch

import matplotlib as mpl
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.fonttype'] = 42   # keep fonts editable / display nicely
mpl.rcParams['ps.fonttype']  = 42

def candidate_sigma(data, perc=90):
    # this function estimates the width of the gaussian kernel.                          
    # use on a (small) sample of reference data (standardize first if necessary)         
    pairw = pdist(data)
    return np.around(np.percentile(pairw,perc),1)

def get_logflk_config(M,flk_sigma,lam,weight,iter=[1000000],seed=None,cpu=False):
    # it returns logfalkon parameters                                                                                                              
    return {
            'kernel' : GaussianKernel(sigma=flk_sigma),
            'M' : M, #number of Nystrom centers,                                                                                                  
            'penalty_list' : lam, # list of regularization parameters,                                                                            
            'iter_list' : iter, #list of number of CG iterations,                                                                                 
            'options' : FalkonOptions(cg_tolerance=np.sqrt(float(1e-7)), keops_active='no', use_cpu=cpu, debug = False),
            'seed' : seed, # (int or None), the model seed (used for Nystrom center selection) is manually set,                                   
            'loss' : WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=flk_sigma), neg_weight=weight),
            }

def trainer(X,Y,flk_config):
    # trainer for logfalkon model
    Xtorch= torch.from_numpy(X)
    Ytorch= torch.from_numpy(Y)
    model = LogisticFalkon(**flk_config)
    model.fit(Xtorch, Ytorch)
    return model.predict(Xtorch).numpy()

def compute_t(preds,Y,weight):
    # it returns extended log likelihood ratio from predictions
    diff = weight*np.sum(1 - np.exp(preds[Y==0]))
    return 2 * (diff + np.sum(preds[Y==1]))

def run_toy(test_label, X_train, Y_train, weight, flk_config, seed, bins_centroids, f_bins_centroids, ferr_bins_centroids, 
            plot=False, verbose=False, savefig=False, output_path='', df=10, binsrange=None, yrange=None, xlabels=None, 
            ):
    

    if not os.path.exists(output_path):
      os.makedirs(output_path, exist_ok=True)
    dim = X_train.shape[1]                                                                                                        
    flk_config['seed']=seed # select different centers for different toys                                                   
    st_time = time.time()
    preds = trainer(X_train,Y_train,flk_config)
    t = compute_t(preds,Y_train,weight)
    dt = round(time.time()-st_time,2)
    if verbose:
        print("toy {}\n---LRT = {}\n---Time = {} sec\n\t".format(seed,t,dt))
    if plot:
        for i in range(X_train.shape[1]):
            print(f"\n[DEBUG] Feature {i+1}")
            print("  f_bins_centroids has NaNs? ", np.any(np.isnan(f_bins_centroids[:, i])))
            print("  f_bins_centroids == 0?    ", np.any(f_bins_centroids[:, i] == 0))
            print("  ferr_bins_centroids has NaNs? ", np.any(np.isnan(ferr_bins_centroids[:, i])))
            print("  ferr_bins_centroids == 0?    ", np.any(ferr_bins_centroids[:, i] == 0))
        plot_reconstruction(data=X_train[Y_train.flatten()==1], weight_data=1,
                            ref=X_train[Y_train.flatten()==0], weight_ref=weight,
                            ref_preds=preds[Y_train.flatten()==0],                 
                            xlabels=xlabels, 
                            bins_centroids=bins_centroids, f_bins_centroids=f_bins_centroids, ferr_bins_centroids=ferr_bins_centroids, 
                            yrange=yrange,
                            save=savefig, save_path=output_path+'/plots/',
                            file_name=test_label+'.pdf'
                )
    return t, preds

def centroids_to_edges(centroids):
    """
    Convert an array of bin centroids to bin edges.
    Parameters:
        centroids (array-like): 1D array of bin centroids (must be sorted).
    Returns:
        np.ndarray: 1D array of bin edges, length = len(centroids) + 1
    """
    centroids = np.asarray(centroids)
    if centroids.ndim != 1:
        raise ValueError("Input must be a 1D array of centroids.")
    if len(centroids) < 2:
        raise ValueError("At least two centroids are required to compute edges.")
    
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    first_edge = centroids[0] - (midpoints[0] - centroids[0])
    last_edge = centroids[-1] + (centroids[-1] - midpoints[-1])
    edges = np.concatenate([[first_edge], midpoints, [last_edge]])
    return edges

def ratio_with_error(a, b, a_err, b_err):
    """
    Compute the element-wise ratio of two arrays with error propagation.
    
    Parameters:
        a (np.ndarray): Numerator values
        b (np.ndarray): Denominator values
        a_err (np.ndarray): Errors in numerator
        b_err (np.ndarray): Errors in denominator
        
    Returns:
        ratio (np.ndarray): Element-wise ratio a / b
        ratio_err (np.ndarray): Propagated uncertainty of the ratio
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.asarray(a_err)
    b_err = np.asarray(b_err)
    if not (a.shape == b.shape == a_err.shape == b_err.shape):
        raise ValueError("All input arrays must have the same shape.")
    ratio = a / b
    ratio_err = ratio * np.sqrt((a_err / a)**2 + (b_err / b)**2)
    return ratio, ratio_err


from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_reconstruction(
    data, weight_data, ref, weight_ref, ref_preds,
    xlabels=[],                     # ["Feature 1","Feature 2"] recommended
    bins_centroids=None,            # shape [Nb, D]
    f_bins_centroids=None,          # shape [Nb, D]
    ferr_bins_centroids=None,       # shape [Nb, D]
    yrange=None,
    save=False, save_path='', file_name=''
):
    """
    Makes ONE figure with 2 columns (Feature 1 | Feature 2), each with
    a top 'events' panel and a bottom 'ratio' panel. Legends live in the
    middle white column so the plots are bigger.
    """
    assert data.shape[1] >= 2, "This combined plot expects at least 2 features."

    eps = 1e-10
    weight_ref = np.ones(len(ref)) * weight_ref
    weight_data = np.ones(len(data)) * weight_data

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 28,        # base
        "axes.labelsize": 44,   # axis labels
        "xtick.labelsize": 32,
        "ytick.labelsize": 32,
        "legend.fontsize": 30,
    })

    # ---- figure layout: 2 rows x 3 columns (left plots | legend column | right plots)
    # widen plots, keep a slim center column for legends
    fig = plt.figure(figsize=(32, 12))   # was (24, 12) → make figure wider
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.0, 0.60, 1.0],   # widen legend column a lot
        height_ratios=[1.3, 0.7],
        left=0.055, right=0.985, bottom=0.10, top=0.95,
        wspace=0.05, hspace=0.10         # a little more spacing
    )



    ax1L = fig.add_subplot(gs[0, 0])  # top left (events, feat 1)
    ax2L = fig.add_subplot(gs[1, 0])  # bottom left (ratio, feat 1)
    axC  = fig.add_subplot(gs[:, 1])  # center legend column
    ax1R = fig.add_subplot(gs[0, 2])  # top right (events, feat 2)
    ax2R = fig.add_subplot(gs[1, 2])  # bottom right (ratio, feat 2)
    axC.axis("off")

    def centroids_to_edges(centroids):
        c = np.asarray(centroids)
        mid = (c[:-1] + c[1:]) / 2
        first = c[0] - (mid[0] - c[0])
        last  = c[-1] + (c[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])

    def ratio_with_error(a, b, a_err, b_err):
        a, b, a_err, b_err = map(np.asarray, (a, b, a_err, b_err))
        ratio = a / b
        ratio_err = ratio * np.sqrt((a_err / a)**2 + (b_err / b)**2)
        return ratio, ratio_err

    # --------- helper: draw one feature into (ax_top, ax_bot)
    def draw_feature(i, ax_top, ax_bot):
        bins = centroids_to_edges(bins_centroids[:, i])
        N = np.sum(weight_ref)
        bin_lengths = (bins[1:] - bins[:-1]) * N

        # --- top panel (events)
        hD = ax_top.hist(
            data[:, i], weights=weight_data, bins=bins,
            label="DATA", color="black", lw=1.0, histtype="step", zorder=2
        )
        hR = ax_top.hist(
            ref[:, i], weights=weight_ref, bins=bins,
            color="green", ec="green", alpha=0.25, lw=1, label="REFERENCE", zorder=1
        )
        hN = ax_top.hist(
            ref[:, i], weights=np.exp(ref_preds[:, 0]) * weight_ref,
            bins=bins, histtype="step", lw=0
        )

        # data error bars & red LRT dots
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax_top.errorbar(
            centers, hD[0], yerr=np.sqrt(hD[0]),
            color="black", ls="", marker="o", ms=6, zorder=3,
            elinewidth=0.8, markeredgewidth=0.6, capsize=2
        )
        ax_top.scatter(
            centers, hN[0],
            label="LRT RECO", color="#e41a1c",
            edgecolors="white", linewidths=0.6, s=100, zorder=5
        )

        ax_top.set_yscale("log")
        ax_top.set_xlim(bins[0], bins[-1])
        ax_top.set_ylabel("events")
        ax_top.tick_params(axis="x", labelbottom=False)

        # --- bottom panel (ratio)
        x = bins_centroids[:, i]
        area = np.sum(bin_lengths * f_bins_centroids[:, i])
        # GEN/REF with uncertainty from f and ferr
        ratio, ratio_err = ratio_with_error(
            N / area * bin_lengths * f_bins_centroids[:, i],
            hR[0],
            N / area * bin_lengths * ferr_bins_centroids[:, i],
            np.sqrt(hR[0])
        )
        # LRT/REF
        ratio_nplm = hN[0] / (hR[0] + eps)
        mask = ratio_nplm > 0
        ax_bot.errorbar(
            x, hD[0] / (hR[0] + eps),
            yerr=np.sqrt(hD[0]) / (hR[0] + eps),
            ls="", marker="o", label="DATA/REF",
            color="black", zorder=1, ms=6, elinewidth=0.8, markeredgewidth=0.6, capsize=2
        )
        ax_bot.plot(x[mask], ratio_nplm[mask], label="LRT RECO/REF", color="#e41a1c", lw=2)
        ax_bot.plot(x, ratio, label="GEN/REF", color="blue", lw=2)
        ax_bot.fill_between(x, ratio - ratio_err, ratio + ratio_err, color="blue", alpha=0.30, label=r"$\pm 1\sigma$")
        ax_bot.fill_between(x, ratio - 2 * ratio_err, ratio + 2 * ratio_err, color="blue", alpha=0.15, label=r"$\pm 2\sigma$")

        # faint grid
        ax_bot.set_axisbelow(True)
        ax_bot.grid(True, which="major", axis="both", linestyle=":", linewidth=0.8, alpha=0.25)

        ax_bot.set_xlim(bins[0], bins[-1])
        ax_bot.set_ylim(0.05, 2.0)
        ax_bot.set_ylabel("ratio")
        xlabel = xlabels[i] if (len(xlabels) > i) else f"Feature {i+1}"
        ax_bot.set_xlabel(xlabel)

        if yrange is not None and isinstance(yrange, dict) and (xlabel in yrange):
            ax_bot.set_ylim(yrange[xlabel][0], yrange[xlabel][1])

    # draw left (feature 1) and right (feature 2)
    draw_feature(0, ax1L, ax2L)
    draw_feature(1, ax1R, ax2R)

    # ---- centered legends in the middle column (two stacks)
    # ---- centered legends in the middle column (split into two blocks)

    # ---- two figure-level legends, truly centered horizontally
    top_handles = [
        Line2D([0], [0], color="black", lw=1.0, label="DATA"),
        Patch(facecolor="green", edgecolor="green", alpha=0.25, label="REFERENCE"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=10,
            markerfacecolor="#e41a1c", markeredgecolor="white", label="LRT RECO"),
    ]
    bottom_handles = [
        Line2D([0], [0], color="#e41a1c", lw=2, label="LRT RECO/REF"),
        Line2D([0], [0], color="blue", lw=2, label="GEN/REF"),
        Patch(facecolor="blue", alpha=0.30, label=r"$\pm1\sigma$"),
        Patch(facecolor="blue", alpha=0.15, label=r"$\pm2\sigma$"),
        Line2D([0], [0], marker="o", linestyle="None", color="black", label="DATA/REF"),
    ]

    # keep axC just as a spacer
    axC.axis("off")

    leg_top = fig.legend(
        handles=top_handles, loc="center",
        bbox_to_anchor=(0.50, 0.62),   # dead-center in the figure horizontally
        bbox_transform=fig.transFigure,
        frameon=False, ncol=1,
        handlelength=1.8, handletextpad=0.6, borderaxespad=0.0
    )
    leg_bot = fig.legend(
        handles=bottom_handles, loc="center",
        bbox_to_anchor=(0.50, 0.38),   # second block below the first
        bbox_transform=fig.transFigure,
        frameon=False, ncol=1,
        handlelength=1.8, handletextpad=0.6, borderaxespad=0.0
    )


    if save:
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, file_name if file_name.endswith(".pdf")
                                else file_name.replace(".png", ".pdf"))
        if not out_path.endswith(".pdf"):
            out_path = out_path + ".pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)  # vector PDF
    plt.close(fig)

