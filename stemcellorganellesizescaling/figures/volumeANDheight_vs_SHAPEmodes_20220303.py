# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
from matplotlib.colors import ListedColormap
import pickle

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

#%% Directories
#%% Directories
if platform.system() == "Windows":
    data_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Oct2021")
    pic_root = Path("Z:/modeling/theok/Projects/Data/scoss/Pics/Oct2021")
elif platform.system() == "Linux":
    1 / 0

pic_rootT = pic_root / "edgeviolins"
pic_rootT.mkdir(exist_ok=True)


# %% Resolve directories and load data
tableIN = "SizeScaling_20211101.csv"
# Load dataset
cells = pd.read_csv(all_root / tableIN)



if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Dec2020")
    pic_root = Path("E:/DA/Data/scoss/Pics/Dec2020/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/Dec2020")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Dec2020/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

# Load dataset
tableIN = "SizeScaling_20201215.csv"
statsIN = "Stats_20201215"
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())

# %% Parameters, updated directories
save_flag = 1  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "plot_cellmetrics_vs_PCA"
pic_root.mkdir(exist_ok=True)

# %% Feature sets
FS = {}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    "Cytoplasmic volume",
]
FS["cellnuc_abbs"] = [
    "Cell area",
    "Cell vol",
    "Cell height",
    "Nuclear area",
    "Nuclear vol",
    "Nucleus height",
    "Cyto vol",
]

# %% Plotting function
def scatter(
    ax,
    axB,
    axS,
    metricX,
    metricY,
    abbX,
    abbY,
    cells,
    stats_root,
    kde_flag=False,
    fourcolors_flag=False,
    colorpoints_flag=False,
    rollingavg_flag=False,
    ols_flag=False,
    N2=1000,
    fs2=10,
    fs=5,
    typ=["vol", "vol"],
    PrintType="all",
):

    #%% Change labels
    if typ[0] == "vol":
        abbX = f"{abbX} (\u03BCm\u00b3)"
    elif typ[0] == "area":
        abbX = f"{abbX} (\u03BCm\u00b2)"
    elif typ[0] == "height":
        abbX = f"{abbX} (\u03BCm)"
    if typ[1] == "vol":
        abbY = f"{abbY} (\u03BCm\u00b3)"
    elif typ[1] == "area":
        abbY = f"{abbY} (\u03BCm\u00b2)"
    elif typ[1] == "height":
        abbY = f"{abbY} (\u03BCm)"
    if typ[0] == "vol":
        facX = 1 / ((0.108333) ** 3)
    elif typ[0] == "area":
        facX = 1 / ((0.108333) ** 2)
    elif typ[0] == "height":
        facX = 1 / ((0.108333) ** 1)
    else:
        facX = 1
    if typ[1] == "vol":
        facY = 1 / ((0.108333) ** 3)
    elif typ[1] == "area":
        facY = 1 / ((0.108333) ** 2)
    elif typ[1] == "height":
        facY = 1 / ((0.108333) ** 1)
    else:
        facY = 1

    #%% Archery new colormap
    white = np.array([1, 1, 1, 1])
    green = np.array([0, 1, 0, 1])
    blue = np.array([0, 0, 1, 1])
    red = np.array([1, 0, 0, 1])
    magenta = np.array([0.5, 0, 0.5, 1])
    black = np.array([0, 0, 0, 1])
    newcolors = np.zeros((100, 4))
    newcolors[0:10, :] = white
    newcolors[10:90, :] = blue
    newcolors[50:90, :] = red
    newcolors[90:100, :] = magenta
    newcmp = ListedColormap(newcolors)

    #%% Spectral
    cpmap = plt.cm.get_cmap(plt.cm.plasma)
    cpmap = cpmap(np.linspace(0, 1, 100) ** 0.4)
    cpmap[0:10, 3] = np.linspace(0.3, 1, 10)
    cpmap = ListedColormap(cpmap)

    #%% Plotting parameters
    ms = 0.5
    lw2 = 2
    nbins = 100
    plt.rcParams.update({"font.size": fs})

    # data
    x = cells[metricX]
    y = cells[metricY]
    x = x / facX
    y = y / facY

    # plot
    if kde_flag is True:
        xii = loadps(stats_root, f"{metricX}_{metricY}_xii") / facX
        yii = loadps(stats_root, f"{metricX}_{metricY}_yii") / facY
        zii = loadps(stats_root, f"{metricX}_{metricY}_zii")
        cii = loadps(stats_root, f"{metricX}_{metricY}_cell_dens")
        ax.set_ylim(top=np.max(yii))
        ax.set_ylim(bottom=np.min(yii))
        if fourcolors_flag is True:
            ax.pcolormesh(xii, yii, zii, cmap=newcmp)
        elif colorpoints_flag is True:
            sorted_cells = np.argsort(cii)
            cii[sorted_cells] = np.arange(len(sorted_cells))
            qqq = ax.scatter(x, y, c=cii, s=ms, cmap=cpmap)
        else:
            ax.pcolormesh(xii, yii, zii, cmap=plt.cm.magma)
            sorted_cells = np.argsort(cii)
            np.random.shuffle(sorted_cells)
            min_cells = sorted_cells[0:N2]
            min_cells = min_cells.astype(int)
            ax.plot(x[min_cells], y[min_cells], "w.", markersize=ms)
    else:
        ax.plot(x, y, "b.", markersize=ms)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if PrintType == "svg":
        qqq.remove()
    if PrintType != "png":
        ax.grid()
    # ax.text(xlim[0],ylim[1],f"{abbX} vs {abbY}",fontsize=fs2, verticalalignment = 'top')
    if kde_flag is True:
        if (fourcolors_flag is True) or (colorpoints_flag is True):
            if PrintType != "png":
                ax.text(
                    xlim[1],
                    ylim[1],
                    f"n= {len(x)}",
                    fontsize=fs,
                    verticalalignment="top",
                    horizontalalignment="right",
                    color="black",
                )
        else:
            ax.text(
                xlim[1],
                ylim[1],
                f"n= {len(x)}",
                fontsize=fs,
                verticalalignment="top",
                horizontalalignment="right",
                color="white",
            )
    else:
        ax.text(
            xlim[1],
            ylim[1],
            f"n= {len(x)}",
            fontsize=fs,
            verticalalignment="top",
            horizontalalignment="right",
        )
    if rollingavg_flag is True:
        if PrintType != "png":
            rollavg_x = loadps(stats_root, f"{metricX}_{metricY}_x_ra") / facX
            rollavg_y = loadps(stats_root, f"{metricX}_{metricY}_y_ra") / facY
            ax.plot(rollavg_x, rollavg_y[:, 0], "lime", linewidth=lw2)

    if ols_flag is True:
        xii = loadps(stats_root, f"{metricX}_{metricY}_xii") / facX
        pred_yL = loadps(stats_root, f"{metricX}_{metricY}_pred_matL") / facY
        pred_yC = loadps(stats_root, f"{metricX}_{metricY}_pred_matC") / facY
        if kde_flag is True:
            if PrintType != "png":
                if fourcolors_flag is True:
                    ax.plot(xii, pred_yL, "gray")
                elif colorpoints_flag is True:
                    ax.plot(xii, pred_yL, "gray")
                else:
                    ax.plot(xii, pred_yL, "w")
        else:
            ax.plot(xii, pred_yL, "r")
            ax.plot(xii, pred_yC, "m")

    if ols_flag is True:
        val = loadps(stats_root, f"{metricX}_{metricY}_rs_vecL")
        ci = np.round(np.percentile(val, [2, 98]), 2)
        cim = np.round(np.percentile(val, [50]), 2)
        pc = np.round(np.sqrt(np.percentile(val, [50])), 2)
        val2 = loadps(stats_root, f"{metricX}_{metricY}_rs_vecC")
        ci2 = np.round(np.percentile(val2, [2, 98]), 2)
        pc2 = np.round(np.sqrt(np.percentile(val2, [50])), 2)

        if kde_flag is True:
            if fourcolors_flag is True:
                plt.text(
                    xlim[0],
                    ylim[1],
                    f"rs={ci[0]}-{ci[1]}",
                    fontsize=fs2 - 2,
                    verticalalignment="top",
                    color="gray",
                )
                plt.text(
                    xlim[0],
                    0.9 * ylim[1],
                    f"pc={pc[0]}",
                    fontsize=fs2 - 2,
                    verticalalignment="top",
                    color="gray",
                )
            elif colorpoints_flag is True:
                if PrintType != "png":
                    plt.text(
                        xlim[0],
                        ylim[1],
                        # f"rs={cim[0]}, pc={pc[0]}",
                        f"R\u00b2={cim[0]}",
                        fontsize=fs,
                        verticalalignment="top",
                        horizontalalignment="left",
                        color="black",
                    )
            else:
                plt.text(
                    xlim[0],
                    ylim[1],
                    f"rs={ci[0]}-{ci[1]}",
                    fontsize=fs2 - 2,
                    verticalalignment="top",
                    color="w",
                )
                plt.text(
                    xlim[0],
                    0.9 * ylim[1],
                    f"pc={pc[0]}",
                    fontsize=fs2 - 2,
                    verticalalignment="top",
                    color="w",
                )
        else:
            plt.text(
                xlim[0],
                ylim[1],
                f"rs={ci[0]}-{ci[1]}",
                fontsize=fs2 - 2,
                verticalalignment="top",
                color="r",
            )
            plt.text(
                xlim[0],
                0.9 * ylim[1],
                f"pc={pc[0]}",
                fontsize=fs2 - 2,
                verticalalignment="top",
                color="r",
            )
            plt.text(
                xlim[0],
                0.8 * ylim[1],
                f"rs={ci2[0]}-{ci2[1]}",
                fontsize=fs2 - 2,
                verticalalignment="top",
                color="m",
            )
            plt.text(
                xlim[0],
                0.7 * ylim[1],
                f"pc={pc2[0]}",
                fontsize=fs2 - 2,
                verticalalignment="top",
                color="m",
            )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

    # Bottom histogram
    axB.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 0.5])
    ylimBH = axB.get_ylim()
    axB.set_xticks(xticks)
    axB.set_yticks([])
    axB.set_yticklabels([])
    axB.set_xticklabels([])
    axB.set_xlim(left=xlim[0], right=xlim[1])
    axB.grid()
    axB.invert_yaxis()
    for n, val in enumerate(xticks):
        if val >= xlim[0] and val <= xlim[1]:
            if int(val) == val:
                val = int(val)
            else:
                val = np.round(val, 2)
            if kde_flag is True:
                if (fourcolors_flag is True) or (colorpoints_flag is True):
                    if PrintType != "png":
                        axB.text(
                            val,
                            ylimBH[0],
                            f"{val}",
                            fontsize=fs,
                            horizontalalignment="center",
                            verticalalignment="bottom",
                            color=[0.5, 0.5, 0.5, 0.5],
                        )
                else:
                    axB.text(
                        val,
                        ylimBH[0],
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color=[1, 1, 1, 0.5],
                    )

            else:
                axB.text(
                    val,
                    ylimBH[0],
                    f"{val}",
                    fontsize=fs,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    color=[0.5, 0.5, 0.5, 0.5],
                )

    if PrintType != "png":
        axB.text(
            np.mean(xlim),
            np.mean(ylimBH),
            f"{abbX}",
            fontsize=fs2,
            horizontalalignment="center",
            verticalalignment="center",
        )
    axB.axis("off")

    # Side histogram
    axS.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 0.5], orientation="horizontal")
    xlimSH = axS.get_xlim()
    axS.set_yticks(yticks)
    axS.set_xticks([])
    axS.set_xticklabels([])
    axS.set_yticklabels([])
    axS.set_ylim(bottom=ylim[0], top=ylim[1])
    axS.grid()
    axS.invert_xaxis()
    for n, val in enumerate(yticks):
        if val >= ylim[0] and val <= ylim[1]:
            if int(val) == val:
                val = int(val)
            else:
                val = np.round(val, 2)
            if kde_flag is True:
                if (fourcolors_flag is True) or (colorpoints_flag is True):
                    if PrintType != "png":
                        axS.text(
                            xlimSH[0],
                            val,
                            f"{val}",
                            fontsize=fs,
                            horizontalalignment="left",
                            verticalalignment="center",
                            color=[0.5, 0.5, 0.5, 0.5],
                        )
                else:
                    axS.text(
                        xlimSH[0],
                        val,
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="left",
                        verticalalignment="center",
                        color=[1, 1, 1, 0.5],
                    )
            else:
                axS.text(
                    xlimSH[0],
                    val,
                    f"{val}",
                    fontsize=fs,
                    horizontalalignment="left",
                    verticalalignment="center",
                    color=[0.5, 0.5, 0.5, 0.5],
                )

    if PrintType != "png":
        axS.text(
            np.mean(xlimSH),
            np.mean(ylim),
            f"{abbY}",
            fontsize=fs2,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=90,
        )
    axS.axis("off")


# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result


# %% measurements
w1 = 0.05
w2 = 0.05
h1 = 0.05
h2 = 0.05

xs = 0.1
ys = 0.1
x = 1 - w1 - w2 - xs
y = 1 - h1 - h2 - ys

# %% fontsize
fs = 20

# %% Cell height vs dna_mem_pc1
fig = plt.figure(figsize=(5, 5))
plt.rcParams.update({"font.size": fs})
plt.rcParams["svg.fonttype"] = "none"
axScatter = fig.add_axes([w1 + xs, h1 + ys, x, y])
axScatterB = fig.add_axes([w1 + xs, h1, x, ys])
axScatterS = fig.add_axes([w1, h1 + ys, xs, y])

plotname = "test"
ps = data_root / statsIN / "cell_nuc_PCA_metrics"
scatter(
    axScatter,
    axScatterB,
    axScatterS,
    "DNA_MEM_PC1",
    "Cell height",
    "Shape mode 1",
    "Cell height",
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    fs2=fs,
    fs=fs,
    typ=["pca", "height"],
    PrintType="png",
)

if save_flag == 1:
    plot_save_path = pic_root / f"Cell height vs dna_mem_pc1_20201216.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    # plot_save_path = pic_root / f"Cell height vs dna_mem_pc1_20201216.svg"
    # plt.savefig(plot_save_path, format="svg")
    plt.close()
else:
    plt.show()

# %% cell volume vs dna_mem_pc2
fig = plt.figure(figsize=(5, 5))
plt.rcParams.update({"font.size": fs})
plt.rcParams["svg.fonttype"] = "none"
axScatter = fig.add_axes([w1 + xs, h1 + ys, x, y])
axScatterB = fig.add_axes([w1 + xs, h1, x, ys])
axScatterS = fig.add_axes([w1, h1 + ys, xs, y])

plotname = "test"
ps = data_root / statsIN / "cell_nuc_PCA_metrics"
scatter(
    axScatter,
    axScatterB,
    axScatterS,
    "DNA_MEM_PC2",
    "Cell volume",
    "Shape Mode 2",
    "Cell volume",
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    fs2=fs,
    fs=fs,
    typ=["pca", "vol"],
    PrintType="png",
)

if save_flag == 1:
    plot_save_path = pic_root / f"Cell volume vs dna_mem_pc2_20201216.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    # plot_save_path = pic_root / f"Cell volume vs dna_mem_pc2_20201216.svg"
    # plt.savefig(plot_save_path, format="svg")
    plt.close()
else:
    plt.show()
