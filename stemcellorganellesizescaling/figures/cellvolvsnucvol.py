#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
import sys, importlib
from skimage.morphology import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
import vtk
from aicsshparam import shtools
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import pearsonr
import pickle
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pickle
import locale

locale.setlocale(locale.LC_ALL, "")
from scipy import interpolate

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################
#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020")
elif platform.system() == "Linux":
    1/0

pic_rootT = pic_root / "hp"
pic_rootT.mkdir(exist_ok=True)

# %% Resolve directories and load data
tableIN = "SizeScaling_20201102.csv"
statsIN = "Stats_20201102"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())

# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
plt.rcParams['svg.fonttype'] = 'none'

# %% Feature sets
FS={}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    "Cytoplasmic volume",
]


FS["pca_components"] = ['DNA_MEM_PC1', 'DNA_MEM_PC2',
       'DNA_MEM_PC3', 'DNA_MEM_PC4', 'DNA_MEM_PC5', 'DNA_MEM_PC6',
       'DNA_MEM_PC7', 'DNA_MEM_PC8']

FS["pca_abbs"] = ['sm1', 'sm2',
       'sm3', 'sm4', 'sm5', 'sm6',
       'sm7', 'sm8']

FS['struct_metrics'] = [
        "Structure volume",
    ]

# %% Preparation of PCA

# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result

# %% measurements
w1 = 0.1
w2 = 0.1
xw = 0.05
x = 1-w1-w2-xw

h1 = 0.01
h2 = 0.01
yh = 0.075
y = 1-h1-h2-yh




# %%layout
fs = 13
fs2 = 20
fsP = 20
fig = plt.figure(figsize=(16, 9),facecolor='black')
plt.rcParams.update({"font.size": fs})


# Nuc
axNuc = fig.add_axes([w1+xw, h1+yh, x, y])
# Nuc side
axNucS = fig.add_axes([w1, h1+yh, xw, y])
# Nuc bottom
axNucB = fig.add_axes([w1+xw, h1, x, yh])
ps = data_root / statsIN / "cellnuc_struct_metrics"
ps = data_root / statsIN / "cell_nuc_metrics"
bscatter(
    axNuc,
    axNucB,
    axNucS,
    FS['cellnuc_metrics'][1],
    FS['cellnuc_metrics'][4],
    FS['cellnuc_metrics'][1],
    FS['cellnuc_metrics'][4],
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=False,
    ols_flag=False,
    N2=1000,
    fs2=fs2,
    fs=fs,
    cell_doubling = [],
    typ=['vol','vol'],
    PrintType='all',
)

plt.show()


# %%
def bscatter(
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
    cell_doubling=[],
    typ=["vol", "vol"],
    PrintType="all",
):

    #%% Change labels
    if typ[0] == "vol":
        abbX = f"{abbX} (\u03BCm\u00b3)"
    elif typ[0] == "area":
        abbX = f"{abbX} (\u03BCm\u00b2)"
    if typ[1] == "vol":
        abbY = f"{abbY} (\u03BCm\u00b3)"
    elif typ[1] == "area":
        abbY = f"{abbY} (\u03BCm\u00b2)"
    if typ[0] == "vol":
        facX = 1 / ((0.108333) ** 3)
    elif typ[0] == "area":
        facX = 1 / ((0.108333) ** 2)
    else:
        facX = 1000
    if typ[1] == "vol":
        facY = 1 / ((0.108333) ** 3)
    elif typ[1] == "area":
        facY = 1 / ((0.108333) ** 2)
    else:
        facY = 1000

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
    cpmap = cpmap(np.linspace(0, 1, 1000) ** 0.4)
    nupo = 10
    uv = np.linspace(0.5, 0, nupo)
    cpmap[0:nupo, 0] = np.minimum(cpmap[0:nupo, 0]+uv,0.5)
    cpmap[0:nupo, 1] = np.minimum(cpmap[0:nupo, 0]+uv,1)
    cpmap[0:nupo, 2] = np.minimum(cpmap[0:nupo, 0]+uv,1)
    cpmap = ListedColormap(cpmap)
    darkgreen = [0.0, 0.26666667, 0.10588235, 1.0]
    darkgreen_t = [0.0, 0.26666667, 0.10588235, 0.5]

    #%% Plotting parameters
    ms = 0.5
    lw2 = 2
    nbins = 100
    offwhite = [0.7, 0.7, 0.7, 1]
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
            ax.set_facecolor('k')
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
        qqq = ax.plot(x, y, "b.", markersize=ms)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if PrintType == "svg":
        try:
            qqq.remove()
        except:
            qqqq = qqq.pop(0)
            qqqq.remove()
            ax.plot(0, 0, "b.", markersize=ms)
    # if PrintType != "png":
    #     ax.grid()

    # ax.text(xlim[0],ylim[1],f"{abbX} vs {abbY}",fontsize=fs2, verticalalignment = 'top')
    if kde_flag is True:
        if (fourcolors_flag is True) or (colorpoints_flag is True):
            if PrintType != "png":
                ax.text(
                    -0.02 * (xlim[1] - xlim[0]) + xlim[1],
                    -0.02 * (ylim[1] - ylim[0]) + ylim[1],
                    f"n= {len(x):n}",
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
        if PrintType != "png":
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
                    if len(cell_doubling) > 0:
                        cd0 = cell_doubling[0] / facX
                        cd1 = cell_doubling[1] / facX
                        f = interpolate.interp1d(xii[:, 0], pred_yL)
                        y0 = f(cd0)
                        y1 = f(cd1)
                        ax.plot(
                            [xlim[0], cd1 + 1950],
                            [y1, y1],
                            color=darkgreen,
                            linewidth=1,
                            linestyle="dashdot",
                        )
                        ax.plot(
                            [xlim[0], cd0 + 1950],
                            [y0, y0],
                            color=darkgreen,
                            linewidth=1,
                            linestyle="dashdot",
                        )
                        ax.plot([cd0, cd0], [ylim[0], y0], color=darkgreen, linewidth=2)
                        ax.plot([cd1, cd1], [ylim[0], y1], color=darkgreen, linewidth=2)
                        ax.plot([xlim[0], cd0], [y0, y0], color=darkgreen, linewidth=2)
                        ax.plot([xlim[0], cd1], [y1, y1], color=darkgreen, linewidth=2)
                        ax.text(
                            cd0 + 2000,
                            y0,
                            f"{int(np.round(y0))} \u03BCm\u00b3",
                            color=darkgreen,
                            verticalalignment="center_baseline",
                        )
                        ax.text(
                            cd1 + 2000,
                            y1,
                            f"{int(np.floor(y1))} \u03BCm\u00b3",
                            color=darkgreen,
                            verticalalignment="center_baseline",
                        )
                        ax.text(
                            (cd1 + cd0) / 2 + 2200,
                            (y1 + y0) / 2,
                            f"{int(np.floor(100*(y1-y0)/y0))}% increase",
                            color=darkgreen,
                            verticalalignment="center_baseline",
                        )
                        y0a = f(cd0 + 200)
                        y1a = f(cd1 - 200)
                        x0a = cd0 + 200 + 2000
                        x1a = cd1 - 200 + 2000

                        ax.arrow(
                            x0a[0],
                            y0a[0],
                            (x1a[0] - x0a[0]),
                            (y1a[0] - y0a[0]),
                            color=darkgreen,
                            width=10,
                            length_includes_head=True,
                            head_width=50,
                            head_length=30,
                        )
                        # ax.arrow(1000, 1000, 100, 100)

                        # f"{int(y1)} \u03BCm\u00b3", fontsize = fs)
                        # ax.text([cd0 + 1600], y0, f"{int(y0)} \u03BCm\u00b3", fontsize=fs)
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
                        0.02 * (xlim[1] - xlim[0]) + xlim[0],
                        -0.02 * (ylim[1] - ylim[0]) + ylim[1],
                        f" R\u00b2={cim[0]} (Expl. var. is {int(100*cim[0])}%)",
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
    _, bine, _ = axB.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 0.5])
    if len(cell_doubling) > 0:
        pos = (
            np.argwhere(
                np.all(
                    np.concatenate(
                        (
                            np.expand_dims(x, axis=0) >= cell_doubling[0] / facX,
                            np.expand_dims(x, axis=0) <= cell_doubling[1] / facX,
                        ),
                        axis=0,
                    ),
                    axis=0,
                )
            )
            .astype(np.int)
            .squeeze()
        )
        axB.hist(x[pos], bins=bine, color=darkgreen_t)
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

    if len(cell_doubling) > 0:
        xpos = xlim[0] + 0.75 * (xlim[1] - xlim[0])
    else:
        xpos = np.mean(xlim)
    axB.text(
        xpos,
        np.mean(ylimBH),
        f"{abbX}",
        fontsize=fs2,
        horizontalalignment="center",
        verticalalignment="center",
        color = offwhite
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

    axS.text(
        np.mean(xlimSH),
        np.mean(ylim),
        f"{abbY}",
        fontsize=fs2,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        color = offwhite
    )
    axS.axis("off")



# %%
