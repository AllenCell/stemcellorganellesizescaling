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
if platform.system() == "Windows":
    data_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Oct2021")
    pic_root = Path("Z:/modeling/theok/Projects/Data/scoss/Pics/Oct2021")
elif platform.system() == "Linux":
    1 / 0
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

# Load dataset
tableIN = "SizeScaling_20211101.csv"
statsIN = "Stats_20211101"
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())

# %% Parameters, updated directories
save_flag = 1  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "plot_cellmetrics_vs_PCA"
pic_root.mkdir(exist_ok=True)
pt = 'All'

# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result

# %% Plotting function
def scatterN(
    ax,
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
    PrintType=pt,
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

    if metricX == metricY:

        # data
        x = cells[metricX]
        x = x / facX
        xlim = [np.min(x), np.max(x)]

        # get xticks
        qqq= ax.scatter(x, x)
        ax.set_xlim(left=xlim[0],right=xlim[1])
        xticks = ax.get_xticks()
        # print(metricX)
        # print(xticks)
        qqq.remove()

        # Histogram
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 0.5])
        ylimBH = ax.get_ylim()
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.set_ylim(bottom=ylimBH[0], top=ylimBH[1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid()

        if PrintType != "png":
            s = abbX.split()
            ax.text(
                np.mean(xlim),
                ylimBH[0]+.6*np.diff(ylimBH),
                f"{s[0]}",
                fontsize=fs2,
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.text(
                np.mean(xlim),
                ylimBH[0]+.4*np.diff(ylimBH),
                f"{s[1]} {s[2]}",
                fontsize=fs2,
                horizontalalignment="center",
                verticalalignment="center",
                )

    else:

        # data
        x = cells[metricX]
        y = cells[metricY]
        x = x / facX
        y = y / facY
        xlim = [np.min(x), np.max(x)]
        ylim = [np.min(y), np.max(y)]

        # get ticks
        qqq= ax.scatter(x, y)
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        # print(metricX)
        # print(xticks)
        # print(metricY)
        # print(yticks)
        qqq.remove()

        # plot
        if kde_flag is True:
            try:
                xii = loadps(stats_root, f"{metricX}_{metricY}_xii") / facX
                yii = loadps(stats_root, f"{metricX}_{metricY}_yii") / facY
                zii = loadps(stats_root, f"{metricX}_{metricY}_zii")
                cii = loadps(stats_root, f"{metricX}_{metricY}_cell_dens")
                ax.set_ylim(top=ylim[1])
                ax.set_ylim(bottom=ylim[0])
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
            except:
                rollingavg_flag = False
                ols_flag = False
                ax.plot(x, y, "b.", markersize=ms)
        else:
            ax.plot(x, y, "b.", markersize=ms)


        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

        if PrintType == "svg":
            qqq.remove()
        if PrintType != "png":
            ax.grid()

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
                            xlim[0]+0.05*np.diff(xlim),
                            ylim[1]-0.05*np.diff(ylim),
                            # f"rs={cim[0]}, pc={pc[0]}",
                            f"R\u00b2={cim[0]}",
                            fontsize=fs + 2,
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

        #Tick numbers
        for n, val in enumerate(xticks):
            if val >= xlim[0] and val <= xlim[1]:
                if int(val) == val:
                    val = int(val)
                else:
                    val = np.round(val, 2)
                if kde_flag is True:
                    if (fourcolors_flag is True) or (colorpoints_flag is True):
                        if PrintType != "png":
                            ax.text(
                                val,
                                ylim[0],
                                f"{val}",
                                fontsize=fs,
                                horizontalalignment="center",
                                verticalalignment="bottom",
                                color=[0.5, 0.5, 0.5, 0.5],
                            )
                    else:
                        ax.text(
                            val,
                            ylim[0],
                            f"{val}",
                            fontsize=fs,
                            horizontalalignment="center",
                            verticalalignment="bottom",
                            color=[1, 1, 1, 0.5],
                        )

                else:
                    ax.text(
                        val,
                        ylim[0],
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color=[0.5, 0.5, 0.5, 0.5],
                    )

        for n, val in enumerate(yticks):
            if val >= ylim[0] and val <= ylim[1]:
                if int(val) == val:
                    val = int(val)
                else:
                    val = np.round(val, 2)
                if kde_flag is True:
                    if (fourcolors_flag is True) or (colorpoints_flag is True):
                        if PrintType != "png":
                            ax.text(
                                xlim[0],
                                val,
                                f"{val}",
                                fontsize=fs,
                                horizontalalignment="left",
                                verticalalignment="center",
                                color=[0.5, 0.5, 0.5, 0.5],
                            )
                    else:
                        ax.text(
                            xlim[0],
                            val,
                            f"{val}",
                            fontsize=fs,
                            horizontalalignment="left",
                            verticalalignment="center",
                            color=[1, 1, 1, 0.5],
                        )
                else:
                    ax.text(
                        xlim[0],
                        val,
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="left",
                        verticalalignment="center",
                        color=[0.5, 0.5, 0.5, 0.5],
                    )




# %% Feature definitions
name = 'test2'
fs = 12
fs2 = 15
FS = {}
FS["ShapeModes"] = ["Cell volume",'Cell height',"NUC_MEM_PC1","NUC_MEM_PC2","NUC_MEM_PC3","NUC_MEM_PC4","NUC_MEM_PC5","NUC_MEM_PC6","NUC_MEM_PC7","NUC_MEM_PC8"]
FS["Shape Mode Names"] = ["Cell Volume",'Cell Height','Shape Mode 1','Shape Mode 2','Shape Mode 3','Shape Mode 4','Shape Mode 5','Shape Mode 6','Shape Mode 7','Shape Mode 8']
FS["Shape Mode Types"] = ['vol','height','pca','pca','pca','pca','pca','pca','pca','pca']
# FS["ShapeModes"] = ["Cell volume","NUC_MEM_PC1","NUC_MEM_PC2"]
# FS["Shape Mode Names"] = ["Cell Volume",'Shape Mode 1','Shape Mode 2']
# FS["Shape Mode Types"] = ['vol','pca','pca']
ps = data_root / statsIN / "cell_pca_metrics"

fig = plt.figure(figsize=(16, 16))
plt.rcParams.update({"font.size": fs})
plt.rcParams["svg.fonttype"] = "none"

ncols = len(FS['ShapeModes'])
nrows = len(FS["ShapeModes"])

w1 = 0
w2 = 0
w3 = 0
h1 = 0
h2 = 0
h3 = 0
xp = 0
yp = 0
xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
xw = xx * xp
xx = xx * (1 - xp)
yw = yy * yp
yy = yy * (1 - yp)

i = 0

for col_i, (c_feat, c_name, c_type) in enumerate(zip(FS["ShapeModes"], FS["Shape Mode Names"],FS["Shape Mode Types"])):
    for row_i, (r_feat, r_name, r_type) in enumerate(zip(FS["ShapeModes"], FS["Shape Mode Names"],FS["Shape Mode Types"])):

        # select subplot
        i = i + 1
        row = nrows - np.ceil(i / ncols) + 1
        row = row.astype(np.int64)
        col = i % ncols
        if col == 0:
            col = ncols
        print(f"{i}_{row}_{col}")

        # Main scatterplot
        axScatter = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xx,
                yy,
                ]
        )

        scatterN(
            axScatter,
            r_feat,
            c_feat,
            r_name,
            c_name,
            cells,
            ps,
            kde_flag=True,
            fourcolors_flag=False,
            colorpoints_flag=True,
            rollingavg_flag=True,
            ols_flag=True,
            N2=1000,
            fs2=fs2,
            fs=fs,
            typ=[r_type, c_type],
            PrintType="all",
        )

if save_flag:
    plot_save_path = pic_root / f"{name}.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

