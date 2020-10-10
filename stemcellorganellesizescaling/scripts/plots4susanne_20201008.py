#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample
import datetime
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from matplotlib import cm
import pickle
import os, platform

# Third party

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################
#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/")
    pic_root = Path("E:/DA/Data/scoss/Pics/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# %%
data_root = dirs[0]
pic_root = dirs[1]

dataset = "SizeScaling_20201006_clean.csv"
dataset_comp = "SizeScaling_20201006_comp.csv"
statsOUTdir = "Stats_20201006"

# Load datasets
cells = pd.read_csv(data_root / dataset)
print(np.any(cells.isnull()))
save_flag = 0

# %% Parameters, updated directories
save_flag = 1  # save plot (1) or show on screen (0)
pic_root = pic_root / "plots4susanne_20201008"
pic_root.mkdir(exist_ok=True)


# %% Load a simple function
def load(x):
    with open(x, "rb") as f:
        test = pickle.load(f)
    return test

# %% Load a simple function
def loadps(x):
    with open(data_root / statsOUTdir / 'cell_nuc_metrics' / f"{x}.pickle", "rb") as f:
        test = pickle.load(f)
    return test

# %% Feature sets
cell_metrics = ['Cell surface area', 'Cell volume', 'Cell height']
nuc_metrics = ['Nuclear surface area', 'Nuclear volume', 'Nucleus height']
cellnuc_metrics = ['Cell surface area', 'Cell volume', 'Cell height',
                     'Nuclear surface area', 'Nuclear volume', 'Nucleus height',
                     'Cytoplasmic volume']
cellnuc_abbs = ['Cell area', 'Cell vol', 'Cell height', 'Nuc area', 'Nuc vol', 'Nuc height', 'Cyto vol']
struct_metrics =   ['Structure volume', 'Number of pieces', 'Piece average', 'Piece max', 'Piece min', 'Piece std', 'Piece sum']

# %% plot function
def oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, save_flag, pic_root, name, kde_flag = False, fourcolors_flag = False, rollingavg_flag = False, ols_flag = False, N2 = 1000):

    #%% Selecting number of pairs
    no_of_pairs, _ = pairs.shape
    nrows = np.floor(np.sqrt(2 / 3 *no_of_pairs))
    if nrows == 0:
        nrows = 1
    ncols = np.floor(nrows*3/2)
    while nrows*ncols<no_of_pairs:
        ncols += 1

    #%% Archery new colormap
    white = np.array([1, 1, 1, 1])
    green = np.array([0, 1, 0, 1])
    blue = np.array([0, 0, 1, 1])
    red = np.array([1, 0, 0, 1])
    magenta = np.array([.5, 0, 0.5, 1])
    black = np.array([0,0,0,1])
    newcolors = np.zeros((100,4))
    newcolors[0:10, :] = white
    newcolors[10:90, :] = blue
    newcolors[50:90, :] = red
    newcolors[90:100, :] = magenta
    newcmp = ListedColormap(newcolors)

    #%% Plotting parameters
    fac = 1000
    ms = 0.5
    fs2 = np.round(np.interp(nrows*ncols,[6, 21, 50],[25, 12, 8]))
    fs = np.round(fs2*2/3)
    lw2 = 1.5
    nbins = 100
    plt.rcParams.update({"font.size": fs})

    #%% Plotting flags
    # W = 500

    #%% Time for a flexible scatterplot
    w1 = 0.001
    w2 = 0.01
    w3 = 0.001
    h1 = 0.001
    h2 = 0.01
    h3 = 0.001
    xp = 0.1
    yp = 0.1
    xx = (1-w1-((ncols-1)*w2)-w3)/ncols
    yy = (1-h1-((nrows-1)*h2)-h3)/nrows
    xw = xx*xp
    xx = xx*(1-xp)
    yw = yy*yp
    yy = yy*(1-yp)

    fig = plt.figure(figsize=(16, 9))

    for i, xy_pair in enumerate(pairs):

        print(i)

        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        abbX = cellnuc_abbs[xy_pair[0]]
        abbY = cellnuc_abbs[xy_pair[1]]

        # data
        x = cells[metricX]
        y = cells[metricY]
        x = x / fac
        y = y / fac
        # x = x.sample(n=1000,random_state=86)
        # y = y.sample(n=1000,random_state=86)

        # select subplot
        row = nrows-np.ceil((i+1)/ncols)+1
        row = row.astype(np.int64)
        col = (i+1) % ncols
        if col == 0: col = ncols
        col = col.astype(np.int64)
        print(f"{i}_{row}_{col}")

        # Main scatterplot
        ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)) + xw, h1 + ((row - 1) * (yw + yy + h2)) + yw, xx, yy])

        if kde_flag is True:
            xii = loadps(f"{metricX}_{metricY}_xii")/fac
            yii = loadps(f"{metricX}_{metricY}_yii")/fac
            zii = loadps(f"{metricX}_{metricY}_zii")
            cii = loadps(f"{metricX}_{metricY}_cell_dens")
            ax.set_ylim(top=np.max(yii))
            ax.set_ylim(bottom=np.min(yii))
            if fourcolors_flag is True:
                ax.pcolormesh(xii, yii, zii, cmap=plt.cm.tab20)
            else:
                ax.pcolormesh(xii, yii, zii, cmap=plt.cm.magma)
                sorted_cells = np.argsort(cii)
                # min_cells = sorted_cells[0:N2]
                # min_cells = min_cells.astype(int)
                # ax.plot(x[min_cells], y[min_cells], 'w.', markersize=ms)
                np.random.shuffle(sorted_cells)
                min_cells = sorted_cells[0:N2]
                min_cells = min_cells.astype(int)
                ax.plot(x[min_cells], y[min_cells], 'w.', markersize=ms)
        else:
            ax.plot(x, y, 'b.', markersize=ms)


        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        # ax.text(xlim[0],ylim[1],f"{abbX} vs {abbY}",fontsize=fs2, verticalalignment = 'top')
        if kde_flag is True:
            if fourcolors_flag is True:
                ax.text(xlim[1], ylim[1], f"n= {len(x)}", fontsize=fs, verticalalignment='top', horizontalalignment = 'right', color = 'black')
            else:
                ax.text(xlim[1], ylim[1], f"n= {len(x)}", fontsize=fs, verticalalignment='top', horizontalalignment='right',
                        color='white')

        else:
            ax.text(xlim[1], ylim[1], f"n= {len(x)}", fontsize=fs, verticalalignment='top', horizontalalignment='right')
        if rollingavg_flag is True:
            rollavg_x = loadps(f"{metricX}_{metricY}_x_ra")/fac
            rollavg_y = loadps(f"{metricX}_{metricY}_y_ra")/fac
            ax.plot(rollavg_x, rollavg_y[:, 0], 'lime', linewidth=lw2)

        if ols_flag is True:
            xii = loadps(f"{metricX}_{metricY}_xii") / fac
            pred_yL = loadps(f"{metricX}_{metricY}_pred_matL") / fac
            pred_yC = loadps(f"{metricX}_{metricY}_pred_matC") / fac
            if kde_flag is True:
                if fourcolors_flag is True:
                    ax.plot(xii,pred_yL, 'gray')
                else:
                    ax.plot(xii, pred_yL, 'w')
            else:
                ax.plot(xii, pred_yL, 'r')
                ax.plot(xii, pred_yC, 'm')

        if ols_flag is True:
            val = loadps(f"{metricX}_{metricY}_rs_vecL")
            ci = np.round(np.percentile(val,[2, 98]),2)
            pc = np.round(np.sqrt(np.percentile(val, [50])),2)
            val2 = loadps(f"{metricX}_{metricY}_rs_vecC")
            ci2 = np.round(np.percentile(val2, [2, 98]), 2)
            pc2 = np.round(np.sqrt(np.percentile(val2, [50])), 2)

            if kde_flag is True:
                if fourcolors_flag is True:
                    plt.text(xlim[0], ylim[1], f"rs={ci[0]}-{ci[1]}", fontsize=fs2 - 2, verticalalignment='top',
                             color='gray')
                    plt.text(xlim[0], .9 * ylim[1], f"pc={pc[0]}", fontsize=fs2 - 2, verticalalignment='top',
                             color='gray')
                else:
                    plt.text(xlim[0], ylim[1], f"rs={ci[0]}-{ci[1]}", fontsize=fs2 - 2, verticalalignment='top',
                             color='w')
                    plt.text(xlim[0], .9 * ylim[1], f"pc={pc[0]}", fontsize=fs2 - 2, verticalalignment='top',
                             color='w')
            else:
                plt.text(xlim[0], ylim[1], f"rs={ci[0]}-{ci[1]}", fontsize=fs2 - 2, verticalalignment='top',
                         color='r')
                plt.text(xlim[0], .9 * ylim[1], f"pc={pc[0]}", fontsize=fs2 - 2, verticalalignment='top',
                         color='r')
                plt.text(xlim[0], .8 * ylim[1], f"rs={ci2[0]}-{ci2[1]}", fontsize=fs2 - 2, verticalalignment='top',
                         color='m')
                plt.text(xlim[0], .7 * ylim[1], f"pc={pc2[0]}", fontsize=fs2 - 2, verticalalignment='top',
                         color='m')





        # Bottom histogram
        ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)) + xw, h1 + ((row - 1) * (yw + yy + h2)), xx, yw])
        ax.hist(x, bins = nbins, color = [.5,.5,.5,.5])
        ylimBH = ax.get_ylim()
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.invert_yaxis()
        for n, val in enumerate(xticks):
            if val>=xlim[0] and val<=xlim[1]:
                if int(val)==val:
                    val = int(val)
                else:
                    val = np.round(val,2)
                if kde_flag is True:
                    if fourcolors_flag is True:
                        ax.text(val, ylimBH[0], f"{val}", fontsize=fs, horizontalalignment = 'center', verticalalignment = 'bottom', color = white)
                    else:
                        ax.text(val, ylimBH[0], f"{val}", fontsize=fs, horizontalalignment='center',
                                verticalalignment='bottom', color=[1, 1, 1, .5])

                else:
                    ax.text(val, ylimBH[0], f"{val}", fontsize=fs, horizontalalignment='center', verticalalignment='bottom',
                            color=[.5, .5, .5, .5])

        ax.text(np.mean(xlim), ylimBH[1], f"{abbX}", fontsize=fs2, horizontalalignment='center', verticalalignment='bottom')
        ax.axis('off')

        # Side histogram
        ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)), h1 + ((row - 1) * (yw + yy + h2))+yw, xw, yy])
        ax.hist(y, bins=nbins, color=[.5,.5,.5,.5], orientation='horizontal')
        xlimSH = ax.get_xlim()
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.invert_xaxis()
        for n, val in enumerate(yticks):
            if val >= ylim[0] and val <= ylim[1]:
                if int(val) == val:
                    val = int(val)
                else:
                    val = np.round(val,2)
                if kde_flag is True:
                    if fourcolors_flag is True:
                        ax.text(xlimSH[0], val, f"{val}", fontsize=fs, horizontalalignment='left', verticalalignment='center', color = white)
                    else:
                        ax.text(xlimSH[0], val, f"{val}", fontsize=fs, horizontalalignment='left',
                                verticalalignment='center', color=[1, 1, 1, .5])
                else:
                    ax.text(xlimSH[0], val, f"{val}", fontsize=fs, horizontalalignment='left', verticalalignment='center', color = [.5,.5,.5,.5])

        ax.text(xlimSH[1], np.mean(ylim), f"{abbY}", fontsize=fs2, horizontalalignment='left', verticalalignment='center',rotation=90)
        ax.axis('off')

    if save_flag:
        plot_save_path = pic_root / f"{name}.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

# %%
xvec = [1, 1, 6, 1, 4, 6]
yvec = [4, 6, 4, 0, 3, 3]
pairs = np.stack((xvec, yvec)).T

# %%
N = 13
xvec = np.random.choice(len(cellnuc_metrics),N)
yvec = np.random.choice(len(cellnuc_metrics),N)
pairs = np.stack((xvec, yvec)).T

# %%
L = len(cellnuc_metrics)
pairs = np.zeros((int(L*(L-1)/2),2)).astype(np.int)
i = 0
for f1 in np.arange(L):
    for f2 in np.arange(L):
        if f2>f1:
            pairs[i,:] = [f1, f2]
            i += 1


#%% Plot some
plotname = 't21'
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_plain", kde_flag = False, fourcolors_flag = False, rollingavg_flag = False, ols_flag = False, N2 = 1000)
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_roll", kde_flag = False, fourcolors_flag = False, rollingavg_flag = True, ols_flag = False, N2 = 1000)
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_rollols", kde_flag = False, fourcolors_flag = False, rollingavg_flag = True, ols_flag = True, N2 = 1000)
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_kde", kde_flag = True, fourcolors_flag = False, rollingavg_flag = False, ols_flag = False, N2 = 1000)
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_arch", kde_flag = True, fourcolors_flag = True, rollingavg_flag = False, ols_flag = False, N2 = 1000)
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_archroll", kde_flag = True, fourcolors_flag = True, rollingavg_flag = True, ols_flag = False, N2 = 1000)
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_archrollols", kde_flag = True, fourcolors_flag = True, rollingavg_flag = True, ols_flag = True, N2 = 1000)
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_kderollos", kde_flag = True, fourcolors_flag = False, rollingavg_flag = True, ols_flag = True, N2 = 1000)

#%%
plotname = 't21'
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_archX", kde_flag = True, fourcolors_flag = True, rollingavg_flag = False, ols_flag = False, N2 = 1000)

# %%
xvec = [1]
yvec = [4]
pairs = np.stack((xvec, yvec)).T
plotname = 't21'
oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, False, pic_root, f"{plotname}_archX", kde_flag = True, fourcolors_flag = True, rollingavg_flag = False, ols_flag = False, N2 = 1000)
