#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm
from sklearn.utils import resample
import datetime
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy.stats import pearsonr
from matplotlib import cm
import pickle
from datetime import datetime
import seaborn as sns
import os, platform
import sys, importlib

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    fscatter,
)

importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.scatter_plotting_func"]
)
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    fscatter,
)


print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################
#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020/")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]
pic_rootT = pic_root / "pca"
pic_rootT.mkdir(exist_ok=True)

tableIN = "SizeScaling_20201102.csv"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())
save_flag = 0

# %% Time vs. structure
timestr = cells["ImageDate"]
time = np.zeros((len(timestr), 1))
for i, val in tqdm(enumerate(timestr)):
    date_time_obj = datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
    # time[i] = int(date_time_obj.strftime("%Y%m%d%H%M%S"))
    time[i] = int(date_time_obj.timestamp())
cells["int_acquisition_time"] = time

# %% Order of structures and FOVs
table = pd.pivot_table(
    cells, index="structure_name", values="int_acquisition_time", aggfunc="mean"
)
table = table.sort_values(by=["int_acquisition_time"])
sortedStructures = table.index.values

# %% Selecting number of pairs
pairs = ['DNA_MEM_PC1', 'DNA_MEM_PC2',
       'DNA_MEM_PC3', 'DNA_MEM_PC4', 'DNA_MEM_PC5', 'DNA_MEM_PC6',
       'DNA_MEM_PC7', 'DNA_MEM_PC8']
no_of_pairs = len(pairs)
nrows = np.floor(np.sqrt(2 / 3 * no_of_pairs))
if nrows == 0:
    nrows = 1
ncols = np.floor(nrows * 3 / 2)
while nrows * ncols < no_of_pairs:
    ncols += 1

# %% Plotting parameters
ms = 0.5
fs2 = np.round(np.interp(nrows * ncols, [6, 21, 50], [25, 12, 8]))
fs = np.round(fs2 * 2 / 3)
lw2 = 1.5
plt.rcParams.update({"font.size": fs})

# %% Time for a flexible scatterplot
w1 = 0.05
w2 = 0.05
w3 = 0.05
h1 = 0.05
h2 = 0.1
h3 = 0.05
xp = 0
yp = 0
xx = (1-w1-((ncols-1)*w2)-w3)/ncols
yy = (1-h1-((nrows-1)*h2)-h3)/nrows
xw = xx*xp
xx = xx*(1-xp)
yw = yy*yp
yy = yy*(1-yp)

fig = plt.figure(figsize=(16, 9))

for i, xy_pair in enumerate(pairs):

    # data
    x = cells[xy_pair]

    # select subplot
    row = nrows - np.ceil((i + 1) / ncols) + 1
    row = row.astype(np.int64)
    col = (i + 1) % ncols
    if col == 0: col = ncols
    col = col.astype(np.int64)
    print(f"{i}_{row}_{col}")

    # Main scatterplot
    ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)) + xw, h1 + ((row - 1) * (yw + yy + h2)) + yw, xx, yy])
    ax.hist(x,bins=100)
    ax.set_xlim((-40, 40))
    ylim = ax.get_ylim()
    ax.grid()
    ax.set_title(xy_pair)
    xp = np.round(np.percentile(x,[1, 25, 75, 99]))
    ax.text(-40,.1*ylim[1],xp,color='k')
    ax.text(-40, .2 * ylim[1], [1, 25, 75, 99], color='k')


if save_flag:
    plot_save_path = pic_rootT / f"HISTofPCAcoeff.png"
    plt.savefig(plot_save_path, format="png", dpi=1000)
    plt.close()
else:
    plt.show()

# %%
pcanum = [1,2,3,4,5,6,7,8]
pcatext = 'DNA_MEM_PC'
xrange = [1, 99]
zrange = [1, 99]
nbins = 7

for i,pc in enumerate(pcanum):
    cells[f"{pcatext}bin{pc}"] = np.nan
    x = cells[f"{pcatext}{pc}"].values
    XR = np.percentile(x,xrange)
    y = np.digitize(x, np.linspace(XR[0], XR[1], nbins+1)[:-1]).astype(np.float)
    y[x < XR[0]] = np.nan
    y[x > XR[1]] = np.nan
    for j, pcz in enumerate(pcanum):
        if j is not i:
            z = cells[f"{pcatext}{pcz}"].values
            ZR = np.percentile(z, zrange)
            y[z < ZR[0]] = np.nan
            y[z > ZR[1]] = np.nan
    cells[f"{pcatext}bin{pc}"] = pd.Series(y,dtype='category')


# %%
pcanum = [1,2,3,4,5,6,7,8]
bins = np.arange(nbins)+1

CountArray = np.zeros((len(pcanum),nbins))
pclabels = []
binlabels = []
for i, pc in enumerate(pcanum):
    pclabels.append(f"DNA_MEM_PC{pc}")
    for j,bin in enumerate(bins):
        if i == 0:
            binlabels.append(f"Bin {bin}")
        CountArray[i,j] = np.sum(cells[f"{pcatext}bin{pc}"] == bin)

# %%

fig, ax = plt.subplots(figsize=(16, 9))
im = ax.imshow(CountArray)
#
# We want to show all ticks...
ax.set_xticks(np.arange(len(binlabels)))
ax.set_yticks(np.arange(len(pclabels)))
# ... and label them with the respective list entries
ax.set_xticklabels(binlabels)
ax.set_yticklabels(pclabels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(pclabels)):
    for j in range(len(binlabels)):
        text = ax.text(j, i, np.int(np.round(CountArray[i, j])),
                       ha="center", va="center", color="w")

fig.tight_layout()
if save_flag:
    plot_save_path = pic_rootT / f"HEATMAPofCounts.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

# %%
plt.rcParams.update({"font.size": 10})
selected_metrics = ['Cell surface area', 'Cell volume', 'Cell height',
                    'Nuclear surface area', 'Nuclear volume', 'Nucleus height',
                    'Cytoplasmic volume', 'Structure volume', 'Number of pieces']
for j, metric in enumerate(selected_metrics):

    fig = plt.figure(figsize=(16, 9))

    for i, pc in enumerate(pcanum):
        y = f"{pcatext}bin{pc}"

        # select subplot
        row = nrows - np.ceil((i + 1) / ncols) + 1
        row = row.astype(np.int64)
        col = (i + 1) % ncols
        if col == 0: col = ncols
        col = col.astype(np.int64)
        print(f"{i}_{row}_{col}")

        # Main violinplot
        ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)) + xw, h1 + ((row - 1) * (yw + yy + h2)) + yw, xx, yy])
        sns.violinplot(
            y=y, x=metric, color='orange', data=cells, scale='width', ax=ax
        )

        ax.set_title(f"{metric} across PC {pc}")
        ax.grid(True, which="major", axis="both")
        ax.set_axisbelow(True)
        ax.set_ylabel(None)
        ax.set_xlabel(None)

    if save_flag:
        plot_save_path = pic_rootT / f"VIOLIN_{metric}acrossPCAcomps.png"
        plt.savefig(plot_save_path, format="png", dpi=300)
        plt.close()
    else:
        plt.show()

# %% Heatmap of correlations between
structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201019.csv")
sortedStructures = structures['Gene'].to_numpy()


plt.rcParams.update({"font.size": 7})
fs = 10
FS={}
xvec = []
yvec = []
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
FS['struct_metrics'] = [
        "Structure volume",
        "Number of pieces",
        "Piece average",
        "Piece std",
    ]
# FS['struct_metrics'] = [
#         "Structure volume",
#     ]

X = cells[FS["cellnuc_metrics"]].to_numpy()
Y = cells[FS["pca_components"]].to_numpy()
CM = np.zeros((X.shape[1],Y.shape[1]))
for x in np.arange(X.shape[1]):
    for y in np.arange(Y.shape[1]):
        CM[x,y],_ = pearsonr(X[:,x],Y[:,y])
CMT = np.zeros((len(sortedStructures)+len(FS["cellnuc_metrics"]),len(FS["pca_components"])))
CMT[0:len(FS["cellnuc_metrics"]),:] = CM

for m, metric in enumerate(FS['struct_metrics']):
    print(metric)
    ylabels = FS["cellnuc_metrics"].copy()
    for s, struct in enumerate(sortedStructures):
        X = cells.loc[cells['structure_name']==struct,metric].to_numpy()
        Y = cells.loc[cells['structure_name'] == struct, FS["pca_components"]].to_numpy()
        for y in np.arange(Y.shape[1]):
            CMT[s+len(FS["cellnuc_metrics"]), y], _ = pearsonr(X, Y[:, y])
        ylabels.append(f"{metric}_{struct}")

    fig, ax = plt.subplots(figsize=(12, 8))

    xlabels = FS["pca_components"]
    ax.imshow(CMT, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
    for i in range(CMT.shape[0]):
        for j in range(CMT.shape[1]):
            val = CMT[i, j]
            if abs(val) > .7:
                text = ax.text(j, i, np.round(CMT[i, j],2),
                                  ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
            elif abs(val) > .1:
                text = ax.text(j, i, np.round(CMT[i, j],2),
                                  ha="center", va="center", color="k", fontsize=fs, fontweight='bold')

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)

    if save_flag:
        plot_save_path = pic_rootT / f"HEATMAP_{metric}VSPCAcomps.png"
        plt.savefig(plot_save_path, format="png", dpi=300)
        plt.close()
    else:
        plt.show()


# %%
from scipy.stats import chi2_contingency

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

# %% Heatmaps of PCA components themselves
Y = cells[FS["pca_components"]].to_numpy()
CM = np.zeros((Y.shape[1],Y.shape[1]))
for x in np.arange(Y.shape[1]):
    for y in np.arange(Y.shape[1]):
        CM[x,y],_ = pearsonr(Y[:,x],Y[:,y])

fig, ax = plt.subplots(figsize=(12, 8))

xlabels = FS["pca_components"]
ylabels = FS["pca_components"]
ax.imshow(CM, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
for i in range(CM.shape[0]):
    for j in range(CM.shape[1]):
        val = CM[i, j]
        if abs(val) > .7:
            text = ax.text(j, i, np.round(CM[i, j],2),
                              ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
        elif abs(val) > 0:
            text = ax.text(j, i, np.round(CM[i, j],2),
                              ha="center", va="center", color="k", fontsize=fs, fontweight='bold')

ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels)
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)

if save_flag:
    plot_save_path = pic_rootT / f"HEATMAP_PCAcompsVSPCAcomps_PC.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

# %% Heatmaps of PCA components themselves
bins = 20
xrange = [2, 98]
Y = cells[FS["pca_components"]].to_numpy()
CM = np.zeros((Y.shape[1],Y.shape[1]))
for x in np.arange(Y.shape[1]):
    for y in np.arange(Y.shape[1]):
        v1 = Y[:,y]
        v2 = Y[:,x]
        vp = np.zeros(v1.shape)
        v1r = np.percentile(v1, xrange)
        v2r = np.percentile(v2, xrange)
        vp[v1 > v1r[1]] = np.nan
        vp[v1 < v1r[0]] = np.nan
        vp[v2 > v2r[1]] = np.nan
        vp[v2 < v2r[0]] = np.nan
        v1p = np.delete(v1,np.argwhere(np.isnan(vp)))
        v2p = np.delete(v2, np.argwhere(np.isnan(vp)))
        CM[x,y] = calc_MI(v1,v2,bins)

fig, ax = plt.subplots(figsize=(12, 8))

xlabels = FS["pca_components"]
ylabels = FS["pca_components"]
ax.imshow(CM, aspect='auto', cmap='seismic', vmin=-1, vmax=1)
for i in range(CM.shape[0]):
    for j in range(CM.shape[1]):
        val = CM[i, j]
        if abs(val) > .7:
            text = ax.text(j, i, np.round(CM[i, j],2),
                              ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
        elif abs(val) > 0:
            text = ax.text(j, i, np.round(CM[i, j],2),
                              ha="center", va="center", color="k", fontsize=fs, fontweight='bold')

ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels)
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels)

if save_flag:
    plot_save_path = pic_rootT / f"HEATMAP_PCAcompsVSPCAcomps_MI.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

# %% Heatmaps of PCA components themselves

from mpl_toolkits.axes_grid1 import Grid
nrows = len(FS["pca_components"])
ncols = len(FS["pca_components"])

plt.close('all')

fig = plt.figure(figsize=(15, 9))
grid = Grid(fig, rect=111, nrows_ncols=(nrows, ncols),
            axes_pad=0.05, label_mode='L',
            )

for i, ax in enumerate(grid):
    row = nrows - np.ceil((i + 1) / ncols) + 1
    row = row.astype(np.int64)
    col = (i + 1) % ncols
    if col == 0:
        col = ncols
    # col = col.astype(np.int64)
    ax.plot(Y[:,row-1],Y[:,col-1],'b.',markersize=.5)
    ax.text(0,0,f"{row} vs {col}",fontsize=20,horizontalalignment='center',color='red')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
ax.title.set_visible(False)

plt.tight_layout()
if save_flag:
    plot_save_path = pic_rootT / f"SCATTER_PCAcompsVSPCAcomps.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

# %%
