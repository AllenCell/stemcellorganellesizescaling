#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
from scipy.stats import ranksums
import sys, importlib
from tqdm import tqdm
from decimal import Decimal
from scipy.stats.stats import pearsonr

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################
#%% Directories
if platform.system() == "Windows":
    edge_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Subsample_Oct2021/edge")
    nonedge_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Subsample_Oct2021/non-edge")
    all_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Oct2021")
    pic_root = Path("Z:/modeling/theok/Projects/Data/scoss/Pics/Oct2021")
elif platform.system() == "Linux":
    1 / 0

pic_rootT = pic_root / "edgeviolins"
pic_rootT.mkdir(exist_ok=True)

# %% Resolve directories and load data
tableIN = "SizeScaling_20211101.csv"
# Load dataset
cells = pd.read_csv(all_root / tableIN)
print(np.any(cells.isnull()),len(cells))
ecells = pd.read_csv(edge_root / tableIN)
print(np.any(ecells.isnull()),len(ecells))
ncells = pd.read_csv(nonedge_root / tableIN)
print(np.any(ncells.isnull()),len(ncells))

# Remove outliers
# %% Parameters, updated directories
save_flag = 1  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 8})
plt.rcParams["svg.fonttype"] = "none"


# %% Annotation
structures = pd.read_csv(pic_rootT / "structure_annotated_20220217.csv")
print(len(structures))
nrows = 3
ncols = 6

# %% Parametrization
xv = "Cell volume"
yv = "Structure volume"
pt = 0.01
et = 0.1
from matplotlib.ticker import ScalarFormatter
sf = ScalarFormatter()
sf.set_scientific(True)
sf.set_powerlimits((-2, 2))

#%% Time for a flexible scatterplot
plt.rcParams.update({"font.size": 8})
w1 = 0.05
w2 = 0.05
w3 = 0.01
h1 = 0.05
h2 = 0.08
h3 = 0.03
xp = 0
yp = 0
xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
xw = xx * xp
xx = xx * (1 - xp)
yw = yy * yp
yy = yy * (1 - yp)

fig = plt.figure(figsize=(16, 9))

i = -1
for j in np.arange(ncols*nrows):
    i = i + 1

    # select subplot
    row = nrows - np.ceil((i + 1) / ncols) + 1
    row = row.astype(np.int64)
    col = (i + 1) % ncols
    if col == 0:
        col = ncols
    # col = col.astype(np.int64)
    print(f"{i}_{row}_{col}")

    # Main scatterplot
    ax = fig.add_axes(
        [
            w1 + ((col - 1) * (xw + xx + w2)) + xw,
            h1 + ((row - 1) * (yw + yy + h2)) + yw,
            xx,
            yy,
            ]
    )

    gene = structures.iloc[col-1]['Gene']

    if row==1:
        x = (
            cells.loc[cells["structure_name"] == gene, xv]
                .squeeze()
                .to_numpy()
            )
        y = (
            cells.loc[cells["structure_name"] == gene, yv]
                .squeeze()
                .to_numpy()
        )
        r,_ = pearsonr(x, y)
        t = f'All cells (n={len(x)}, r={r:.2f})'
        c = 'blue'
    elif row==2:
        x = (
            ncells.loc[ncells["structure_name"] == gene, xv]
                .squeeze()
                .to_numpy()
        )
        y = (
            ncells.loc[ncells["structure_name"] == gene, yv]
                .squeeze()
                .to_numpy()
        )
        r,_ = pearsonr(x, y)
        t = f'Non-edge cells (n={len(x)}, r={r:.2f})'
        c = 'orange'
    elif row==3:
        x = (
            ecells.loc[ecells["structure_name"] == gene, xv]
                .squeeze()
                .to_numpy()
        )
        y = (
            ecells.loc[ecells["structure_name"] == gene, yv]
                .squeeze()
                .to_numpy()
        )
        if len(x)>0:
            r,_ = pearsonr(x, y)
            t = f'Edge cells (n={len(x)}, r={r:.2f})'
        else:
            t = f'Edge cells (n={len(x)})'
        c = 'green'

    ax.scatter(x,y,10,color=c,alpha=0.25)
    ax.set_title(t)
    ax.set_xlabel(xv)
    ax.set_ylabel(f'{gene} {yv}')
    ax.yaxis.set_major_formatter(sf)
    ax.ticklabel_format(axis='y',style='sci')
    # if row==1:
    #     xlim = ax.get_xlim()
    #     ylim = ax.get_ylim()
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.grid()

if save_flag == 1:
    plot_save_path = pic_rootT / f"SizeScalingMetrics_CellVolume_vs_StructVolume.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()





