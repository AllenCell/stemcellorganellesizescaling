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

# %% Feature sets
FS = {}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
]

FS["struct_metrics"] = [
    "Structure volume",
]

# %% Annotation
structures = pd.read_csv(all_root / "structure_annotated_20201113.csv")
print(len(structures)+len(FS['cellnuc_metrics']))
nrows = 3
ncols = 7

# %% Parametrization
quantiles=[0.25,0.75]
comp4volume = False
cv = "Cell volume"
pt = 0.01
et = 0.1
from matplotlib.ticker import ScalarFormatter
sf = ScalarFormatter()
sf.set_scientific(True)
sf.set_powerlimits((-2, 2))

#%% Time for a flexible scatterplot
w1 = 0.03
w2 = 0.03
w3 = 0
h1 = 0.05
h2 = 0.1
h3 = 0.05
xp = 0
yp = 0
xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
xw = xx * xp
xx = xx * (1 - xp)
yw = yy * yp
yy = yy * (1 - yp)

fig = plt.figure(figsize=(16, 9))
fs2 = 6

i = -1
for j,ncm in enumerate(FS["cellnuc_metrics"]):
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

    if comp4volume is True:
        xc = cells[ncm].values/cells[cv].values
        nc = ncells[ncm].values/ncells[cv].values
        ec = ecells[ncm].values/ecells[cv].values
    else:
        xc = cells[ncm].values
        nc = ncells[ncm].values
        ec = ecells[ncm].values

    ax.violinplot(xc,[0],widths=1,showmedians=True,quantiles=quantiles)
    ax.violinplot(nc,[1],widths=1,showmedians=True,quantiles=quantiles)
    ax.violinplot(ec,[2],widths=1,showmedians=True,quantiles=quantiles)

    # Statistical test
    st1, p1 = ranksums(xc, nc)
    st2, p2 = ranksums(nc, ec)
    e1 = abs(np.median(nc)-np.median(xc))/np.median(xc)
    e2 = abs(np.median(ec)-np.median(nc))/np.median(nc)

    ylim = ax.get_ylim()
    ypa = .8*(ylim[1]-ylim[0]) + ylim[0]
    ypb = .85*(ylim[1]-ylim[0]) + ylim[0]

    if p1<pt:
        ax.text(.5,ypb,f'p:{p1:.2E}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(.5,ypb,f'p:{p1:.2E}',fontsize=fs2,ha='center')
    if e1>et:
        ax.text(.5,ypa,f'e:{e1:.2f}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(.5,ypa,f'e:{e1:.2f}',fontsize=fs2,ha='center')
    if p2<pt:
        ax.text(1.5,ypb,f'p:{p2:.2E}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(1.5,ypb,f'p:{p2:.2E}',fontsize=fs2,ha='center')
    if e2>et:
        ax.text(1.5,ypa,f'e:{e2:.2f}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(1.5,ypa,f'e:{e2:.2f}',fontsize=fs2,ha='center')

    if comp4volume is False:
        ax.text(0,np.median(xc),f'{np.median(xc):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.median(nc),f'{np.median(nc):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.median(ec),f'{np.median(ec):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[0]),f'{np.percentile(xc,100*quantiles[0]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[0]),f'{np.percentile(nc,100*quantiles[0]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[0]),f'{np.percentile(ec,100*quantiles[0]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[1]),f'{np.percentile(xc,100*quantiles[1]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[1]),f'{np.percentile(nc,100*quantiles[1]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[1]),f'{np.percentile(ec,100*quantiles[1]):.0f}',ha='left',va='bottom',fontsize=fs2)
    else:
        ax.text(0,np.median(xc),f'{np.median(xc):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.median(nc),f'{np.median(nc):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.median(ec),f'{np.median(ec):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[0]),f'{np.percentile(xc,100*quantiles[0]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[0]),f'{np.percentile(nc,100*quantiles[0]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[0]),f'{np.percentile(ec,100*quantiles[0]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[1]),f'{np.percentile(xc,100*quantiles[1]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[1]),f'{np.percentile(nc,100*quantiles[1]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[1]),f'{np.percentile(ec,100*quantiles[1]):.2E}',ha='left',va='bottom',fontsize=fs2)


    ax.set_title(ncm)
    # ax.set_yticklabels([])
    ax.yaxis.set_major_formatter(sf)
    ax.ticklabel_format(axis='y',style='sci')
    xlabel = [f'All (n={len(xc)})',f'Non-edge (n={len(nc)})',f'Edge (n={len(ec)})']
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(xlabel,rotation=-15)

    if p2<pt and e2>et:
        ax.set_facecolor('lightyellow')

for index, rowX in structures.iterrows():
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

    xc = (
        cells.loc[cells["structure_name"] == rowX['Gene'], FS["struct_metrics"][0]]
            .squeeze()
            .to_numpy()
    )
    nc = (
        ncells.loc[ncells["structure_name"] == rowX['Gene'], FS["struct_metrics"][0]]
            .squeeze()
            .to_numpy()
    )
    ec = (
        ecells.loc[ecells["structure_name"] == rowX['Gene'], FS["struct_metrics"][0]]
            .squeeze()
            .to_numpy()
    )

    xc_v = (
        cells.loc[cells["structure_name"] == rowX['Gene'], cv]
            .squeeze()
            .to_numpy()
    )
    nc_v = (
        ncells.loc[ncells["structure_name"] == rowX['Gene'], cv]
            .squeeze()
            .to_numpy()
    )
    ec_v = (
        ecells.loc[ecells["structure_name"] == rowX['Gene'], cv]
            .squeeze()
            .to_numpy()
    )


    if comp4volume is True:
        xc = xc/xc_v
        nc = nc/nc_v
        ec = ec/ec_v

    ax.violinplot(xc,[0],widths=1,showmedians=True,quantiles=quantiles)
    ax.violinplot(nc,[1],widths=1,showmedians=True,quantiles=quantiles)
    ax.violinplot(ec,[2],widths=1,showmedians=True,quantiles=quantiles)

    # Statistical test
    st1, p1 = ranksums(xc, nc)
    st2, p2 = ranksums(nc, ec)
    e1 = abs(np.median(nc)-np.median(xc))/np.median(xc)
    e2 = abs(np.median(ec)-np.median(nc))/np.median(nc)

    ylim = ax.get_ylim()
    ypa = .8*(ylim[1]-ylim[0]) + ylim[0]
    ypb = .85*(ylim[1]-ylim[0]) + ylim[0]

    if p1<pt:
        ax.text(.5,ypb,f'p:{p1:.2E}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(.5,ypb,f'p:{p1:.2E}',fontsize=fs2,ha='center')
    if e1>et:
        ax.text(.5,ypa,f'e:{e1:.2f}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(.5,ypa,f'e:{e1:.2f}',fontsize=fs2,ha='center')
    if p2<pt:
        ax.text(1.5,ypb,f'p:{p2:.2E}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(1.5,ypb,f'p:{p2:.2E}',fontsize=fs2,ha='center')
    if e2>et:
        ax.text(1.5,ypa,f'e:{e2:.2f}',fontsize=fs2,ha='center',weight='bold')
    else:
        ax.text(1.5,ypa,f'e:{e2:.2f}',fontsize=fs2,ha='center')

    if comp4volume is False:
        ax.text(0,np.median(xc),f'{np.median(xc):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.median(nc),f'{np.median(nc):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.median(ec),f'{np.median(ec):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[0]),f'{np.percentile(xc,100*quantiles[0]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[0]),f'{np.percentile(nc,100*quantiles[0]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[0]),f'{np.percentile(ec,100*quantiles[0]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[1]),f'{np.percentile(xc,100*quantiles[1]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[1]),f'{np.percentile(nc,100*quantiles[1]):.0f}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[1]),f'{np.percentile(ec,100*quantiles[1]):.0f}',ha='left',va='bottom',fontsize=fs2)
    else:
        ax.text(0,np.median(xc),f'{np.median(xc):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.median(nc),f'{np.median(nc):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.median(ec),f'{np.median(ec):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[0]),f'{np.percentile(xc,100*quantiles[0]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[0]),f'{np.percentile(nc,100*quantiles[0]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[0]),f'{np.percentile(ec,100*quantiles[0]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(0,np.percentile(xc,100*quantiles[1]),f'{np.percentile(xc,100*quantiles[1]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(1,np.percentile(nc,100*quantiles[1]),f'{np.percentile(nc,100*quantiles[1]):.2E}',ha='left',va='bottom',fontsize=fs2)
        ax.text(2,np.percentile(ec,100*quantiles[1]),f'{np.percentile(ec,100*quantiles[1]):.2E}',ha='left',va='bottom',fontsize=fs2)

    ax.set_title(rowX['Structure'])
    # ax.set_yticklabels()
    ax.yaxis.set_major_formatter(sf)
    ax.ticklabel_format(axis='y',style='sci')
    xlabel = [f'All (n={len(xc)})',f'Non-edge (n={len(nc)})',f'Edge (n={len(ec)})']
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(xlabel,rotation=-15)

    if p2<pt and e2>et:
        ax.set_facecolor('lightyellow')


if save_flag == 1:
    plot_save_path = pic_rootT / f"SizeScalingMetrics_Edgecells_compensate4volume{comp4volume}_withnumbersandtests_ylabels.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

# %% plot for comparing PCA components
# %% Feature sets
FS = {}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    'NUC_MEM_PC1',
    'NUC_MEM_PC2', 'NUC_MEM_PC3', 'NUC_MEM_PC4', 'NUC_MEM_PC5',
    'NUC_MEM_PC6', 'NUC_MEM_PC7', 'NUC_MEM_PC8'
]

# %% Annotation
nrows = 2
ncols = 7

# %% Parametrization
quantiles=[0.25,0.75]
comp4volume = False
cv = "Cell volume"

#%% Time for a flexible scatterplot
w1 = 0.01
w2 = 0.01
w3 = 0
h1 = 0.05
h2 = 0.1
h3 = 0.05
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
for j,ncm in enumerate(FS["cellnuc_metrics"]):
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

    if comp4volume is True:
        xc = cells[ncm].values/cells[cv].values
        nc = ncells[ncm].values/ncells[cv].values
        ec = ecells[ncm].values/ecells[cv].values
    else:
        xc = cells[ncm].values
        nc = ncells[ncm].values
        ec = ecells[ncm].values

    ax.violinplot(xc,[0],widths=1,showmedians=True,quantiles=quantiles)
    ax.violinplot(nc,[1],widths=1,showmedians=True,quantiles=quantiles)
    ax.violinplot(ec,[2],widths=1,showmedians=True,quantiles=quantiles)

    ax.set_title(ncm)
    ax.set_yticklabels([])
    xlabel = [f'All (n={len(xc)})',f'Non-edge (n={len(nc)})',f'Edge (n={len(ec)})']
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(xlabel,rotation=-15)

if save_flag == 1:
    plot_save_path = pic_rootT / f"SizeScalingMetrics_Edgecells_includingSHAPEMODES.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()


# %% Some quick scatter plots
List1 = ["Cell height","Cell volume"]
List2 = ["NUC_MEM_PC1","NUC_MEM_PC2"]
nrows = 2
ncols = 3

#%% Time for a flexible scatterplot
plt.rcParams.update({"font.size": 12})
w1 = 0.1
w2 = 0.1
w3 = 0.01
h1 = 0.05
h2 = 0.1
h3 = 0.05
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

    xlab = List1[row-1]
    ylab = List2[row-1]
    if col==1:
        x = cells[List1[row-1]].values
        y = cells[List2[row-1]].values
        t = f'All cells (n={len(x)})'
        c = 'blue'
    elif col==2:
        x = ncells[List1[row-1]].values
        y = ncells[List2[row-1]].values
        t = f'Non-edge cells (n={len(x)})'
        c = 'orange'
    elif col==3:
        x = ecells[List1[row-1]].values
        y = ecells[List2[row-1]].values
        t = f'Edge cells (n={len(x)})'
        c = 'green'


    ax.scatter(x,y,10,color=c,alpha=0.1)
    ax.set_title(t)
    ax.set_xlabel(List1[row-1])
    ax.set_ylabel(List2[row-1])
    if col==1:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()

if save_flag == 1:
    plot_save_path = pic_rootT / f"SizeScalingMetrics_ComparingMETRICSwithSHAPEMODES.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

# %% Some quick scatter plots
List1 = ["Cell height"]
List2 = ["Cell volume"]
nrows = 1
ncols = 3
save_flag = 0
from scipy.stats.stats import pearsonr

#%% Time for a flexible scatterplot
plt.rcParams.update({"font.size": 12})
w1 = 0.1
w2 = 0.1
w3 = 0.01
h1 = 0.1
h2 = 0.1
h3 = 0.05
xp = 0
yp = 0
xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
xw = xx * xp
xx = xx * (1 - xp)
yw = yy * yp
yy = yy * (1 - yp)

fig = plt.figure(figsize=(16, 5))

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

    xlab = List1[row-1]
    ylab = List2[row-1]
    if col==1:
        x = cells[List1[row-1]].values
        y = cells[List2[row-1]].values
        r,_ = pearsonr(x, y)
        t = f'All cells (n={len(x)}, r={r:.2f})'
        c = 'blue'
    elif col==2:
        x = ncells[List1[row-1]].values
        y = ncells[List2[row-1]].values
        r,_ = pearsonr(x, y)
        t = f'Non-edge cells (n={len(x)}, r={r:.2f})'
        c = 'orange'
    elif col==3:
        x = ecells[List1[row-1]].values
        y = ecells[List2[row-1]].values
        r,_ = pearsonr(x, y)
        t = f'Edge cells (n={len(x)}, r={r:.2f})'
        c = 'green'



    ax.scatter(x,y,10,color=c,alpha=0.1)
    ax.set_title(t)
    ax.set_xlabel(List1[row-1])
    ax.set_ylabel(List2[row-1])
    if col==1:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()

if save_flag == 1:
    plot_save_path = pic_rootT / f"SizeScalingMetrics_ComparingHEIGHTwithVOLUME.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()

# %%
# %% Parameters, updated directories
save_flag = 1  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 8})
plt.rcParams["svg.fonttype"] = "none"

# %% Feature sets
FS = {}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
]

FS["struct_metrics"] = [
    "Structure volume",
]


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
plt.rcParams.update({"font.size": 12})
w1 = 0.1
w2 = 0.1
w3 = 0.01
h1 = 0.1
h2 = 0.1
h3 = 0.05
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

    gene = structures.iloc[col]['Gene']

    if col==1:
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
    elif col==2:
        x = (
            ncells.loc[cells["structure_name"] == gene, xv]
                .squeeze()
                .to_numpy()
        )
        y = (
            ncells.loc[cells["structure_name"] == gene, yv]
                .squeeze()
                .to_numpy()
        )
        r,_ = pearsonr(x, y)
        t = f'Non-edge cells (n={len(x)}, r={r:.2f})'
        c = 'orange'
    elif col==3:
        x = (
            ecells.loc[cells["structure_name"] == gene, xv]
                .squeeze()
                .to_numpy()
        )
        y = (
            ecells.loc[cells["structure_name"] == gene, yv]
                .squeeze()
                .to_numpy()
        )
        if len(x)>0:
            r,_ = pearsonr(x, y)
            t = f'Edge cells (n={len(x)}, r={r:.2f})'
        else:
            t = f'Edge cells (n={len(x)})'
        c = 'green'

    ax.set_title(t)
    ax.set_xlabel(xv)
    ax.set_ylabel(f'{gene} {yv}')
    # if row==1:
    #     xlim = ax.get_xlim()
    #     ylim = ax.get_ylim()
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    ax.grid()

if save_flag == 1:
    plot_save_path = pic_rootT / f"SizeScalingMetrics__.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
else:
    plt.show()





