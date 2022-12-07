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
no_of_Rows = len(structures)+len(FS['cellnuc_metrics'])+2
no_of_Cols = 17

# %% Pre-fill the tabl
res_table = pd.DataFrame(index=range(no_of_Rows),columns=range(no_of_Cols))
res_table.iat[0,1] = 'All cells'
res_table.iat[0,5] = 'Non-edge cells'
res_table.iat[0,9] = 'Edge cells'
res_table.iat[0,13] = 'All vs Non-edge TEST'
res_table.iat[0,15] = 'Non-edge vs Edge TEST'
res_table.iat[1,1] = 'n'
res_table.iat[1,2] = 'Q1'
res_table.iat[1,3] = 'median'
res_table.iat[1,4] = 'Q3'
res_table.iat[1,5] = 'n'
res_table.iat[1,6] = 'Q1'
res_table.iat[1,7] = 'median'
res_table.iat[1,8] = 'Q3'
res_table.iat[1,9] = 'n'
res_table.iat[1,10] = 'Q1'
res_table.iat[1,11] = 'median'
res_table.iat[1,12] = 'Q3'
res_table.iat[1,13] = 'P-value'
res_table.iat[1,14] = 'Effect size'
res_table.iat[1,15] = 'P-value'
res_table.iat[1,16] = 'Effect size'

# %% Parametrization
quantiles=[0.25,0.75]
comp4volume = True
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

    ncm_ann=ncm
    if comp4volume is False:
        if 'height' in ncm_ann:
            ncm_ann = f"{ncm_ann} (\u03BCm)"
        elif 'area' in ncm_ann:
            ncm_ann = f"{ncm_ann} (\u03BCm\u00b2)"
        else:
            ncm_ann = f"{ncm_ann} (\u03BCm\u00b3)"

    res_table.iat[i+2,0] = ncm_ann

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
        if 'height' in ncm:
            xc = xc * ((0.108333) ** 1)
            nc = nc * ((0.108333) ** 1)
            ec = ec * ((0.108333) ** 1)
        elif 'area' in ncm:
            xc = xc * ((0.108333) ** 2)
            nc = nc * ((0.108333) ** 2)
            ec = ec * ((0.108333) ** 2)
        else:
            xc = xc * ((0.108333) ** 3)
            nc = nc * ((0.108333) ** 3)
            ec = ec * ((0.108333) ** 3)

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

    res_table.iat[i+2,1] = len(xc)
    res_table.iat[i+2,2] = np.percentile(xc,100*quantiles[0])
    res_table.iat[i+2,3] = np.median(xc)
    res_table.iat[i+2,4] = np.percentile(xc,100*quantiles[1])
    res_table.iat[i+2,5] = len(nc)
    res_table.iat[i+2,6] = np.percentile(nc,100*quantiles[0])
    res_table.iat[i+2,7] = np.median(nc)
    res_table.iat[i+2,8] = np.percentile(nc,100*quantiles[1])
    res_table.iat[i+2,9] = len(ec)
    res_table.iat[i+2,10] = np.percentile(ec,100*quantiles[0])
    res_table.iat[i+2,11] = np.median(ec)
    res_table.iat[i+2,12] = np.percentile(ec,100*quantiles[1])
    res_table.iat[i+2,13] = p1
    res_table.iat[i+2,14] = e1
    res_table.iat[i+2,15] = p2
    res_table.iat[i+2,16] = e2

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

    ncm = rowX['Structure']

    ncm_ann = ncm

    if comp4volume is False:
        if 'height' in ncm_ann:
            ncm_ann = f"{ncm_ann} (\u03BCm)"
        elif 'area' in ncm_ann:
            ncm_ann = f"{ncm_ann} (\u03BCm\u00b2)"
        else:
            ncm_ann = f"{ncm_ann} (\u03BCm\u00b3)"

    res_table.iat[i+2,0] = ncm_ann

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
    else:
        if 'height' in ncm:
            xc = xc * ((0.108333) ** 1)
            nc = nc * ((0.108333) ** 1)
            ec = ec * ((0.108333) ** 1)
        elif 'area' in ncm:
            xc = xc * ((0.108333) ** 2)
            nc = nc * ((0.108333) ** 2)
            ec = ec * ((0.108333) ** 2)
        else:
            xc = xc * ((0.108333) ** 3)
            nc = nc * ((0.108333) ** 3)
            ec = ec * ((0.108333) ** 3)

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


    res_table.iat[i+2,1] = len(xc)
    res_table.iat[i+2,2] = np.percentile(xc,100*quantiles[0])
    res_table.iat[i+2,3] = np.median(xc)
    res_table.iat[i+2,4] = np.percentile(xc,100*quantiles[1])
    res_table.iat[i+2,5] = len(nc)
    res_table.iat[i+2,6] = np.percentile(nc,100*quantiles[0])
    res_table.iat[i+2,7] = np.median(nc)
    res_table.iat[i+2,8] = np.percentile(nc,100*quantiles[1])
    res_table.iat[i+2,9] = len(ec)
    res_table.iat[i+2,10] = np.percentile(ec,100*quantiles[0])
    res_table.iat[i+2,11] = np.median(ec)
    res_table.iat[i+2,12] = np.percentile(ec,100*quantiles[1])
    res_table.iat[i+2,13] = p1
    res_table.iat[i+2,14] = e1
    res_table.iat[i+2,15] = p2
    res_table.iat[i+2,16] = e2



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

# %%
res_table.to_csv(pic_rootT / f"DataFileS4_compensate4volume{comp4volume}.csv", encoding='utf-8-sig')

