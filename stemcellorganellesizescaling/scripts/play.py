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
from matplotlib import cm
import pickle
from datetime import datetime
import seaborn as sns
import os, platform

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats, explain_var_compositemodels, bootstrap_linear_and_log_model
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression"])
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats, explain_var_compositemodels, bootstrap_linear_and_log_model

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

tableIN = "SizeScaling_20201102.csv"
table_compIN = "SizeScaling_20201102_comp.csv"
statsIN = "Stats_20201102"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())

# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "growing"
pic_root.mkdir(exist_ok=True)

# %% Select feature sets
FS = {}
FS['cellnuc_metrics'] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    "Cytoplasmic volume",
]

FS['struct_metrics'] = [
    "Structure volume",
    "Number of pieces",
    "Piece average",
    "Piece std",
]

# %% Parameters
nbins = 50 # for doubling values estimate
growfac = 2
nbins2 = 100 # for plotting curve
shift = 50 # for plotting curve
perc_values = [5, 25, 50, 75, 95]  # for plotting curve
type = 'Linear'

# %%
fig = plt.figure(figsize=(16, 9))
plt.plot(PM[:,0],PM[:,1],'r.',markersize=5)
plt.grid()
plt.xlabel('Percentage scaling factor')
plt.ylabel('Log-log scaling factor')
plt.show()

# %% Doubling values
x = cells['Cell volume'].to_numpy()
xc, xbins = np.histogram(x,bins=nbins)
xbincenter = np.zeros((nbins, 1))
for n in range(nbins):
    xbincenter[n] = np.mean([xbins[n], xbins[n + 1]])
ylm = 1.05 * xc.max()
idx = np.digitize(x, xbins)

xstats = np.zeros((nbins, 5))
for n in range(nbins):
    pos = np.argmin(abs(xbincenter-(xbincenter[n]*growfac)))
    xstats[n,0] = xc[n]
    xstats[n,1] = xc[pos]
    xstats[n,2] = np.minimum(xc[pos],xc[n])
    xstats[n,3] = np.sum(xc[n:pos+1])
    xstats[n,4] = pos

start_bin = np.argmax(xstats[:,2])
end_bin = int(xstats[start_bin,4])

cell_doubling = xbincenter[[start_bin, end_bin]]
cell_doubling[1] = 2*cell_doubling[0]

print(cell_doubling)

# %% Compute
ScaleMat = pd.DataFrame()
for yi, ylabel in enumerate(FS['cellnuc_metrics']):
    x = cells['Cell volume'].squeeze().to_numpy()
    y = cells[ylabel].squeeze().to_numpy()
    scaling_stats = bootstrap_linear_and_log_model(x, y, 'Cell volume', ylabel, type, cell_doubling, 'None')
    ScaleMat[f"{ylabel}_prc"] = scaling_stats[:,0]
    ScaleMat[f"{ylabel}_log"] = scaling_stats[:,1]
    if yi==0:
        PM = scaling_stats
    else:
        PM = np.concatenate((PM, scaling_stats), axis=0)

for yi, ylabel in enumerate(FS['struct_metrics']):
    selected_structures = cells["structure_name"].unique()
    for si, struct in enumerate(selected_structures):
        x = (
            cells.loc[cells["structure_name"] == struct, 'Cell volume']
                .squeeze()
                .to_numpy()
        )
        y = (
            cells.loc[cells["structure_name"] == struct, ylabel]
                .squeeze()
                .to_numpy()
        )
        scaling_stats = bootstrap_linear_and_log_model(x, y, 'Cell volume', ylabel, type, cell_doubling, struct)
        ScaleMat[f"{ylabel}_{struct}prc"] = scaling_stats[:, 0]
        ScaleMat[f"{ylabel}_{struct}log"] = scaling_stats[:, 1]
        PM = np.concatenate((PM, scaling_stats), axis=0)

# %% Saving
save_dir = data_root / "growing"
save_dir.mkdir(exist_ok=True)
ScaleMat.to_csv(save_dir / "ScaleStats_20201113.csv")

# %% Make signals
cellnuc_metrics = ['Cell surface area', 'Cell volume', 'Cell height',
                     'Nuclear surface area', 'Nuclear volume', 'Nucleus height',
                     'Cytoplasmic volume']
struct_metrics =  ['Structure volume', 'Number of pieces']
Grow = pd.DataFrame()

x = cells['Cell volume'].to_numpy()/fac
xbins = np.linspace()
for i, metric in tqdm(enumerate(FS['cellnuc_metrics']), 'Cell metrics'):
    y = cells[metric].to_numpy() / fac
    y_res = np.zeros((nbins2, len(perc_values)))
    for n in range(nbins2):
        sc = np.argwhere(idx2 == (n + 1))
        if len(sc) < 1:
            y_res[n, :] = np.nan
            y_res[n, 0] = len(sc)
        else:
            y_res[n, 0] = len(sc)
            y_res[n, 1] = np.mean(y[sc])
            y_res[n, 2:(len(perc_values)+2)] = np.percentile(y[sc],perc_values)
    Grow[(f"{metric}_n")] = y_res[:,0]
    Grow[(f"{metric}_mean")] = y_res[:,1]
    for n in np.arange(len(perc_values)):
        Grow[f"{metric}_{perc_values[n]}"] = y_res[:,n+2]

for i, metric in tqdm(enumerate(struct_metrics), 'Organelles'):
    y = cells[metric].to_numpy() / fac
    selected_structures = cells['structure_name'].unique()
    for si, struct in enumerate(selected_structures):
        pos = np.argwhere(cells['structure_name'].to_numpy() == struct)
        y_res = np.zeros((nbins, 7))
        for n in range(nbins):
            sc = np.argwhere(idx == (n + 1))
            sc = np.intersect1d(sc,pos)
            if len(sc) < 1:
                y_res[n, :] = np.nan
                y_res[n, 0] = len(sc)
            else:
                y_res[n, 0] = len(sc)
                y_res[n, 1] = np.mean(y[sc])
                y_res[n, 2:(len(perc_values) + 2)] = np.percentile(y[sc], perc_values)
        Grow[(f"{metric}_{struct}_n")] = y_res[:, 0]
        Grow[(f"{metric}_{struct}_mean")] = y_res[:, 1]
        for n in np.arange(len(perc_values)):
            Grow[f"{metric}_{struct}_{perc_values[n]}"] = y_res[:, n + 2]

#%% Add final row with data
gnp = Grow.to_numpy()
growth_rates = np.divide((gnp[end_bin,:] - gnp[start_bin,:]),gnp[start_bin,:])
growth_rates = np.round(100*growth_rates).astype(np.int)
Grow = Grow.append(pd.Series(), ignore_index=True)
Grow.iloc[nbins,:] = growth_rates

for i, metric in tqdm(enumerate(struct_metrics), 'Organelles'):
    for si, struct in enumerate(selected_structures):
        v25 = Grow[f"{metric}_{struct}_{25}"].to_numpy()
        v50 = Grow[f"{metric}_{struct}_{50}"].to_numpy()
        v75 = Grow[f"{metric}_{struct}_{75}"].to_numpy()
        f25 = np.median(np.divide(v25[start_bin:end_bin],v50[start_bin:end_bin]))
        f75 = np.median(np.divide(v75[start_bin:end_bin], v50[start_bin:end_bin]))
        Grow.loc[51,f"{metric}_{struct}_{25}"] = f25*Grow.loc[50,f"{metric}_{struct}_{50}"]
        Grow.loc[51, f"{metric}_{struct}_{75}"] = f75 * Grow.loc[50, f"{metric}_{struct}_{50}"]
save_dir = data_root / "growing"
save_dir.mkdir(exist_ok=True)
# Grow.to_csv(save_dir / "Growthstats_20201012.csv")


# %% Add bincenters as well
Grow.loc[np.arange(len(xbincenter)),'bins'] = xbincenter.squeeze()
Grow.loc[50,'bins'] = start_bin
Grow.loc[51,'bins'] = end_bin

Grow.to_csv(save_dir / "Growthstats_20201102.csv")

# %% Select metrics to plot
selected_structures = cells['structure_name'].unique()

# selected_structures = ['LMNB1', 'ST6GAL1', 'TOMM20', 'SEC61B']

struct_metrics = ['Structure volume']
# %%

for i, struct in enumerate(cellnuc_metrics):
    dus = struct
# for i, metric in enumerate(struct_metrics):
#     for j, struct in enumerate(selected_structures):
#         dus = f"{metric}_{struct}"


    # %% Growth plot
    w1 = 0.1
    w2 = 0.01
    h1 = 0.05
    h2 = 0.01
    y0 = 0.07
    y1 = 0.3
    y2 = 1-y0-y1-h1-h2
    x1 = 1 - w1 - w2
    lw = 1

    fig = plt.figure(figsize=(10, 10))

    # Zoom
    axZoom = fig.add_axes([w1,h1+y2,x1,y0])
    axZoom.plot(5,5)
    axZoom.set_xlim(left=x.min(), right=x.max())
    axZoom.set_ylim(bottom=0, top=ylm)
    axZoom.plot([x.min(), xbincenter[start_bin]], [0, ylm],'r',linewidth=lw)
    axZoom.plot([x.max(), xbincenter[end_bin]], [0, ylm],'r',linewidth=lw)
    axZoom.axis('off')

    # Cell Size
    axSize = fig.add_axes([w1,h1+y2+y0,x1,y1])
    # axSize.stem(x, max(xc)/10*np.ones(x.shape), linefmt='g-', markerfmt=None, basefmt=None)
    axSize.hist(x, bins=nbins, color=[.5, .5, .5, .5])
    axSize.grid()
    axSize.set_xlim(left=x.min(), right=x.max())
    axSize.set_ylim(bottom=0, top=ylm)
    axSize.plot(xbincenter[[start_bin, start_bin]], [0, ylm],'r',linewidth=lw)
    axSize.plot(xbincenter[[end_bin, end_bin]], [0, ylm],'r',linewidth=lw)
    axSize.set_xlabel('Cell size')

    # Grow
    axGrow = fig.add_axes([w1,h1,x1,y2])
    xd = Grow['Cell volume_mean'].to_numpy()
    xd = xd[start_bin:(end_bin+1)]
    xd = xd/xd[0]
    xd = np.log2(xd)
    yd = Grow['Cell volume_mean'].to_numpy()
    yd = yd[start_bin:(end_bin+1)]
    yd = yd/yd[0]
    yd = np.log2(yd)
    axGrow.plot(xd,yd,'k--')

    ym = Grow[f"{dus}_mean"].to_numpy()
    ym = ym[start_bin:(end_bin+1)]
    ym = ym/ym[0]
    ym = np.log2(ym)

    ymat = np.zeros((len(ym),len(perc_values)))
    for i, n in enumerate(perc_values):
        yi = Grow[f"{dus}_{n}"].to_numpy()
        yi = yi[start_bin:(end_bin + 1)]
        ymat[:,i] = yi
    ymat = ymat / ymat[0,2]
    ymat = np.log2(ymat)


    yv = [0,4]
    xf = np.concatenate((np.expand_dims(xd,axis=1),np.flipud(np.expand_dims(xd,axis=1))))
    yf = np.concatenate((np.expand_dims(ymat[:,yv[0]],axis=1),np.flipud(np.expand_dims(ymat[:,yv[1]],axis=1))))
    axGrow.fill(xf,yf,color=[0.95, 0.95, 1, 0.8])
    yv = [1,3]
    xf = np.concatenate((np.expand_dims(xd,axis=1),np.flipud(np.expand_dims(xd,axis=1))))
    yf = np.concatenate((np.expand_dims(ymat[:,yv[0]],axis=1),np.flipud(np.expand_dims(ymat[:,yv[1]],axis=1))))
    axGrow.fill(xf,yf,color=[0.5, 0.5, 1, 0.8])


    axGrow.plot(xd,ymat[:,2],color=[0, 0, 1, 1])
    axGrow.grid()
    axGrow.set_xlabel('Cell growth (log 2)')
    axGrow.set_ylabel('Organnele growth (log 2)')
    axGrow.set_xlim(left=0,right=np.log2(growfac))
    axGrow.set_ylim(bottom=-0.5, top=1.5)
    axGrow.text(np.log2(growfac)/2,1.5,dus,fontsize=20,color=[0, 0, 1, 1],verticalalignment='top',horizontalalignment='center')


    if save_flag:
        plot_save_path = pic_root / f"{dus}.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

# %%

