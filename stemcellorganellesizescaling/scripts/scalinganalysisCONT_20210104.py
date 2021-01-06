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
# importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression"])
# from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats, explain_var_compositemodels, bootstrap_linear_and_log_model

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
structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201113.csv")

# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "growing"
pic_root.mkdir(exist_ok=True)

# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result


# %% Select feature sets
FS = {}
FS['cellnuc_metrics'] = [
    "Cell surface area",
    "Nuclear surface area",
    "Nuclear volume",
    "Cytoplasmic volume",
]

FS['struct_metrics'] = [
    "Structure volume",
]

# %% Parameters
nbins = 50 # for doubling values estimate
growfac = 2
nbins2 = 25 # for plotting curve
perc_values = [5, 25, 50, 75, 95]  # for plotting curve
type = 'Linear'
save_dir = data_root / "growing"
save_dir.mkdir(exist_ok=True)

# %%
# fig = plt.figure(figsize=(16, 9))
# plt.plot(x,y,'r.',markersize=5)
# plt.plot(x[pos],y[pos],'b.',markersize=5)
# plt.grid()
# # plt.xlabel('Percentage scaling factor')
# # plt.ylabel('Log-log scaling factor')
# plt.show()

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

# %% Compute cell volume vs. nuclear volume
th = 50
ScaleMat = pd.DataFrame()
for yi, ylabel in enumerate(FS['cellnuc_metrics']):
    x = cells['Cell volume'].squeeze().to_numpy()
    y = cells[ylabel].squeeze().to_numpy()
    loaddir = (data_root / statsIN / 'cell_nuc_metrics')
    cii = loadps(loaddir, f"Cell volume_{ylabel}_cell_dens")
    pos = np.argwhere(cii>np.percentile(cii,th))
    x = x[pos]
    y = y[pos]
    model, _ = fit_ols(x, y, 'Linear')
    xC = cell_doubling.copy()
    xC = sm.add_constant(xC)
    yC = model.predict(xC)
    ScaleMat.loc[yi,"prc"] = (yC[1] - yC[0]) / yC[0]
    model_l2, _ = fit_ols(np.log2(x), np.log2(y), 'Linear')
    ScaleMat.loc[yi, "log2"] = model_l2.params[1]
    model_l10, _ = fit_ols(np.log10(x), np.log10(y), 'Linear')
    ScaleMat.loc[yi, "log10"] = model_l10.params[1]
    model_le, _ = fit_ols(np.log(x), np.log(y), 'Linear')
    ScaleMat.loc[yi, "loge"] = model_le.params[1]
    ScaleMat.loc[yi, "name"] = ylabel

counter = yi

for yi, ylabel in enumerate(FS['struct_metrics']):
    selected_structures = structures['Gene']
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
        loaddir = (data_root / statsIN / 'cellnuc_struct_metrics')
        cii = loadps(loaddir, f"Cell volume_{ylabel}_{struct}_cell_dens")
        pos = np.argwhere(cii > np.percentile(cii, th))
        x = x[pos]
        y = y[pos]
        model, _ = fit_ols(x, y, 'Linear')
        xC = cell_doubling.copy()
        xC = sm.add_constant(xC)
        yC = model.predict(xC)
        ScaleMat.loc[si+counter, "prc"] = (yC[1] - yC[0]) / yC[0]
        model_l2, _ = fit_ols(np.log2(x), np.log2(y), 'Linear')
        ScaleMat.loc[si+counter, "log2"] = model_l2.params[1]
        model_l10, _ = fit_ols(np.log10(x), np.log10(y), 'Linear')
        ScaleMat.loc[si+counter, "log10"] = model_l10.params[1]
        model_le, _ = fit_ols(np.log(x), np.log(y), 'Linear')
        ScaleMat.loc[si+counter, "loge"] = model_le.params[1]
        ScaleMat.loc[si + counter, "name"] = structures.loc[si,'Structure']

# %%
fs = 10
plt.rcParams.update({"font.size": fs})
fig, axes = plt.subplots(1,1, figsize=(8, 8), dpi=100)
axes.scatter(ScaleMat["prc"],ScaleMat["log2"])
for i, name in enumerate(ScaleMat['name']):
    axes.text(ScaleMat.loc[i,'prc'],ScaleMat.loc[i,'log2'],name,ha='left',va='top')
axes.grid()
# axes.set_xlim(left=0)
# axes.set_ylim(bottom=0)
plt.xlabel('r_lin')
plt.ylabel('r_log')
plt.plot([0,1],[0,1],'--',color=[0,0,0,0.1])
plt.show()

# %%
fs = 10
plt.rcParams.update({"font.size": fs})
fig, axes = plt.subplots(1,1, figsize=(8, 8), dpi=100)
axes.scatter(x,y)
plt.show()


# %%
x_lin = cells['Cell volume'].squeeze().to_numpy()
y_lin = cells['Nuclear volume'].squeeze().to_numpy()
x_log = np.log2(x_lin)
y_log = np.log2(y_lin)

x_linA = x_lin.copy()
x_linA = sm.add_constant(x_linA)
model_lin = sm.OLS(y_lin, x_linA)
fittedmodel_lin = model_lin.fit()

xC = cell_doubling.copy()
xC = sm.add_constant(xC)
yC = fittedmodel_lin.predict(xC)
r_lin = (yC[1] - yC[0]) / yC[0]
print(r_lin)

fs = 10
plt.rcParams.update({"font.size": fs})
fig, axes = plt.subplots(1,2, figsize=(16, 9), dpi=100)
axes.scatter(np.log2(x),np.log2(y))
axes.grid()
plt.show()



# %%














x = np.log2(cells.loc[cells['structure_name']==struct,'Cell volume'].to_numpy())
y = np.log2(cells.loc[cells['structure_name']==struct,'Structure volume'].to_numpy())
x_lin = x.copy()
x_lin = sm.add_constant(x_lin)
model_lin = sm.OLS(y, x_lin)
fittedmodel_lin = model_lin.fit()
rs1 = fittedmodel_lin.rsquared
ax2.plot(x,y,'r.',markersize=5)
xC = [np.amin(x), np.amax(x)]
xC_lin = sm.add_constant(xC)
yC = fittedmodel_lin.predict(xC_lin)
xD = [9e5, 18e5]
xD_lin = sm.add_constant(xD)
yD = fittedmodel_lin.predict(xD_lin)
sf = np.round(fittedmodel_lin.params[1],2)
ax2.plot(xC,yC,'b')
ax2.set_xlabel('Cell Vol')
ax1.set_ylabel(f"{struct} vol")
ax2.set_title(f"rs={np.round(rs1,2)}     sf={sf}")
ax2.grid()
plt.show()




