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
import sys, importlib

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

# %% Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201102.csv"
table_compIN = "SizeScaling_20201102_comp.csv"
statsIN = "Stats_20201102"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())
structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201113.csv")

animals = pd.read_csv(data_root / 'animals' / "Mammals.csv")


# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "scalinv"
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

# %% Doubling values
x = cells['Nuclear volume'].to_numpy()
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

nuc_doubling = xbincenter[[start_bin, end_bin]]
nuc_doubling[1] = 2*nuc_doubling[0]

print(nuc_doubling)



# %%
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression"])
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols


# %% Testing various fits, getting scaling parameters and explained variances
fs = 16
plt.rcParams.update({"font.size": fs})
ScaleMat = pd.DataFrame()
yi = 0
fig = plt.figure(figsize=(16, 9))
lw = 3

# data
xlabel = 'Nuclear volume'
ylabel = 'Nuclear surface area'
x = cells[xlabel].squeeze().to_numpy()
y = cells[ylabel].squeeze().to_numpy()
ybar = np.mean(y)
sstot = np.sum((y - ybar)**2)
xs = np.linspace(1,max(x),100)

# plot data
plt.plot(x,y,'b.')

# linear model
model, yhat_lin = fit_ols(x, y, 'Linear',x)
ssres = np.sum((y - yhat_lin)**2)
cod_lin = 1- (ssres/sstot)
ScaleMat.loc[yi,"CoD_lin"] = cod_lin
xC = nuc_doubling.copy()
xC = sm.add_constant(xC)
yC = model.predict(xC)
scaling_rate_lin = (yC[1] - yC[0]) / yC[0]
ScaleMat.loc[yi,"ScalingRateLin"] = scaling_rate_lin

# plot linear model
model, ys = fit_ols(x, y, 'Linear',xs)
plt.plot(xs,ys,'r',linewidth=lw)

# log power model
model, yhat_log = fit_ols(x, y, 'LogPower',x)
ssres = np.sum((y - yhat_log)**2)
cod_log = 1- (ssres/sstot)
ScaleMat.loc[yi,"CoD_log"] = cod_log
scaling_factor_log = model.params[1]
ScaleMat.loc[yi,"ScalingFactorLog"] = scaling_factor_log
scaling_rate_log = (2**scaling_factor_log) - 1
ScaleMat.loc[yi,"ScalingRateLog"] = scaling_rate_log

# plot log model
model, ys = fit_ols(x, y, 'LogPower',xs)
plt.plot(xs,ys,'m',linewidth=lw)

# power model
model, yhat_pow = fit_ols(x, y, 'Power',x)
ssres = np.sum((y - yhat_pow)**2)
cod_pow = 1- (ssres/sstot)
ScaleMat.loc[yi,"CoD_pow"] = cod_pow
scaling_factor_pow = model[2]
ScaleMat.loc[yi,"ScalingFactorPowg"] = scaling_factor_pow
scaling_rate_pow = (2**scaling_factor_pow) - 1
ScaleMat.loc[yi,"ScalingRatePow"] = scaling_rate_pow

# plot powe model
model, ys = fit_ols(x, y, 'Power',xs)
plt.plot(xs,ys,'g',linewidth=lw)



# legend
plt.legend([f"All cells (n={len(x)})",
            f"Linear model              R\u00b2={np.round(cod_lin,2)}      Scaling rate={np.round(scaling_rate_lin,2)}",
            f"Power law (log est)     R\u00b2={np.round(cod_log,2)}      Scaling rate={np.round(scaling_rate_log,2)}   Scaling factor={np.round(scaling_factor_log,2)}",
            f"Power law (curve fit)   R\u00b2={np.round(cod_pow,2)}      Scaling rate={np.round(scaling_rate_pow,2)}   Scaling factor={np.round(scaling_factor_pow,2)}"],
            loc='lower right',framealpha=1,fontsize=fs)


plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid()
plt.show()

# %%
x = np.linspace(0,1.2,1000)
y = (2**x)-1
fig = plt.figure(figsize=(5, 4))
plt.plot(x,y,'r',linewidth = 5)
plt.plot(x,x,'g--',linewidth = 1)
plt.grid()
plt.xlabel('Scaling Factor')
plt.ylabel('Scaling Rate')
plt.show()

# %%
x = np.linspace(1,2,1000)
y = (x**2)
fig = plt.figure(figsize=(5, 4))
plt.plot(x,y,'b',linewidth = 5)
plt.grid()
plt.xlabel('x')
plt.ylabel('y=x^2')
plt.show()


# %%
fs = 16
plt.rcParams.update({"font.size": fs})
ScaleMat = pd.DataFrame()
yi = 0
lw = 3
th = 0

for yi, ylabel in enumerate(FS['cellnuc_metrics']):
    x = cells['Cell volume'].squeeze().to_numpy()
    y = cells[ylabel].squeeze().to_numpy()
    loaddir = (data_root / statsIN / 'cell_nuc_metrics')
    cii = loadps(loaddir, f"Cell volume_{ylabel}_cell_dens")
    pos = np.argwhere(cii>np.percentile(cii,th))
    x = x[pos]
    y = y[pos]
    x = x.squeeze()
    y = y.squeeze()
    ScaleMat.loc[yi, "name"] = ylabel

    # make figure
    fig = plt.figure(figsize=(16, 9))

    # some stats
    ybar = np.mean(y)
    sstot = np.sum((y - ybar) ** 2)
    xs = np.linspace(1, max(x), 100)

    # plot data
    plt.plot(x, y, 'b.')

    # linear model
    model, yhat_lin = fit_ols(x, y, 'Linear', x)
    ssres = np.sum((y - yhat_lin) ** 2)
    cod_lin = 1 - (ssres / sstot)
    ScaleMat.loc[yi, "CoD_lin"] = cod_lin
    xC = cell_doubling.copy()
    xC = sm.add_constant(xC)
    yC = model.predict(xC)
    scaling_rate_lin = (yC[1] - yC[0]) / yC[0]
    ScaleMat.loc[yi, "ScalingRateLin"] = scaling_rate_lin

    # plot linear model
    model, ys = fit_ols(x, y, 'Linear', xs)
    plt.plot(xs, ys, 'r', linewidth=lw)

    # log power model
    model, yhat_log = fit_ols(x, y, 'LogPower', x)
    ssres = np.sum((y - yhat_log) ** 2)
    cod_log = 1 - (ssres / sstot)
    ScaleMat.loc[yi, "CoD_log"] = cod_log
    scaling_factor_log = model.params[1]
    ScaleMat.loc[yi, "ScalingFactorLog"] = scaling_factor_log
    scaling_rate_log = (2 ** scaling_factor_log) - 1
    ScaleMat.loc[yi, "ScalingRateLog"] = scaling_rate_log

    # plot log model
    model, ys = fit_ols(x, y, 'LogPower', xs)
    plt.plot(xs, ys, 'm', linewidth=lw)

    # power model
    model, yhat_pow = fit_ols(x, y, 'Power', x)
    ssres = np.sum((y - yhat_pow) ** 2)
    cod_pow = 1 - (ssres / sstot)
    ScaleMat.loc[yi, "CoD_pow"] = cod_pow
    scaling_factor_pow = model[2]
    ScaleMat.loc[yi, "ScalingFactorPowg"] = scaling_factor_pow
    scaling_rate_pow = (2 ** scaling_factor_pow) - 1
    ScaleMat.loc[yi, "ScalingRatePow"] = scaling_rate_pow

    # plot pow model
    model, ys = fit_ols(x, y, 'Power', xs)
    plt.plot(xs, ys, 'g', linewidth=lw)

    # legend
    plt.legend([f"All cells (n={len(x)})",
                f"Linear model              R\u00b2={np.round(cod_lin, 2)}      Scaling rate={np.round(scaling_rate_lin, 2)}",
                f"Power law (log est)     R\u00b2={np.round(cod_log, 2)}      Scaling rate={np.round(scaling_rate_log, 2)}   Scaling factor={np.round(scaling_factor_log, 2)}",
                f"Power law (curve fit)   R\u00b2={np.round(cod_pow, 2)}      Scaling rate={np.round(scaling_rate_pow, 2)}   Scaling factor={np.round(scaling_factor_pow, 2)}"],
               loc='lower right', framealpha=1, fontsize=fs)

    plt.xlabel('Cell volume')
    plt.ylabel(ylabel)
    plt.grid()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plot_save_path = pic_root / f"{ylabel}{th}.png"
    plt.savefig(plot_save_path, format="png", dpi=300)
    plt.close()
    print(ylabel)

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
        x = x.squeeze()
        y = y.squeeze()
        ScaleMat.loc[si + counter, "name"] = structures.loc[si,'Structure']

        # make figure
        fig = plt.figure(figsize=(16, 9))

        # some stats
        ybar = np.mean(y)
        sstot = np.sum((y - ybar) ** 2)
        xs = np.linspace(1, max(x), 100)

        # plot data
        plt.plot(x, y, 'b.')

        # linear model
        model, yhat_lin = fit_ols(x, y, 'Linear', x)
        ssres = np.sum((y - yhat_lin) ** 2)
        cod_lin = 1 - (ssres / sstot)
        ScaleMat.loc[si + counter, "CoD_lin"] = cod_lin
        xC = cell_doubling.copy()
        xC = sm.add_constant(xC)
        yC = model.predict(xC)
        scaling_rate_lin = (yC[1] - yC[0]) / yC[0]
        ScaleMat.loc[si + counter, "ScalingRateLin"] = scaling_rate_lin

        # plot linear model
        model, ys = fit_ols(x, y, 'Linear', xs)
        plt.plot(xs, ys, 'r', linewidth=lw)

        # log power model
        model, yhat_log = fit_ols(x, y, 'LogPower', x)
        ssres = np.sum((y - yhat_log) ** 2)
        cod_log = 1 - (ssres / sstot)
        ScaleMat.loc[si + counter, "CoD_log"] = cod_log
        scaling_factor_log = model.params[1]
        ScaleMat.loc[si + counter, "ScalingFactorLog"] = scaling_factor_log
        scaling_rate_log = (2 ** scaling_factor_log) - 1
        ScaleMat.loc[si + counter, "ScalingRateLog"] = scaling_rate_log

        # plot log model
        model, ys = fit_ols(x, y, 'LogPower', xs)
        plt.plot(xs, ys, 'm', linewidth=lw)

        # power model
        model, yhat_pow = fit_ols(x, y, 'Power', x)
        ssres = np.sum((y - yhat_pow) ** 2)
        cod_pow = 1 - (ssres / sstot)
        ScaleMat.loc[si + counter, "CoD_pow"] = cod_pow
        scaling_factor_pow = model[2]
        ScaleMat.loc[si + counter, "ScalingFactorPowg"] = scaling_factor_pow
        scaling_rate_pow = (2 ** scaling_factor_pow) - 1
        ScaleMat.loc[si + counter, "ScalingRatePow"] = scaling_rate_pow

        # plot pow model
        model, ys = fit_ols(x, y, 'Power', xs)
        plt.plot(xs, ys, 'g', linewidth=lw)

        # legend
        plt.legend([f"All cells (n={len(x)})",
                    f"Linear model              R\u00b2={np.round(cod_lin, 2)}      Scaling rate={np.round(scaling_rate_lin, 2)}",
                    f"Power law (log est)     R\u00b2={np.round(cod_log, 2)}      Scaling rate={np.round(scaling_rate_log, 2)}   Scaling factor={np.round(scaling_factor_log, 2)}",
                    f"Power law (curve fit)   R\u00b2={np.round(cod_pow, 2)}      Scaling rate={np.round(scaling_rate_pow, 2)}   Scaling factor={np.round(scaling_factor_pow, 2)}"],
                   loc='lower right', framealpha=1, fontsize=fs)

        plt.xlabel('Cell volume')
        plt.ylabel(struct)
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid()
        plot_save_path = pic_root / f"{struct}{th}.png"
        plt.savefig(plot_save_path, format="png", dpi=300)
        plt.close()
        print(struct)

# %%
xlabel = "AdultBodyMass_g"
x = animals[xlabel]
ylabel = "BasalMetRate_mLO2hr"
y = animals[ylabel]


fs = 16
plt.rcParams.update({"font.size": fs})
fig = plt.figure(figsize=(16, 9))
lw = 3

# plot data
plt.loglog(x,y,'b.')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid()
plt.show()

# %%

# %%
xlabel = "AdultBodyMass_g"
x = np.log10(animals[xlabel])
ylabel = "BasalMetRate_mLO2hr"
y = np.log10(animals[ylabel])


fs = 16
plt.rcParams.update({"font.size": fs})
fig = plt.figure(figsize=(16, 9))
lw = 3

# plot data
plt.plot(x,y,'b.')
plt.xlabel(xlabel)
plt.ylabel(ylabel)


#
# linear model
xs = np.linspace(min(x), max(x), 100)
model, yhat_lin = fit_ols(x, y, 'Linear',x)
k = model.params[1]

# plot linear model
model, ys = fit_ols(x, y, 'Linear',xs)
plt.plot(xs,ys,'r',linewidth=lw)

plt.grid()
plt.show()

# %%
# %% Compute cell volume vs. nuclear volume
x = cells['Cell volume'].to_numpy()
r = 0.1
snr = 1
# scaling
xp = cell_doubling.copy()
yp = cell_doubling.copy()
yp[0] = xp[0]
yp[1] = r*xp[0] + xp[0]
a = (yp[1]-yp[0])/(xp[1]-xp[0])
b = yp[0]-a*xp[0]
print(f"{a} {b}")
y = a*x + b
# noise injection
# xn = np.random.normal(loc=0, scale=np.sqrt((1/snr)*np.var(x)), size=x.shape)
yn = np.random.normal(loc=0, scale=np.sqrt((1/snr)*np.var(y)), size=y.shape)
xs = x
ys = y + yn

fs = 10
plt.rcParams.update({"font.size": fs})
fig, axes = plt.subplots(1,1, figsize=(6, 6), dpi=100)
axes.scatter(xs,ys)
axes.axis('equal')
axes.set_xlim(left=-2e6,right=5e6)
axes.set_ylim(bottom=-2e6,top=5e6)

plt.title(f"Scaling rate = {r}   Signal-to-Noise ratio = {snr}")
plt.grid()
plt.show()

sM = bootstrap_linear_and_log_model(xs, ys, 'None', 'None', 'Linear', cell_doubling, 'None', Nbootstrap=10)
np.mean(sM[:,1])


# %%
xx, yy = resample(xs, ys)
x_linA = xx.copy()
x_linA = sm.add_constant(x_linA)
model_lin = sm.OLS(yy, x_linA)
fittedmodel_lin = model_lin.fit()
print(fittedmodel_lin.rsquared)


# %%
fs = 10
plt.rcParams.update({"font.size": fs})
fig, axes = plt.subplots(1,1, figsize=(8, 8), dpi=100)
Rvec = np.linspace(0.01,1.2,7)
SNRvec = [100, 10, 5, 2, 1, 1/2, 1/5, 1/10, 1/100]
# SNRvec = [100, 1, 1/10000]
for i,r in enumerate(Rvec):
    xplot = np.zeros(len(SNRvec))
    yplot = np.zeros(len(SNRvec))
    xerr = np.zeros(len(SNRvec))
    yerr = np.zeros(len(SNRvec))
    for j,snr in enumerate(SNRvec):

        # scaling
        xp = cell_doubling.copy()
        yp = cell_doubling.copy()
        yp[0] = xp[0]
        yp[1] = r*xp[0] + xp[0]
        a = (yp[1]-yp[0])/(xp[1]-xp[0])
        b = yp[0]-a*xp[0]
        y = a*x + b
        # noise injection
        yn = np.random.normal(loc=0, scale=np.sqrt((1/snr)*np.var(y)), size=y.shape)
        xn = np.random.normal(loc=0, scale=np.sqrt((1/snr) * np.var(x)), size=x.shape)
        xs = x + xn
        ys = y + yn
        #bootstrap
        sM = bootstrap_linear_and_log_model(xs, ys, 'None', 'None', 'Linear', cell_doubling, 'None', Nbootstrap=10)
        # fill in
        xplot[j] = np.mean(sM[:,1])
        xerr[j] = np.std(sM[:, 1])
        yplot[j] = np.mean(sM[:, 0]/100)
        yerr[j] = np.std(sM[:, 0]/100)

    axes.scatter(xplot, yplot)
    axes.errorbar(xplot, yplot, yerr=yerr, xerr=xerr)
    axes.text(xplot[0],yplot[0]+0.02,np.round(r,2),ha='right',fontsize=16)
plt.xlabel('Explained Variance')
plt.ylabel('Scaling rate')
plt.xlim(left=0,right=1)
plt.ylim(bottom=0,top=1.3)
plt.grid()
plt.show()
