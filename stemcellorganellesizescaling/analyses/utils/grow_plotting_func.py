#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import numpy as np
import pickle
from scipy import interpolate

# Third party

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %%
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result

# %%
def growplot(
    axGrow,
    metric1,
    metric2,
    ScaleCurve,
    structures,
    fs,
    stats_root,
):
    plt.rcParams.update({"font.size": fs})
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = "sans-serif"

    # Parameters
    tf = (0.108333 ** 3)
    lw = 3
    xlim = [0, 1]
    xticks = [0, 0.5, 1]
    ylim = [-0.25, 1.25]
    yticks = [0, 0.5, 1]
    perc_values = [5, 25, 50, 75, 95]
    c = structures.loc[structures["Gene"] == metric2, "Color"].item()
    color1 = mcolors.to_rgb(c)
    darkgreen = [0., 0.26666667, 0.10588235, 1.]

    # %% Grow
    xii = loadps(stats_root, f"{metric1}_Structure volume_{metric2}_xii")
    pred_yL = loadps(stats_root, f"{metric1}_Structure volume_{metric2}_pred_matL")
    xd = ScaleCurve['cell_doubling_interval'].to_numpy()
    cd0 = xd[0]
    cd1 = xd[-1]
    f = interpolate.interp1d(xii[:, 0], pred_yL)
    y0 = f(cd0)
    y1 = f(cd1)
    ym = ScaleCurve[f"Structure volume_{metric2}_mean"].to_numpy()
    ymat = np.zeros((len(ym), len(perc_values)))
    for i, n in enumerate(perc_values):
        yi = ScaleCurve[f"Structure volume_{metric2}_{n}"].to_numpy()
        ymat[:, i] = yi

    yPmat = (ymat-y0)/y0
    yPline = ([y0, y1]-y0)/y0
    xPvec = (xd-xd[0])/xd[0]

    yv = [1, 3]
    xf = np.concatenate(
        (np.expand_dims(xPvec, axis=1), np.flipud(np.expand_dims(xPvec, axis=1)))
    )
    yf = np.concatenate(
        (
            np.expand_dims(yPmat[:, yv[0]], axis=1),
            np.flipud(np.expand_dims(yPmat[:, yv[1]], axis=1)),
        )
    )

    # plotting
    axGrow.fill(xf, yf, color='gray')
    axGrow.plot([0, 1], [0,1], color='k', linestyle='dashed', linewidth=2.1)
    axGrow.plot([0,1], yPline, color=color1,linewidth=2)
    if metric2=='TOMM20':
        yPline[1] = yPline[1]-0.01
    axGrow.text(0.05, 0.9, f"{int(np.round(100*yPline[1]))}%",fontsize=fs,color=darkgreen)

    axGrow.text(0.5,1.3,structures.loc[structures['Gene']==metric2,'Structure'].values[0],fontsize=fs,ha='center')
    axGrow.set_xlim(left=xlim[0], right=xlim[1])
    axGrow.set_xticks(xticks)
    axGrow.set_xticklabels([])
    axGrow.set_ylim(bottom=ylim[0], top=ylim[1])
    axGrow.set_yticks(yticks)
    axGrow.set_yticklabels([])
    axGrow.grid()

