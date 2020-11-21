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


# %%
def growplot(
    axGrow,
    metric1,
    metrics,
    ScaleCurve,
    structures,
    fs,
):
    # Parameters
    tf = (0.108333 ** 3)
    lw = 3
    xlim = [0, 1]
    xticks = [0, 0.25, 0.5, 0.75, 1]
    xticklabels = ["  0", "0.25", "0.5", "0.75", "1  "]
    ylim = [-0.25, 1.25]
    yticks = [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25]
    yticklabels = ["", "0", "0.25", ".5", "0.75", "1", ""]
    yticklabelsP = [
        "",
        f"{int(100 * np.log2(1 + 0.0))}%",
        f"{int(100 * np.log2(1 + 0.25))}%",
        f"{int(100 * np.log2(1 + 0.5))}%",
        f"{int(100 * np.log2(1 + 0.75))}%",
        f"{int(100 * np.log2(1 + 1))}%",
        "",
    ]
    fac = 0.1
    markerselection = [(0, (1, 1)),'dashed',(0, (3, 1, 1, 1))]
    cfac = [0.2, 0.3, 0.4]
    perc_values = [5, 25, 50, 75, 95]

    # Grow
    xd = ScaleCurve['cell_doubling_interval'].to_numpy()
    xuplabels = np.ceil(tf*np.linspace(xd[0],xd[-1],5)).astype(np.int)
    xd = xd / xd[0]
    xd = np.log2(xd)
    yd = ScaleCurve[metric1].to_numpy()
    yd = yd / yd[0]
    yd = np.log2(yd)
    axGrow.plot(xd, yd, "k-")

    legend_handle=[]
    for j, metric2 in enumerate(metrics):

        c = structures.loc[structures["Gene"] == metric2, "Color"].item()
        color1 = mcolors.to_rgb(c)
        color2 = adjust_lightness(color1, amount=0.7)
        color3 = adjust_lightness(color1, amount=1.5)

        # color1 = np.asarray(mcolors.to_rgba(c))
        # print(color1)
        # color2 = color1
        # color2[0:2] = np.fmin(1,color2[0:2]+.5)
        # color1 = tuple(color1)
        # color2 = tuple(color2)

        ym = ScaleCurve[f"Structure volume_{metric2}_mean"].to_numpy()
        ym = ym / ym[0]
        ym = np.log2(ym)

        ymat = np.zeros((len(ym), len(perc_values)))
        for i, n in enumerate(perc_values):
            yi = ScaleCurve[f"Structure volume_{metric2}_{n}"].to_numpy()
            ymat[:, i] = yi
        ymat = ymat / ymat[0, 2]
        ymat = np.log2(ymat)

        # yv = [0, 4]
        # xf = np.concatenate((np.expand_dims(xd, axis=1), np.flipud(np.expand_dims(xd, axis=1))))
        # yf = np.concatenate((np.expand_dims(ymat[:, yv[0]], axis=1), np.flipud(np.expand_dims(ymat[:, yv[1]], axis=1))))
        # axGrow.fill(xf, yf, color=color1)

        # yv = [1, 3]
        # xf = np.concatenate(
        #     (np.expand_dims(xd, axis=1), np.flipud(np.expand_dims(xd, axis=1)))
        # )
        # yf = np.concatenate(
        #     (
        #         np.expand_dims(ymat[:, yv[0]], axis=1),
        #         np.flipud(np.expand_dims(ymat[:, yv[1]], axis=1)),
        #     )
        # )
        # axGrow.fill(xf, yf, color=[cfac[j],cfac[j],cfac[j]])
        # axGrow.plot(xd, ymat[:, yv[0]], color=color2,linewidth=1, linestyle=markerselection[j])
        # axGrow.plot(xd, ymat[:, yv[1]], color=color2,linewidth=1, linestyle=markerselection[j])

        lh, = axGrow.plot(xd, ym, color=color2, linewidth=lw, linestyle=markerselection[j],label=structures.loc[structures['Gene']==metric2,'Structure'].values[0])
        legend_handle.append(lh)

    axGrow.legend(handles=legend_handle)


        # axGrow.text(
        #     xlim[0],
        #     ylim[1] - (j * fac * (ylim[1] - ylim[0])),
        #     structures.loc[structures['Gene']==metric2,'Structure'].values[0],
        #     fontsize=fs,
        #     color=color2,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        # )

    axGrow.grid()
    axGrow.set_xlim(left=xlim[0], right=xlim[1])
    axGrow.set_xticks(xticks)
    axGrow.set_xticklabels([])
    for n, val in enumerate(xticks):
        axGrow.text(val, ylim[0], xticklabels[n], fontsize=fs,
                    horizontalalignment='center', verticalalignment='bottom', color=[0.5, 0.5, 0.5, 0.5])
        if n==0:
            axGrow.text(val, ylim[1], xuplabels[n], fontsize=fs,
                        horizontalalignment='left', verticalalignment='bottom', color='black')
        elif n==(len(xticks)-1):
            axGrow.text(val, ylim[1], xuplabels[n], fontsize=fs,
                        horizontalalignment='right', verticalalignment='bottom', color='black')
        else:
            axGrow.text(val, ylim[1], xuplabels[n], fontsize=fs,
                        horizontalalignment='center', verticalalignment='bottom', color='black')

    axGrow.text(xlim[0] - 0.16 * (xlim[1] - xlim[0]),0.5, "Scaling rate (%)", fontsize=fs,
                    horizontalalignment='center', verticalalignment='center',rotation=90)
    axGrow.set_ylim(bottom=ylim[0], top=ylim[1])
    axGrow.set_yticks(yticks)
    axGrow.set_yticklabels(yticklabelsP)
    for n, val in enumerate(yticks):
        axGrow.text(xlim[1], val, yticklabels[n], fontsize=fs,
                    horizontalalignment='right', verticalalignment='center', color=[0.5, 0.5, 0.5, 0.5])
    axGrow.text(0.5, ylim[0]-0.03*(ylim[1]-ylim[0]), "Cell volume doublings (log 2)",fontsize=fs,horizontalalignment = 'center',verticalalignment='top')

    # axGrow.set_xticklabels(xticklabels)
    # axGrow.set_ylabel("Organelle volume doublings (log 2)")
    #
    # elif side == "right":
    #     axGrow.set_xticklabels(xticklabelsR)
    #     # axGrow.set_ylabel("Organnele growth (%)")
    #     axGrow.set_ylim(bottom=ylim[0], top=ylim[1])
    #     axGrow.yaxis.set_label_position("right")
    #     axGrow.yaxis.tick_right()
    #     axGrow.set_yticks(yticks)
    #     axGrow.set_yticklabels(yticklabelsP)
