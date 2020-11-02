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
    Grow,
    start_bin,
    end_bin,
    perc_values,
    growfac,
    structures,
    side,
):
    # Parameters
    lw = 3
    xlim = [0, 1]
    xticks = [0, 0.25, 0.5, 0.75, 1]
    xticklabels = ["0", "0.25", "0.5", "0.75", "1"]
    xticklabelsR = ["", "0.25", "0.5", "0.75", "1"]
    ylim = [-0.25, 1.25]
    yticks = [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25]
    yticklabels = ["", "0", "", ".5", "", "1", ""]
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

    # Grow
    xd = Grow[metric1].to_numpy()
    xd = xd[start_bin : (end_bin + 1)]
    xd = xd / xd[0]
    xd = np.log2(xd)
    yd = Grow[metric1].to_numpy()
    yd = yd[start_bin : (end_bin + 1)]
    yd = yd / yd[0]
    yd = np.log2(yd)
    axGrow.plot(xd, yd, "k--")

    for j, metric2 in enumerate(metrics):

        c = structures.loc[structures["Gene"] == metric2, "Color"].item()
        color1 = mcolors.to_rgb(c)
        color2 = adjust_lightness(color1, amount=1.5)

        # color1 = np.asarray(mcolors.to_rgba(c))
        # print(color1)
        # color2 = color1
        # color2[0:2] = np.fmin(1,color2[0:2]+.5)
        # color1 = tuple(color1)
        # color2 = tuple(color2)

        ym = Grow[f"Structure volume_{metric2}_mean"].to_numpy()
        ym = ym[start_bin : (end_bin + 1)]
        ym = ym / ym[0]
        ym = np.log2(ym)

        ymat = np.zeros((len(ym), len(perc_values)))
        for i, n in enumerate(perc_values):
            yi = Grow[f"Structure volume_{metric2}_{n}"].to_numpy()
            yi = yi[start_bin : (end_bin + 1)]
            ymat[:, i] = yi
        ymat = ymat / ymat[0, 2]
        ymat = np.log2(ymat)

        # yv = [0, 4]
        # xf = np.concatenate((np.expand_dims(xd, axis=1), np.flipud(np.expand_dims(xd, axis=1))))
        # yf = np.concatenate((np.expand_dims(ymat[:, yv[0]], axis=1), np.flipud(np.expand_dims(ymat[:, yv[1]], axis=1))))
        # axGrow.fill(xf, yf, color=color1)
        yv = [1, 3]
        xf = np.concatenate(
            (np.expand_dims(xd, axis=1), np.flipud(np.expand_dims(xd, axis=1)))
        )
        yf = np.concatenate(
            (
                np.expand_dims(ymat[:, yv[0]], axis=1),
                np.flipud(np.expand_dims(ymat[:, yv[1]], axis=1)),
            )
        )
        axGrow.fill(xf, yf, color=color2, alpha=0.5)

        axGrow.plot(xd, ymat[:, 2], color=color1, linewidth=lw)

        axGrow.text(
            xlim[0],
            ylim[1] - (j * fac * (ylim[1] - ylim[0])),
            metric2,
            fontsize=20,
            color=color1,
            verticalalignment="top",
            horizontalalignment="left",
        )

    axGrow.grid()
    axGrow.set_xlabel("Cell growth (log 2)")
    axGrow.set_xlim(left=xlim[0], right=xlim[1])
    axGrow.set_xticks(xticks)
    if side == "left":
        axGrow.set_xticklabels(xticklabels)
        axGrow.set_ylabel("Organnele growth (log 2)")
        axGrow.set_ylim(bottom=ylim[0], top=ylim[1])
        axGrow.set_yticks(yticks)
        axGrow.set_yticklabels(yticklabels)
    elif side == "right":
        axGrow.set_xticklabels(xticklabelsR)
        # axGrow.set_ylabel("Organnele growth (%)")
        axGrow.set_ylim(bottom=ylim[0], top=ylim[1])
        axGrow.yaxis.set_label_position("right")
        axGrow.yaxis.tick_right()
        axGrow.set_yticks(yticks)
        axGrow.set_yticklabels(yticklabelsP)
