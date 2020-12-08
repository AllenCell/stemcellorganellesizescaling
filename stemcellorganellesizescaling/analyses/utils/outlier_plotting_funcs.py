#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import datetime
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib import cm
import os, platform

# Third party

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %% function defintion of regression model compensation
# %% plot function
def splot(
    selected_metrics,
    selected_metrics_abb,
    selected_structures,
    structure_metric,
    cells,
    save_flag,
    pic_root,
    name,
    markersize,
    remove_cells,
):

    #%% Rows and columns
    nrows = len(selected_metrics)
    ncols = len(selected_structures)

    #%% Plotting parameters
    fac = 1000
    ms = markersize
    fs2 = np.round(np.interp(nrows * ncols, [6, 21, 50], [25, 15, 10]))
    fs = np.round(fs2 * 2 / 3)
    lw2 = 1.5
    nbins = 100
    plt.rcParams.update({"font.size": fs})

    #%% Plotting flags
    # W = 500

    #%% Time for a flexible scatterplot
    w1 = 0.001
    w2 = 0.01
    w3 = 0.001
    h1 = 0.001
    h2 = 0.01
    h3 = 0.001
    xp = 0.1
    yp = 0.1
    xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
    xw = xx * xp
    xx = xx * (1 - xp)
    yw = yy * yp
    yy = yy * (1 - yp)

    fig = plt.figure(figsize=(16, 9))

    i = 0

    for yi, metric in enumerate(selected_metrics):
        for xi, struct in enumerate(selected_structures):

            pos = np.argwhere((cells["structure_name"] == struct).to_numpy())
            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            # selcel = (cells['structure_name'] == struct).to_numpy()
            # struct_pos = np.argwhere(selcel)
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

            metricX = metric
            metricY = struct
            abbX = selected_metrics_abb[yi]
            abbY = selected_structures[xi]

            # select subplot
            i = i + 1
            row = nrows - np.ceil(i / ncols) + 1
            row = row.astype(np.int64)
            col = i % ncols
            if col == 0:
                col = ncols
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
            ax.plot(x, y, "b.", markersize=ms)
            if len(remove_cells) > 0:
                cr = remove_cells[f"{metric} vs {structure_metric}"].astype(np.int)
                _, i_cr, _ = np.intersect1d(pos, cr, return_indices=True)
                if len(i_cr) > 0:
                    ax.plot(x[i_cr], y[i_cr], "r.", markersize=2 * ms)

            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid()

            ax.text(
                xlim[1],
                ylim[1],
                f"n= {len(x)}",
                fontsize=fs,
                verticalalignment="top",
                horizontalalignment="right",
            )
            if len(remove_cells) > 0:
                ax.text(
                    xlim[0],
                    ylim[1],
                    f"n= {len(i_cr)}",
                    fontsize=fs,
                    verticalalignment="top",
                    horizontalalignment="left",
                    color=[1, 0, 0, 1],
                )

            # Bottom histogram
            ax = fig.add_axes(
                [
                    w1 + ((col - 1) * (xw + xx + w2)) + xw,
                    h1 + ((row - 1) * (yw + yy + h2)),
                    xx,
                    yw,
                ]
            )
            ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 0.5])
            ylimBH = ax.get_ylim()
            ax.set_xticks(xticks)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xlim(left=xlim[0], right=xlim[1])
            ax.grid()
            ax.invert_yaxis()
            for n, val in enumerate(xticks):
                if val >= xlim[0] and val <= xlim[1]:
                    if int(val) == val:
                        val = int(val)
                    else:
                        val = np.round(val, 2)
                    ax.text(
                        val,
                        ylimBH[0],
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color=[0.5, 0.5, 0.5, 0.5],
                    )

            ax.text(
                np.mean(xlim),
                ylimBH[1],
                f"{abbX}",
                fontsize=fs2,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            ax.axis("off")

            # Side histogram
            ax = fig.add_axes(
                [
                    w1 + ((col - 1) * (xw + xx + w2)),
                    h1 + ((row - 1) * (yw + yy + h2)) + yw,
                    xw,
                    yy,
                ]
            )
            ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 0.5], orientation="horizontal")
            xlimSH = ax.get_xlim()
            ax.set_yticks(yticks)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
            ax.grid()
            ax.invert_xaxis()
            for n, val in enumerate(yticks):
                if val >= ylim[0] and val <= ylim[1]:
                    if int(val) == val:
                        val = int(val)
                    else:
                        val = np.round(val, 2)
                    ax.text(
                        xlimSH[0],
                        val,
                        f"{val}",
                        fontsize=fs,
                        horizontalalignment="left",
                        verticalalignment="center",
                        color=[0.5, 0.5, 0.5, 0.5],
                    )

            ax.text(
                xlimSH[1],
                np.mean(ylim),
                f"{abbY}",
                fontsize=fs2,
                horizontalalignment="left",
                verticalalignment="center",
                rotation=90,
            )
            ax.axis("off")

    if save_flag:
        plot_save_path = pic_root / f"{name}.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()


#%% function defintion
def oplot(
    cellnuc_metrics,
    cellnuc_abbs,
    pairs,
    cells,
    save_flag,
    pic_root,
    name,
    markersize,
    remove_cells,
):

    #%% Selecting number of pairs
    no_of_pairs, _ = pairs.shape
    nrows = np.floor(np.sqrt(2 / 3 * no_of_pairs))
    if nrows == 0:
        nrows = 1
    ncols = np.floor(nrows * 3 / 2)
    while nrows * ncols < no_of_pairs:
        ncols += 1

    #%% Plotting parameters
    fac = 1000
    ms = markersize
    fs2 = np.round(np.interp(nrows * ncols, [6, 21, 50], [25, 12, 8]))
    fs = np.round(fs2 * 2 / 3)
    lw2 = 1.5
    nbins = 100
    plt.rcParams.update({"font.size": fs})

    #%% Plotting flags
    # W = 500

    #%% Time for a flexible scatterplot
    w1 = 0.001
    w2 = 0.01
    w3 = 0.001
    h1 = 0.001
    h2 = 0.01
    h3 = 0.001
    xp = 0.1
    yp = 0.1
    xx = (1 - w1 - ((ncols - 1) * w2) - w3) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3) / nrows
    xw = xx * xp
    xx = xx * (1 - xp)
    yw = yy * yp
    yy = yy * (1 - yp)

    fig = plt.figure(figsize=(16, 9))

    for i, xy_pair in enumerate(pairs):

        print(i)

        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        abbX = cellnuc_abbs[xy_pair[0]]
        abbY = cellnuc_abbs[xy_pair[1]]

        # data
        x = cells[metricX].to_numpy() / fac
        y = cells[metricY].to_numpy() / fac

        # select subplot
        row = nrows - np.ceil((i + 1) / ncols) + 1
        row = row.astype(np.int64)
        col = (i + 1) % ncols
        if col == 0:
            col = ncols
        col = col.astype(np.int64)
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
        ax.plot(x, y, "b.", markersize=ms)

        if len(remove_cells) > 0:
            try:
                cr = remove_cells[f"{metricX} vs {metricY}"].astype(np.int)
            except:
                cr = remove_cells[f"{metricY} vs {metricX}"].astype(np.int)
            ax.plot(x[cr], y[cr], "r.", markersize=2 * ms)

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()

        ax.text(
            xlim[1],
            ylim[1],
            f"n= {len(x)}",
            fontsize=fs,
            verticalalignment="top",
            horizontalalignment="right",
            color=[0.75, 0.75, 0.75, 0.75],
        )
        if len(remove_cells) > 0:
            ax.text(
                xlim[0],
                ylim[1],
                f"n= {len(cr)}",
                fontsize=fs,
                verticalalignment="top",
                horizontalalignment="left",
                color=[1, 0, 0, 1],
            )

        # Bottom histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)),
                xx,
                yw,
            ]
        )
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 0.5])
        ylimBH = ax.get_ylim()
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.invert_yaxis()
        for n, val in enumerate(xticks):
            if val >= xlim[0] and val <= xlim[1]:
                if int(val) == val:
                    val = int(val)
                else:
                    val = np.round(val, 2)
                ax.text(
                    val,
                    ylimBH[0],
                    f"{val}",
                    fontsize=fs,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    color=[0.75, 0.75, 0.75, 0.75],
                )

        ax.text(
            np.mean(xlim),
            ylimBH[1],
            f"{abbX}",
            fontsize=fs2,
            horizontalalignment="center",
            verticalalignment="bottom",
        )
        ax.axis("off")

        # Side histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)),
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xw,
                yy,
            ]
        )
        ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 0.5], orientation="horizontal")
        xlimSH = ax.get_xlim()
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.invert_xaxis()
        for n, val in enumerate(yticks):
            if val >= ylim[0] and val <= ylim[1]:
                if int(val) == val:
                    val = int(val)
                else:
                    val = np.round(val, 2)
                ax.text(
                    xlimSH[0],
                    val,
                    f"{val}",
                    fontsize=fs,
                    horizontalalignment="left",
                    verticalalignment="center",
                    color=[0.75, 0.75, 0.75, 0.75],
                )

        ax.text(
            xlimSH[1],
            np.mean(ylim),
            f"{abbY}",
            fontsize=fs2,
            horizontalalignment="left",
            verticalalignment="center",
            rotation=90,
        )
        ax.axis("off")

    if save_flag:
        plot_save_path = pic_root / f"{name}.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()
