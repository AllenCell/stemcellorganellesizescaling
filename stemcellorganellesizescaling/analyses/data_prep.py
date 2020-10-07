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

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def initial_parsing(
    dirs: list, dataset: Path, dataset_snippet, dataset_filtered: Path,
):
    """
    Parses large table to get size scaling data table which contains only a subset of features

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        absolute Path to original data table csv file
    dataset_snippet: Path
        Path to snippet of original table
    dataset_filtered: Path
        Path to size scaling CSV file
    """

    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    if platform.system() == "Linux":
        # Load dataset
        cells = pd.read_csv(dataset)

        # Store snippet
        cells.sample(n=10).to_csv(data_root / dataset_snippet)

        # %% Check out columns, keep a couple
        keepcolumns = ['CellId', 'structure_name', 'mem_roundness_surface_area_lcc', 'mem_shape_volume_lcc',
                       'dna_roundness_surface_area_lcc',
                       'dna_shape_volume_lcc', 'str_connectivity_number_cc', 'str_shape_volume',
                       'mem_position_depth_lcc', 'mem_position_height_lcc', 'mem_position_width_lcc',
                       'dna_position_depth_lcc', 'dna_position_height_lcc', 'dna_position_width_lcc',
                       'DNA_MEM_PC1', 'DNA_MEM_PC2', 'DNA_MEM_PC3', 'DNA_MEM_PC4',
                       'DNA_MEM_PC5', 'DNA_MEM_PC6', 'DNA_MEM_PC7', 'DNA_MEM_PC8']
        cells = cells[keepcolumns]

        # Missing:
        #  'WorkflowId', 'meta_fov_image_date',
        # 'DNA_MEM_UMAP1', 'DNA_MEM_UMAP2'

        # %% Rename columns
        cells = cells.rename(columns={
            'mem_roundness_surface_area_lcc': 'Cell surface area',
            'mem_shape_volume_lcc': 'Cell volume',
            'dna_roundness_surface_area_lcc': 'Nuclear surface area',
            'dna_shape_volume_lcc': 'Nuclear volume',
            'str_connectivity_number_cc': 'Number of pieces',
            'str_shape_volume': 'Structure volume',
            'str_shape_volume_lcc': 'Structure volume alt',
            'mem_position_depth_lcc': 'Cell height',
            'mem_position_height_lcc': 'Cell xbox',
            'mem_position_width_lcc': 'Cell ybox',
            'dna_position_depth_lcc': 'Nucleus height',
            'dna_position_height_lcc': 'Nucleus xbox',
            'dna_position_width_lcc': 'Nucleus ybox'
        })

        # Missing:
        # 'meta_fov_image_date': 'ImageDate'

        # %% Add a column
        cells['Cytoplasmic volume'] = cells['Cell volume'] - cells['Nuclear volume']

        # %% Save
        cells.to_csv(data_root / dataset_filtered)

    else:
        print("Can only be run on Linux machine at AICS")


def outlier_removal(
    dirs: list, dataset: Path, dataset_clean: Path,
):
    """
    Removes outliers, generates diagnostic plots, saves cleaned feature data table

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    dataset_clean: Path
        Path to cleaned CSV file
    """

    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load dataset
    cells = pd.read_csv(data_root / dataset)

    # Remove outliers
    # %% Parameters, updated directories
    save_flag = 1  # save plot (1) or show on screen (0)
    pic_root = pic_root / "outlier_removal"
    pic_root.mkdir(exist_ok=True)

    # %% Yep

    # Load dataset
    cells = pd.read_csv(data_root / dataset)

    ####### Remove outliers ########

    # %% Remove some initial cells
    cells = cells[~cells["Structure volume"].isnull()]
    cells["Piece std"] = cells["Piece std"].replace(np.nan, 0)
    print(np.any(cells.isnull()))
    print(cells.shape)

    # %% Select metrics
    selected_metricsX = [
        "Cell volume",
        "Cell volume",
        "Cytoplasmic volume",
        "Cell volume",
        "Nuclear volume",
        "Cytoplasmic volume",
    ]

    selected_metricsX_abb = [
        "Cell Vol",
        "Cell Vol",
        "Cyt vol",
        "Cell Vol",
        "Nuc Vol",
        "Cyt Vol",
    ]

    selected_metricsY = [
        "Nuclear volume",
        "Cytoplasmic volume",
        "Nuclear volume",
        "Cell surface area",
        "Nuclear surface area",
        "Nuclear surface area",
    ]

    selected_metricsY_abb = [
        "Nuc Vol",
        "Cyt Vol",
        "Nuc Vol",
        "Cell Area",
        "Nuc Area",
        "Nuc Area",
    ]

    # %% Plotting parameters
    fac = 1000
    ms = 0.5
    ms2 = 3
    fs2 = 16
    lw2 = 1.5
    nbins = 100
    plt.rcParams.update({"font.size": 12})

    # %% Time for a flexible scatterplot
    nrows = 2
    ncols = 3
    w1 = 0.07
    w2 = 0.07
    w3 = 0.01
    h1 = 0.07
    h2 = 0.12
    h3 = 0.07
    xw = 0.03
    yw = 0.03
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0
    for xi, pack in enumerate(
        zip(
            selected_metricsX,
            selected_metricsY,
            selected_metricsX_abb,
            selected_metricsY_abb,
        )
    ):
        metric1 = pack[0]
        metric2 = pack[1]
        label1 = pack[2]
        label2 = pack[3]

        # data
        x = cells[metric1]
        y = cells[metric2]
        x = x / fac
        y = y / fac

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
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)

        # Bottom histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)),
                xx,
                yw,
            ]
        )
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.set_xlabel(label1, fontsize=fs2)
        ax.invert_yaxis()

        # Side histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)),
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xw,
                yy,
            ]
        )
        ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.set_ylabel(label2, fontsize=fs2)
        ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "CellNucleus_org_fine.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Time for a flexible scatterplot
    ms = 2
    nrows = 2
    ncols = 3
    w1 = 0.07
    w2 = 0.07
    w3 = 0.01
    h1 = 0.07
    h2 = 0.12
    h3 = 0.07
    xw = 0.03
    yw = 0.03
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0
    for xi, pack in enumerate(
        zip(
            selected_metricsX,
            selected_metricsY,
            selected_metricsX_abb,
            selected_metricsY_abb,
        )
    ):
        metric1 = pack[0]
        metric2 = pack[1]
        label1 = pack[2]
        label2 = pack[3]

        # data
        x = cells[metric1]
        y = cells[metric2]
        x = x / fac
        y = y / fac

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
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)

        # Bottom histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)),
                xx,
                yw,
            ]
        )
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.set_xlabel(label1, fontsize=fs2)
        ax.invert_yaxis()

        # Side histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)),
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xw,
                yy,
            ]
        )
        ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.set_ylabel(label2, fontsize=fs2)
        ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "CellNucleus_org_thick.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Parameters
    nbins = 100
    N = 5000
    Rounds = 5

    # %% Identify pairs, compute stuff and put into dicts
    selected_metrics = [
        "Cell volume",
        "Cell surface area",
        "Nuclear volume",
        "Nuclear surface area",
    ]
    Q = {}

    counter = 0
    for xi, pack in enumerate(
        zip(
            selected_metricsX,
            selected_metricsY,
            selected_metricsX_abb,
            selected_metricsY_abb,
        )
    ):
        metric1 = pack[0]
        metric2 = pack[1]
        label1 = pack[2]
        label2 = pack[3]

        print(counter)
        counter = counter + 1

        # data
        x = cells[metric1]
        y = cells[metric2]
        x = x.to_numpy()
        y = y.to_numpy()
        x = x / fac
        y = y / fac

        # sampling on x and y
        xii, yii = np.mgrid[
            x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
        ]
        xi = xii[:, 0]

        # density estimate
        for round in np.arange(Rounds):
            rs = int(datetime.datetime.utcnow().timestamp())
            xS, yS = resample(
                x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
            )
            # xS, yS = resample(x, y, replace=False, n_samples=len(x), random_state=rs)
            k = gaussian_kde(np.vstack([xS, yS]))
            zii = k(np.vstack([xii.flatten(), yii.flatten()]))
            cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
            cell_dens = cell_dens / np.sum(cell_dens)
            # make into cumulative sum
            zii = zii / np.sum(zii)
            ix = np.argsort(zii)
            zii = zii[ix]
            zii = np.cumsum(zii)
            jx = np.argsort(ix)
            zii = zii[jx]
            zii = zii.reshape(xii.shape)
            Q[f"{metric1}_{metric2}_dens_x_{round}"] = xii
            Q[f"{metric1}_{metric2}_dens_y_{round}"] = yii
            Q[f"{metric1}_{metric2}_dens_z_{round}"] = zii
            Q[f"{metric1}_{metric2}_dens_c_{round}"] = cell_dens

    # %%
    nrows = 2
    ncols = 3
    w1 = 0.07
    w2 = 0.07
    w3 = 0.01
    h1 = 0.07
    h2 = 0.12
    h3 = 0.07
    xw = 0.03
    yw = 0.03
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    dens_th = 1e-40
    remove_cells = []

    i = 0
    for xi, pack in enumerate(
        zip(
            selected_metricsX,
            selected_metricsY,
            selected_metricsX_abb,
            selected_metricsY_abb,
        )
    ):
        metric1 = pack[0]
        metric2 = pack[1]
        label1 = pack[2]
        label2 = pack[3]

        # data
        x = cells[metric1]
        y = cells[metric2]
        x = x / fac
        y = y / fac
        x = x.to_numpy()
        y = y.to_numpy()

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
        pos = []
        for round in np.arange(Rounds):
            cii = Q[f"{metric1}_{metric2}_dens_c_{round}"]
            pos = np.union1d(pos, np.argwhere(cii < dens_th))
            print(len(pos))
        print(len(pos))
        pos = pos.astype(int)
        remove_cells = np.union1d(remove_cells, pos)
        ax.plot(x[pos], y[pos], "r.", markersize=ms2)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)

        # Bottom histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)),
                xx,
                yw,
            ]
        )
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.set_xlabel(label1, fontsize=fs2)
        ax.invert_yaxis()

        # Side histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)),
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xw,
                yy,
            ]
        )
        ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.set_ylabel(label2, fontsize=fs2)
        ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "CellNucleus_outliers.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    print(len(remove_cells))

    # %%  Drop them
    cells = cells.drop(cells.index[remove_cells.astype(int)])

    # %% Time for a flexible scatterplot
    nrows = 2
    ncols = 3
    w1 = 0.07
    w2 = 0.07
    w3 = 0.01
    h1 = 0.07
    h2 = 0.12
    h3 = 0.07
    xw = 0.03
    yw = 0.03
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0
    for xi, pack in enumerate(
        zip(
            selected_metricsX,
            selected_metricsY,
            selected_metricsX_abb,
            selected_metricsY_abb,
        )
    ):
        metric1 = pack[0]
        metric2 = pack[1]
        label1 = pack[2]
        label2 = pack[3]

        # data
        x = cells[metric1]
        y = cells[metric2]
        x = x / fac
        y = y / fac

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
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)

        # Bottom histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)),
                xx,
                yw,
            ]
        )
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.set_xlabel(label1, fontsize=fs2)
        ax.invert_yaxis()

        # Side histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)),
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xw,
                yy,
            ]
        )
        ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.set_ylabel(label2, fontsize=fs2)
        ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "CellNucleus_clean_thick.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Time for a flexible scatterplot
    ms = 0.5
    nrows = 2
    ncols = 3
    w1 = 0.07
    w2 = 0.07
    w3 = 0.01
    h1 = 0.07
    h2 = 0.12
    h3 = 0.07
    xw = 0.03
    yw = 0.03
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0
    for xi, pack in enumerate(
        zip(
            selected_metricsX,
            selected_metricsY,
            selected_metricsX_abb,
            selected_metricsY_abb,
        )
    ):
        metric1 = pack[0]
        metric2 = pack[1]
        label1 = pack[2]
        label2 = pack[3]

        # data
        x = cells[metric1]
        y = cells[metric2]
        x = x / fac
        y = y / fac

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
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid()
        ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)

        # Bottom histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)) + xw,
                h1 + ((row - 1) * (yw + yy + h2)),
                xx,
                yw,
            ]
        )
        ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.grid()
        ax.set_xlabel(label1, fontsize=fs2)
        ax.invert_yaxis()

        # Side histogram
        ax = fig.add_axes(
            [
                w1 + ((col - 1) * (xw + xx + w2)),
                h1 + ((row - 1) * (yw + yy + h2)) + yw,
                xw,
                yy,
            ]
        )
        ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
        ax.set_yticks(yticks)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid()
        ax.set_ylabel(label2, fontsize=fs2)
        ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "CellNucleus_clean_fine.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Now to structures

    # %% Select metrics
    selected_metrics = [
        "Cell volume",
        "Cell surface area",
        "Nuclear volume",
        "Nuclear surface area",
    ]
    selected_metrics_abb = ["Cell Vol", "Cell Area", "Nuc Vol", "Nuc Area"]
    selected_structures = [
        "LMNB1",
        "ST6GAL1",
        "TOMM20",
        "SEC61B",
        "LAMP1",
        "RAB5A",
        "SLC25A17",
        "TUBA1B",
        "TJP1",
        "NUP153",
        "FBL",
        "NPM1",
    ]
    selected_structures_org = [
        "Nuclear envelope",
        "Golgi",
        "Mitochondria",
        "ER",
        "Lysosome",
        "Endosomes",
        "Peroxisomes",
        "Microtubules",
        "Tight junctions",
        "NPC",
        "Nucleolus F",
        "Nucleolus G",
    ]
    selected_structures_cat = [
        "Major organelle",
        "Major organelle",
        "Major organelle",
        "Major organelle",
        "Somes",
        "Somes",
        "Somes",
        "Cytoplasmic structure",
        "Cell-to-cell contact",
        "Nuclear",
        "Nuclear",
        "Nuclear",
    ]
    structure_metric = "Structure volume"

    # %% Plotting parameters
    fac = 1000
    ms = 0.5
    ms2 = 3
    ms3 = 10
    fs2 = 12
    fs3 = 17
    lw2 = 1.5
    lw3 = 3
    lw4 = 2.5
    nbins = 100
    plt.rcParams.update({"font.size": 5})

    categories = np.unique(selected_structures_cat)
    # colors = np.linspace(0, 1, len(categories))
    colors = cm.get_cmap("viridis", len(categories))
    colordict = dict(zip(categories, colors.colors))

    # %% Initial scatterplot
    nrows = len(selected_metrics)
    ncols = len(selected_structures)
    w1 = 0.027
    w2 = 0.01
    w3 = 0.002
    h1 = 0.07
    h2 = 0.08
    h3 = 0.07
    xw = 0
    yw = 0
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0

    for yi, metric in enumerate(selected_metrics):
        for xi, struct in enumerate(selected_structures):

            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            selcel = (cells["structure_name"] == struct).to_numpy()
            struct_pos = np.argwhere(selcel)
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

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
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.grid()
            if xw == 0:
                if yi == 0:
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.3 * (ylim[1] - ylim[0]),
                        struct,
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.2 * (ylim[1] - ylim[0]),
                        selected_structures_org[xi],
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.1 * (ylim[1] - ylim[0]),
                        f"n= {len(x)}",
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                if xi == 0:
                    plt.figtext(
                        0.5,
                        h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
                        metric,
                        fontsize=fs3,
                        horizontalalignment="center",
                    )
            else:
                # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
                ax.set_title(f"n= {len(x)}", fontsize=fs2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["bottom"].set_linewidth(lw3)
            ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["top"].set_linewidth(lw3)
            ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["right"].set_linewidth(lw3)
            ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["left"].set_linewidth(lw3)

            if xw != 0:
                # Bottom histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)) + xw,
                        h1 + ((row - 1) * (yw + yy + h2)),
                        xx,
                        yw,
                    ]
                )
                ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
                ax.set_xticks(xticks)
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xlim(left=xlim[0], right=xlim[1])
                ax.grid()
                # if yi==len(selected_metrics_abb):
                ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
                ax.invert_yaxis()

                # Side histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)),
                        h1 + ((row - 1) * (yw + yy + h2)) + yw,
                        xw,
                        yy,
                    ]
                )
                ax.hist(
                    y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
                )
                ax.set_yticks(yticks)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_ylim(bottom=ylim[0], top=ylim[1])
                ax.grid()
                # if xi==0:
                ax.set_ylabel(selected_structures[xi], fontsize=fs2)
                ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "Structures_org_fine.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% thick
    ms = 2
    nrows = len(selected_metrics)
    ncols = len(selected_structures)
    w1 = 0.027
    w2 = 0.01
    w3 = 0.002
    h1 = 0.07
    h2 = 0.08
    h3 = 0.07
    xw = 0
    yw = 0
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0

    for yi, metric in enumerate(selected_metrics):
        for xi, struct in enumerate(selected_structures):

            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            selcel = (cells["structure_name"] == struct).to_numpy()
            struct_pos = np.argwhere(selcel)
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

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
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.grid()
            if xw == 0:
                if yi == 0:
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.3 * (ylim[1] - ylim[0]),
                        struct,
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.2 * (ylim[1] - ylim[0]),
                        selected_structures_org[xi],
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.1 * (ylim[1] - ylim[0]),
                        f"n= {len(x)}",
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                if xi == 0:
                    plt.figtext(
                        0.5,
                        h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
                        metric,
                        fontsize=fs3,
                        horizontalalignment="center",
                    )
            else:
                # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
                ax.set_title(f"n= {len(x)}", fontsize=fs2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["bottom"].set_linewidth(lw3)
            ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["top"].set_linewidth(lw3)
            ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["right"].set_linewidth(lw3)
            ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["left"].set_linewidth(lw3)

            if xw != 0:
                # Bottom histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)) + xw,
                        h1 + ((row - 1) * (yw + yy + h2)),
                        xx,
                        yw,
                    ]
                )
                ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
                ax.set_xticks(xticks)
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xlim(left=xlim[0], right=xlim[1])
                ax.grid()
                # if yi==len(selected_metrics_abb):
                ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
                ax.invert_yaxis()

                # Side histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)),
                        h1 + ((row - 1) * (yw + yy + h2)) + yw,
                        xw,
                        yy,
                    ]
                )
                ax.hist(
                    y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
                )
                ax.set_yticks(yticks)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_ylim(bottom=ylim[0], top=ylim[1])
                ax.grid()
                # if xi==0:
                ax.set_ylabel(selected_structures[xi], fontsize=fs2)
                ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "Structures_org_thick.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Parameters
    nbins = 100
    N = 1000
    fac = 1000
    Rounds = 5

    # %% Identify pairs, compute stuff and put into dicts

    Q = {}  # 'Structure volume'
    # structure_metric = 'Number of pieces'

    for xm, metric in tqdm(enumerate(selected_metrics), "Iterating metrics"):
        for ys, struct in tqdm(enumerate(selected_structures), "and structures"):

            # data
            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

            # density estimate
            for round in np.arange(Rounds):
                rs = int(datetime.datetime.utcnow().timestamp())
                xS, yS = resample(
                    x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
                )
                k = gaussian_kde(np.vstack([xS, yS]))
                zii = k(np.vstack([xii.flatten(), yii.flatten()]))
                cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
                cell_dens = cell_dens / np.sum(cell_dens)
                # make into cumulative sum
                zii = zii / np.sum(zii)
                ix = np.argsort(zii)
                zii = zii[ix]
                zii = np.cumsum(zii)
                jx = np.argsort(ix)
                zii = zii[jx]
                zii = zii.reshape(xii.shape)
                Q[f"{metric}_{struct}_dens_x_{round}"] = xii
                Q[f"{metric}_{struct}_dens_y_{round}"] = yii
                Q[f"{metric}_{struct}_dens_z_{round}"] = zii
                Q[f"{metric}_{struct}_dens_c_{round}"] = cell_dens

    # %% Initial scatterplot
    nrows = len(selected_metrics)
    ncols = len(selected_structures)
    w1 = 0.027
    w2 = 0.01
    w3 = 0.002
    h1 = 0.07
    h2 = 0.08
    h3 = 0.07
    xw = 0
    yw = 0
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0

    dens_th = 1e-15
    remove_cells = []

    for yi, metric in enumerate(selected_metrics):
        for xi, struct in enumerate(selected_structures):

            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            selcel = (cells["structure_name"] == struct).to_numpy()
            struct_pos = np.argwhere(selcel)
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

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
            pos = []
            for round in np.arange(Rounds):
                cii = Q[f"{metric}_{struct}_dens_c_{round}"]
                pos = np.union1d(pos, np.argwhere(cii < dens_th))
                print(len(pos))
            print(len(pos))
            pos = pos.astype(int)
            remove_cells = np.union1d(remove_cells, struct_pos[pos])
            ax.plot(x[pos], y[pos], "r.", markersize=ms2)
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.grid()
            if xw == 0:
                if yi == 0:
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.3 * (ylim[1] - ylim[0]),
                        struct,
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.2 * (ylim[1] - ylim[0]),
                        selected_structures_org[xi],
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.1 * (ylim[1] - ylim[0]),
                        f"n= {len(x)}",
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                if xi == 0:
                    plt.figtext(
                        0.5,
                        h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
                        metric,
                        fontsize=fs3,
                        horizontalalignment="center",
                    )
            else:
                # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
                ax.set_title(f"n= {len(x)}", fontsize=fs2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["bottom"].set_linewidth(lw3)
            ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["top"].set_linewidth(lw3)
            ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["right"].set_linewidth(lw3)
            ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["left"].set_linewidth(lw3)

            if xw != 0:
                # Bottom histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)) + xw,
                        h1 + ((row - 1) * (yw + yy + h2)),
                        xx,
                        yw,
                    ]
                )
                ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
                ax.set_xticks(xticks)
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xlim(left=xlim[0], right=xlim[1])
                ax.grid()
                # if yi==len(selected_metrics_abb):
                ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
                ax.invert_yaxis()

                # Side histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)),
                        h1 + ((row - 1) * (yw + yy + h2)) + yw,
                        xw,
                        yy,
                    ]
                )
                ax.hist(
                    y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
                )
                ax.set_yticks(yticks)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_ylim(bottom=ylim[0], top=ylim[1])
                ax.grid()
                # if xi==0:
                ax.set_ylabel(selected_structures[xi], fontsize=fs2)
                ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "Structures_outliers.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    print(len(remove_cells))

    # %%  Drop them
    cells = cells.drop(cells.index[remove_cells.astype(int)])

    # %% Initial scatterplot
    ms = 2
    nrows = len(selected_metrics)
    ncols = len(selected_structures)
    w1 = 0.027
    w2 = 0.01
    w3 = 0.002
    h1 = 0.07
    h2 = 0.08
    h3 = 0.07
    xw = 0
    yw = 0
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0

    for yi, metric in enumerate(selected_metrics):
        for xi, struct in enumerate(selected_structures):

            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            selcel = (cells["structure_name"] == struct).to_numpy()
            struct_pos = np.argwhere(selcel)
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

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
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.grid()
            if xw == 0:
                if yi == 0:
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.3 * (ylim[1] - ylim[0]),
                        struct,
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.2 * (ylim[1] - ylim[0]),
                        selected_structures_org[xi],
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.1 * (ylim[1] - ylim[0]),
                        f"n= {len(x)}",
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                if xi == 0:
                    plt.figtext(
                        0.5,
                        h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
                        metric,
                        fontsize=fs3,
                        horizontalalignment="center",
                    )
            else:
                # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
                ax.set_title(f"n= {len(x)}", fontsize=fs2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["bottom"].set_linewidth(lw3)
            ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["top"].set_linewidth(lw3)
            ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["right"].set_linewidth(lw3)
            ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["left"].set_linewidth(lw3)

            if xw != 0:
                # Bottom histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)) + xw,
                        h1 + ((row - 1) * (yw + yy + h2)),
                        xx,
                        yw,
                    ]
                )
                ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
                ax.set_xticks(xticks)
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xlim(left=xlim[0], right=xlim[1])
                ax.grid()
                # if yi==len(selected_metrics_abb):
                ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
                ax.invert_yaxis()

                # Side histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)),
                        h1 + ((row - 1) * (yw + yy + h2)) + yw,
                        xw,
                        yy,
                    ]
                )
                ax.hist(
                    y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
                )
                ax.set_yticks(yticks)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_ylim(bottom=ylim[0], top=ylim[1])
                ax.grid()
                # if xi==0:
                ax.set_ylabel(selected_structures[xi], fontsize=fs2)
                ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "Structures_clean_thick.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Initial scatterplot
    ms = 0.5
    nrows = len(selected_metrics)
    ncols = len(selected_structures)
    w1 = 0.027
    w2 = 0.01
    w3 = 0.002
    h1 = 0.07
    h2 = 0.08
    h3 = 0.07
    xw = 0
    yw = 0
    xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
    yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows

    fig = plt.figure(figsize=(16, 9))

    i = 0

    for yi, metric in enumerate(selected_metrics):
        for xi, struct in enumerate(selected_structures):

            x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
            y = cells.loc[
                cells["structure_name"] == struct, [structure_metric]
            ].squeeze()
            selcel = (cells["structure_name"] == struct).to_numpy()
            struct_pos = np.argwhere(selcel)
            x = x.to_numpy()
            y = y.to_numpy()
            x = x / fac
            y = y / fac

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
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.grid()
            if xw == 0:
                if yi == 0:
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.3 * (ylim[1] - ylim[0]),
                        struct,
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.2 * (ylim[1] - ylim[0]),
                        selected_structures_org[xi],
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                    plt.text(
                        np.mean(xlim),
                        ylim[0] + 1.1 * (ylim[1] - ylim[0]),
                        f"n= {len(x)}",
                        fontsize=fs2,
                        horizontalalignment="center",
                    )
                if xi == 0:
                    plt.figtext(
                        0.5,
                        h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
                        metric,
                        fontsize=fs3,
                        horizontalalignment="center",
                    )
            else:
                # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
                ax.set_title(f"n= {len(x)}", fontsize=fs2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["bottom"].set_linewidth(lw3)
            ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["top"].set_linewidth(lw3)
            ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["right"].set_linewidth(lw3)
            ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
            ax.spines["left"].set_linewidth(lw3)

            if xw != 0:
                # Bottom histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)) + xw,
                        h1 + ((row - 1) * (yw + yy + h2)),
                        xx,
                        yw,
                    ]
                )
                ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
                ax.set_xticks(xticks)
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xlim(left=xlim[0], right=xlim[1])
                ax.grid()
                # if yi==len(selected_metrics_abb):
                ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
                ax.invert_yaxis()

                # Side histogram
                ax = fig.add_axes(
                    [
                        w1 + ((col - 1) * (xw + xx + w2)),
                        h1 + ((row - 1) * (yw + yy + h2)) + yw,
                        xw,
                        yy,
                    ]
                )
                ax.hist(
                    y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
                )
                ax.set_yticks(yticks)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_ylim(bottom=ylim[0], top=ylim[1])
                ax.grid()
                # if xi==0:
                ax.set_ylabel(selected_structures[xi], fontsize=fs2)
                ax.invert_xaxis()

    if save_flag:
        plot_save_path = pic_root / "Structures_clean_fine.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # Save cleaned dataset
    cells.to_csv(data_root / dataset_clean)
