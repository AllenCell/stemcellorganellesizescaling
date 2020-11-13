#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys, importlib
from tqdm import tqdm
import pickle


# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats, explain_var_compositemodels
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression"])
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats, explain_var_compositemodels

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %% function defintion of regression model compensation
def compensate(
    dirs: list, dataset: Path, dataset_comp: Path,
):
    """
    Using a regression model, a metric is 'compensated' by a set of selected variables

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    dataset_comp: Path
        Path to CSV file with compensated feature values
    """

    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load dataset
    cells = pd.read_csv(data_root / dataset)

    #%% Select feature sets
    FS = {}
    FS['cell_metrics_AVH'] = ["Cell surface area", "Cell volume", "Cell height"]
    FS['nuc_metrics_AVH'] = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
    FS['cell_metrics_AV'] = ["Cell surface area", "Cell volume"]
    FS['nuc_metrics_AV'] = ["Nuclear surface area", "Nuclear volume"]
    FS['cell_metrics_H'] = ["Cell height"]
    FS['nuc_metrics_H'] = ["Nucleus height"]
    FS['cellnuc_metrics'] = [
        "Cell surface area",
        "Cell volume",
        "Cell height",
        "Nuclear surface area",
        "Nuclear volume",
        "Nucleus height",
        "Cytoplasmic volume",
    ]
    # struct_metrics = [
    #     "Piece CoV",
    #     "Piece sum/min/max",
    # ]

    FS['struct_metrics'] = [
        "Structure volume",
        "Number of pieces",
        "Piece average",
        "Piece std",
    ]
    # %% Compensate
    cells_COMP = cells[["CellId", "structure_name"]]

    # %% Part 1 compensate cell metrics
    for xi, xlabel in enumerate(['nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H']):
        features_4_comp = FS[xlabel]
        for yi, ylabel in enumerate(FS['cell_metrics_AVH']):
            x = cells[features_4_comp].squeeze().to_numpy()
            y = cells[ylabel].squeeze().to_numpy()
            for ti, type in enumerate(["Linear", "Complex"]):
                fittedmodel, _ = fit_ols(x, y, type)
                yr = np.expand_dims(fittedmodel.resid, axis=1)
                cells_COMP[f"{ylabel}_COMP_{type}_{xlabel}"] = yr

    # %% Part 2 compensate nuclear metrics
    for xi, xlabel in enumerate(['cell_metrics_AVH', 'cell_metrics_AV', 'cell_metrics_H']):
        features_4_comp = FS[xlabel]
        for yi, ylabel in enumerate(FS['nuc_metrics_AVH']):
            x = cells[features_4_comp].squeeze().to_numpy()
            y = cells[ylabel].squeeze().to_numpy()
            for ti, type in enumerate(["Linear", "Complex"]):
                fittedmodel, _ = fit_ols(x, y, type)
                yr = np.expand_dims(fittedmodel.resid, axis=1)
                cells_COMP[f"{ylabel}_COMP_{type}_{xlabel}"] = yr

    # %% Part 3 compensate structure metrics
    for xi, xlabel in enumerate(['cell_metrics_AVH', 'cell_metrics_AV', 'cell_metrics_H', 'nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H']):
        features_4_comp = FS[xlabel]
        for yi, ylabel in enumerate(FS['struct_metrics']):
            selected_structures = cells["structure_name"].unique()
            for si, struct in enumerate(selected_structures):
                x = (
                    cells.loc[cells["structure_name"] == struct, features_4_comp]
                        .squeeze()
                        .to_numpy()
                )
                y = (
                    cells.loc[cells["structure_name"] == struct, ylabel]
                        .squeeze()
                        .to_numpy()
                )
                for ti, type in enumerate(["Linear", "Complex"]):
                    fittedmodel, _ = fit_ols(x, y, type)
                    yr = np.expand_dims(fittedmodel.resid, axis=1)
                    cells_COMP.loc[
                        cells_COMP["structure_name"] == struct, f"{ylabel}_COMP_{type}_{xlabel}"
                    ] = yr

    # %% Save
    cells_COMP.to_csv(data_root / dataset_comp)

# %% function defintion of regression model compensation
def pairwisestats(
    dirs: list, dataset: Path, dataset_comp: Path, statsOUTdir: Path
):
    """
    Computing an array of pairwise statistics

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    dataset_comp: Path
        Path to CSV file with compensated feature values
    statsOUTdir: Path
        Path to pairwise statistics
    """

    # %%
    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load datasets
    cells = pd.read_csv(data_root / dataset)
    cells_COMP = pd.read_csv(data_root / dataset_comp)

    # Create directory to store stats
    (data_root / statsOUTdir).mkdir(exist_ok=True)

    # %% Select feature sets
    FS = {}
    FS['cell_metrics_AVH'] = ["Cell surface area", "Cell volume", "Cell height"]
    FS['nuc_metrics_AVH'] = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
    FS['cell_metrics_AV'] = ["Cell surface area", "Cell volume"]
    FS['nuc_metrics_AV'] = ["Nuclear surface area", "Nuclear volume"]
    FS['cell_metrics_H'] = ["Cell height"]
    FS['nuc_metrics_H'] = ["Nucleus height"]
    FS['cellnuc_metrics'] = [
        "Cell surface area",
        "Cell volume",
        "Cell height",
        "Nuclear surface area",
        "Nuclear volume",
        "Nucleus height",
        "Cytoplasmic volume",
    ]

    FS['pca_components'] = [
        "DNA_MEM_PC1",
        "DNA_MEM_PC2",
        "DNA_MEM_PC3",
        "DNA_MEM_PC4",
        "DNA_MEM_PC5",
        "DNA_MEM_PC6",
        "DNA_MEM_PC7",
        "DNA_MEM_PC8",
    ]


    # struct_metrics = [
    #     "Piece CoV",
    #     "Piece sum/min/max",
    # ]

    FS['struct_metrics'] = [
        "Structure volume",
        "Number of pieces",
        "Piece average",
        "Piece std",
    ]

    # %% Part 1 pairwise stats cell and nucleus measurement
    print('Cell and nucleus metrics')
    D = {}
    for xi, xlabel in enumerate(FS['cellnuc_metrics']):
        for yi, ylabel in enumerate(FS['cellnuc_metrics']):
            if xlabel is not ylabel:
                print(f"{xlabel} vs {ylabel}")
                x = cells[xlabel].squeeze().to_numpy()
                x = np.expand_dims(x, axis=1)
                y = cells[ylabel].squeeze().to_numpy()
                y = np.expand_dims(y, axis=1)
                D.update(calculate_pairwisestats(x, y, xlabel, ylabel, 'None'))
    # prepare directory
    save_dir = (data_root / statsOUTdir / 'cell_nuc_metrics')
    save_dir.mkdir(exist_ok=True)
    # save
    manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    i = 0
    for key in tqdm(D, 'saving all computed statistics'):
        pfile = save_dir / f"{key}.pickle"
        if pfile.is_file():
            pfile.unlink()
        with open(pfile, "wb") as f:
            pickle.dump(D[key], f)
        manifest.loc[i, 'pairstats'] = key
        manifest.loc[i, 'size'] = len(D[key])
        i += 1
    manifest.to_csv(save_dir / 'cell_nuc_metrics_manifest.csv')

    print('done')

    # %% Part 2 pairwise stats cell and nucleus measurement
    print('Cell and nucleus metrics vs structure metrics')
    D = {}
    for xi, xlabel in enumerate(FS['cellnuc_metrics']):
        for yi, ylabel in enumerate(FS['struct_metrics']):
            print(f"{xlabel} vs {ylabel}")
            selected_structures = cells["structure_name"].unique()
            for si, struct in enumerate(selected_structures):
                print(f"{struct}")
                x = cells.loc[cells["structure_name"] == struct, xlabel].squeeze().to_numpy()
                x = np.expand_dims(x, axis=1)
                y = cells.loc[cells["structure_name"] == struct, ylabel].squeeze().to_numpy()
                y = np.expand_dims(y, axis=1)
                D.update(calculate_pairwisestats(x, y, xlabel, ylabel, struct))
    # prepare directory
    save_dir = (data_root / statsOUTdir / 'cellnuc_struct_metrics')
    save_dir.mkdir(exist_ok=True)
    # save
    manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    i = 0
    for key in tqdm(D, 'saving all computed statistics'):
        pfile = save_dir / f"{key}.pickle"
        if pfile.is_file():
            pfile.unlink()
        with open(pfile, "wb") as f:
            pickle.dump(D[key], f)
        manifest.loc[i, 'pairstats'] = key
        manifest.loc[i, 'size'] = len(D[key])
        i += 1
    manifest.to_csv(save_dir / 'cellnuc_struct_metrics_manifest.csv')

    print('done')

    # %% Part 3 pairwise stats compensated metrics
    print('COMP Cell and nucleus metrics vs COMP structure metrics')
    D = {}
    comp_columns = list(cells_COMP.columns)
    for xi, xlabel in enumerate(
        ['nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H', 'cell_metrics_AVH', 'cell_metrics_AV',
         'cell_metrics_H']):
        for zi, zlabel in enumerate(FS['cellnuc_metrics']):
            for ti, type in enumerate(["Linear", "Complex"]):
                col2 = f"{zlabel}_COMP_{type}_{xlabel}"
                if col2 in comp_columns:
                    print(col2)
                    for yi, ylabel in enumerate(FS['struct_metrics']):
                        selected_structures = cells_COMP["structure_name"].unique()
                        for si, struct in enumerate(selected_structures):
                            col1 = f"{ylabel}_COMP_{type}_{xlabel}"
                            x = cells_COMP.loc[cells_COMP["structure_name"] == struct, col2].squeeze().to_numpy()
                            x = np.expand_dims(x, axis=1)
                            y = cells_COMP.loc[cells_COMP["structure_name"] == struct, col1].squeeze().to_numpy()
                            y = np.expand_dims(y, axis=1)
                            D.update(calculate_pairwisestats(x, y, col2, col1, struct))
    # prepare directory
    save_dir = (data_root / statsOUTdir / 'cellnuc_struct_COMP_metrics')
    save_dir.mkdir(exist_ok=True)
    # save
    manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    i = 0
    for key in tqdm(D, 'saving all computed statistics'):
        pfile = save_dir / f"{key}.pickle"
        if pfile.is_file():
            pfile.unlink()
        with open(pfile, "wb") as f:
            pickle.dump(D[key], f)
        manifest.loc[i, 'pairstats'] = key
        manifest.loc[i, 'size'] = len(D[key])
        i += 1
    manifest.to_csv(save_dir / 'cellnuc_struct_COMP_metrics_manifest.csv')

    print('done')

    # %% Part 4 pairwise stats cell and nucleus measurement and PCA components
    print('Cell and nucleus metrics VS PCA components')
    D = {}
    for xi, xlabel in enumerate(FS['pca_components']):
        for yi, ylabel in enumerate(FS['cellnuc_metrics']):
            print(f"{xlabel} vs {ylabel}")
            x = cells[xlabel].squeeze().to_numpy()
            x = np.expand_dims(x, axis=1)
            y = cells[ylabel].squeeze().to_numpy()
            y = np.expand_dims(y, axis=1)
            D.update(calculate_pairwisestats(x, y, xlabel, ylabel, 'None'))
    # prepare directory
    save_dir = (data_root / statsOUTdir / 'cell_nuc_PCA_metrics')
    save_dir.mkdir(exist_ok=True)
    # save
    manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    i = 0
    for key in tqdm(D, 'saving all computed statistics'):
        pfile = save_dir / f"{key}.pickle"
        if pfile.is_file():
            pfile.unlink()
        with open(pfile, "wb") as f:
            pickle.dump(D[key], f)
        manifest.loc[i, 'pairstats'] = key
        manifest.loc[i, 'size'] = len(D[key])
        i += 1
    manifest.to_csv(save_dir / 'cell_nuc_PCA_metrics_manifest.csv')

    print('done')

# %% function defintion of regression model compensation
def compositemodels_explainedvariance(
    dirs: list, dataset: Path, statsOUTdir: Path
):
    """
    Computing R squared (explained variance) statistics for composite models

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    statsOUTdir: Path
        Path to pairwise statistics
    """

    # %%
    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load datasets
    cells = pd.read_csv(data_root / dataset)

    # Create directory to store stats
    (data_root / statsOUTdir).mkdir(exist_ok=True)

    # %% Select feature sets
    FS = {}
    FS['cellnuc_AV'] = ["Cell surface area", "Cell volume", "Nuclear surface area", "Nuclear volume"]
    FS['cell_A'] = ["Cell volume", "Nuclear surface area", "Nuclear volume"]
    FS['cell_V'] = ["Cell surface area", "Nuclear surface area", "Nuclear volume"]
    FS['cell_AV'] = ["Nuclear surface area", "Nuclear volume"]
    FS['nuc_A'] = ["Cell surface area", "Cell volume", "Nuclear volume"]
    FS['nuc_V'] = ["Cell surface area", "Cell volume", "Nuclear surface area"]
    FS['nuc_AV'] = ["Cell surface area", "Cell volume"]

    FS['struct_metrics'] = [
        "Structure volume",
        "Number of pieces",
        "Piece average",
        "Piece std",
    ]

    # %% Part 1 R-squared metrics
    print('Computed explained variance statistics for composite models')
    D = {}
    for xi, xlabel in enumerate(
        ['cellnuc_AV', 'cell_A', 'cell_V', 'cell_AV', 'nuc_A', 'nuc_V',
         'nuc_AV']):
        features_4_comp = FS[xlabel]
        for yi, ylabel in enumerate(FS['struct_metrics']):
            selected_structures = cells["structure_name"].unique()
            for si, struct in enumerate(selected_structures):
                print(f"{xlabel}_{ylabel}_{struct}")
                x = (
                    cells.loc[cells["structure_name"] == struct, features_4_comp]
                        .squeeze()
                        .to_numpy()
                )
                y = (
                    cells.loc[cells["structure_name"] == struct, ylabel]
                        .squeeze()
                        .to_numpy()
                )
                D.update(explain_var_compositemodels(x, y, xlabel, ylabel, struct))
    # prepare directory
    save_dir = (data_root / statsOUTdir / 'struct_composite_metrics')
    save_dir.mkdir(exist_ok=True)
    # save
    manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    i = 0
    for key in tqdm(D, 'saving all computed statistics'):
        pfile = save_dir / f"{key}.pickle"
        if pfile.is_file():
            pfile.unlink()
        with open(pfile, "wb") as f:
            pickle.dump(D[key], f)
        manifest.loc[i, 'pairstats'] = key
        manifest.loc[i, 'size'] = len(D[key])
        i += 1
    manifest.to_csv(save_dir / 'struct_composite_metrics_manifest.csv')

    print('done')

# %% function defintion of regression model compensation
def scaling_stats(
    dirs: list, dataset: Path,
):
    """
    Using linear regression models, also in log-log domain, compute scaling rates. Also compute scaling curves for visualization

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    """

    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load dataset
    cells = pd.read_csv(data_root / dataset)

    #%% Select feature sets
    FS = {}
    FS['cell_metrics_AVH'] = ["Cell surface area", "Cell volume", "Cell height"]
    FS['nuc_metrics_AVH'] = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
    FS['cell_metrics_AV'] = ["Cell surface area", "Cell volume"]
    FS['nuc_metrics_AV'] = ["Nuclear surface area", "Nuclear volume"]
    FS['cell_metrics_H'] = ["Cell height"]
    FS['nuc_metrics_H'] = ["Nucleus height"]
    FS['cellnuc_metrics'] = [
        "Cell surface area",
        "Cell volume",
        "Cell height",
        "Nuclear surface area",
        "Nuclear volume",
        "Nucleus height",
        "Cytoplasmic volume",
    ]
    # struct_metrics = [
    #     "Piece CoV",
    #     "Piece sum/min/max",
    # ]

    FS['struct_metrics'] = [
        "Structure volume",
        "Number of pieces",
        "Piece average",
        "Piece std",
    ]

    # %% Parameters, updated directories
    save_flag = 0  # save plot (1) or show on screen (0)
    plt.rcParams.update({"font.size": 12})
    pic_root = pic_root / "growing"
    pic_root.mkdir(exist_ok=True)

    # %% Preprocessing
    fac = 1000
    nbins = 50
    x = cells['Cell volume'].to_numpy() / fac
    xc, xbins = np.histogram(x, bins=nbins)
    xbincenter = np.zeros((nbins, 1))
    for n in range(nbins):
        xbincenter[n] = np.mean([xbins[n], xbins[n + 1]])
    ylm = 1.05 * xc.max()
    idx = np.digitize(x, xbins)

    growfac = 2
    xstats = np.zeros((nbins, 5))
    for n in range(nbins):
        pos = np.argmin(abs(xbincenter - (xbincenter[n] * growfac)))
        xstats[n, 0] = xc[n]
        xstats[n, 1] = xc[pos]
        xstats[n, 2] = np.minimum(xc[pos], xc[n])
        xstats[n, 3] = np.sum(xc[n:pos + 1])
        xstats[n, 4] = pos

    start_bin = np.argmax(xstats[:, 2])
    end_bin = int(xstats[start_bin, 4])

    print(xbincenter[[start_bin, end_bin]])

    # %% Make signals
    cellnuc_metrics = ['Cell surface area', 'Cell volume', 'Cell height',
                       'Nuclear surface area', 'Nuclear volume', 'Nucleus height',
                       'Cytoplasmic volume']
    struct_metrics = ['Structure volume', 'Number of pieces']
    Grow = pd.DataFrame()
    perc_values = [5, 25, 50, 75, 95]

    for i, metric in tqdm(enumerate(cellnuc_metrics), 'Cell metrics'):
        y = cells[metric].to_numpy() / fac
        y_res = np.zeros((nbins, 7))
        for n in range(nbins):
            sc = np.argwhere(idx == (n + 1))
            if len(sc) < 1:
                y_res[n, :] = np.nan
                y_res[n, 0] = len(sc)
            else:
                y_res[n, 0] = len(sc)
                y_res[n, 1] = np.mean(y[sc])
                y_res[n, 2:(len(perc_values) + 2)] = np.percentile(y[sc], perc_values)
        Grow[(f"{metric}_n")] = y_res[:, 0]
        Grow[(f"{metric}_mean")] = y_res[:, 1]
        for n in np.arange(len(perc_values)):
            Grow[f"{metric}_{perc_values[n]}"] = y_res[:, n + 2]

    for i, metric in tqdm(enumerate(struct_metrics), 'Organelles'):
        y = cells[metric].to_numpy() / fac
        selected_structures = cells['structure_name'].unique()
        for si, struct in enumerate(selected_structures):
            pos = np.argwhere(cells['structure_name'].to_numpy() == struct)
            y_res = np.zeros((nbins, 7))
            for n in range(nbins):
                sc = np.argwhere(idx == (n + 1))
                sc = np.intersect1d(sc, pos)
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

    # %% Add final row with data
    gnp = Grow.to_numpy()
    growth_rates = np.divide((gnp[end_bin, :] - gnp[start_bin, :]), gnp[start_bin, :])
    growth_rates = np.round(100 * growth_rates).astype(np.int)
    Grow = Grow.append(pd.Series(), ignore_index=True)
    Grow.iloc[nbins, :] = growth_rates

    for i, metric in tqdm(enumerate(struct_metrics), 'Organelles'):
        for si, struct in enumerate(selected_structures):
            v25 = Grow[f"{metric}_{struct}_{25}"].to_numpy()
            v50 = Grow[f"{metric}_{struct}_{50}"].to_numpy()
            v75 = Grow[f"{metric}_{struct}_{75}"].to_numpy()
            f25 = np.median(np.divide(v25[start_bin:end_bin], v50[start_bin:end_bin]))
            f75 = np.median(np.divide(v75[start_bin:end_bin], v50[start_bin:end_bin]))
            Grow.loc[51, f"{metric}_{struct}_{25}"] = f25 * Grow.loc[50, f"{metric}_{struct}_{50}"]
            Grow.loc[51, f"{metric}_{struct}_{75}"] = f75 * Grow.loc[50, f"{metric}_{struct}_{50}"]
    save_dir = data_root / "growing"
    save_dir.mkdir(exist_ok=True)
    # Grow.to_csv(save_dir / "Growthstats_20201012.csv")

    # %% Add bincenters as well
    Grow.loc[np.arange(len(xbincenter)), 'bins'] = xbincenter.squeeze()
    Grow.loc[50, 'bins'] = start_bin
    Grow.loc[51, 'bins'] = end_bin

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
        y2 = 1 - y0 - y1 - h1 - h2
        x1 = 1 - w1 - w2
        lw = 1

        fig = plt.figure(figsize=(10, 10))

        # Zoom
        axZoom = fig.add_axes([w1, h1 + y2, x1, y0])
        axZoom.plot(5, 5)
        axZoom.set_xlim(left=x.min(), right=x.max())
        axZoom.set_ylim(bottom=0, top=ylm)
        axZoom.plot([x.min(), xbincenter[start_bin]], [0, ylm], 'r', linewidth=lw)
        axZoom.plot([x.max(), xbincenter[end_bin]], [0, ylm], 'r', linewidth=lw)
        axZoom.axis('off')

        # Cell Size
        axSize = fig.add_axes([w1, h1 + y2 + y0, x1, y1])
        # axSize.stem(x, max(xc)/10*np.ones(x.shape), linefmt='g-', markerfmt=None, basefmt=None)
        axSize.hist(x, bins=nbins, color=[.5, .5, .5, .5])
        axSize.grid()
        axSize.set_xlim(left=x.min(), right=x.max())
        axSize.set_ylim(bottom=0, top=ylm)
        axSize.plot(xbincenter[[start_bin, start_bin]], [0, ylm], 'r', linewidth=lw)
        axSize.plot(xbincenter[[end_bin, end_bin]], [0, ylm], 'r', linewidth=lw)
        axSize.set_xlabel('Cell size')

        # Grow
        axGrow = fig.add_axes([w1, h1, x1, y2])
        xd = Grow['Cell volume_mean'].to_numpy()
        xd = xd[start_bin:(end_bin + 1)]
        xd = xd / xd[0]
        xd = np.log2(xd)
        yd = Grow['Cell volume_mean'].to_numpy()
        yd = yd[start_bin:(end_bin + 1)]
        yd = yd / yd[0]
        yd = np.log2(yd)
        axGrow.plot(xd, yd, 'k--')

        ym = Grow[f"{dus}_mean"].to_numpy()
        ym = ym[start_bin:(end_bin + 1)]
        ym = ym / ym[0]
        ym = np.log2(ym)

        ymat = np.zeros((len(ym), len(perc_values)))
        for i, n in enumerate(perc_values):
            yi = Grow[f"{dus}_{n}"].to_numpy()
            yi = yi[start_bin:(end_bin + 1)]
            ymat[:, i] = yi
        ymat = ymat / ymat[0, 2]
        ymat = np.log2(ymat)

        yv = [0, 4]
        xf = np.concatenate((np.expand_dims(xd, axis=1), np.flipud(np.expand_dims(xd, axis=1))))
        yf = np.concatenate((np.expand_dims(ymat[:, yv[0]], axis=1), np.flipud(np.expand_dims(ymat[:, yv[1]], axis=1))))
        axGrow.fill(xf, yf, color=[0.95, 0.95, 1, 0.8])
        yv = [1, 3]
        xf = np.concatenate((np.expand_dims(xd, axis=1), np.flipud(np.expand_dims(xd, axis=1))))
        yf = np.concatenate((np.expand_dims(ymat[:, yv[0]], axis=1), np.flipud(np.expand_dims(ymat[:, yv[1]], axis=1))))
        axGrow.fill(xf, yf, color=[0.5, 0.5, 1, 0.8])

        axGrow.plot(xd, ymat[:, 2], color=[0, 0, 1, 1])
        axGrow.grid()
        axGrow.set_xlabel('Cell growth (log 2)')
        axGrow.set_ylabel('Organnele growth (log 2)')
        axGrow.set_xlim(left=0, right=np.log2(growfac))
        axGrow.set_ylim(bottom=-0.5, top=1.5)
        axGrow.text(np.log2(growfac) / 2, 1.5, dus, fontsize=20, color=[0, 0, 1, 1], verticalalignment='top',
                    horizontalalignment='center')

        if save_flag:
            plot_save_path = pic_root / f"{dus}.png"
            plt.savefig(plot_save_path, format="png", dpi=1000)
            plt.close()
        else:
            plt.show()

    # %%

    # %% Part 1 compensate cell metrics
    for xi, xlabel in enumerate(['nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H']):
        features_4_comp = FS[xlabel]
        for yi, ylabel in enumerate(FS['cell_metrics_AVH']):
            x = cells[features_4_comp].squeeze().to_numpy()
            y = cells[ylabel].squeeze().to_numpy()
            for ti, type in enumerate(["Linear", "Complex"]):
                fittedmodel, _ = fit_ols(x, y, type)
                yr = np.expand_dims(fittedmodel.resid, axis=1)
                cells_COMP[f"{ylabel}_COMP_{type}_{xlabel}"] = yr

    # %% Part 2 compensate nuclear metrics
    for xi, xlabel in enumerate(['cell_metrics_AVH', 'cell_metrics_AV', 'cell_metrics_H']):
        features_4_comp = FS[xlabel]
        for yi, ylabel in enumerate(FS['nuc_metrics_AVH']):
            x = cells[features_4_comp].squeeze().to_numpy()
            y = cells[ylabel].squeeze().to_numpy()
            for ti, type in enumerate(["Linear", "Complex"]):
                fittedmodel, _ = fit_ols(x, y, type)
                yr = np.expand_dims(fittedmodel.resid, axis=1)
                cells_COMP[f"{ylabel}_COMP_{type}_{xlabel}"] = yr

    # %% Part 3 compensate structure metrics
    for xi, xlabel in enumerate(['cell_metrics_AVH', 'cell_metrics_AV', 'cell_metrics_H', 'nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H']):
        features_4_comp = FS[xlabel]
        for yi, ylabel in enumerate(FS['struct_metrics']):
            selected_structures = cells["structure_name"].unique()
            for si, struct in enumerate(selected_structures):
                x = (
                    cells.loc[cells["structure_name"] == struct, features_4_comp]
                        .squeeze()
                        .to_numpy()
                )
                y = (
                    cells.loc[cells["structure_name"] == struct, ylabel]
                        .squeeze()
                        .to_numpy()
                )
                for ti, type in enumerate(["Linear", "Complex"]):
                    fittedmodel, _ = fit_ols(x, y, type)
                    yr = np.expand_dims(fittedmodel.resid, axis=1)
                    cells_COMP.loc[
                        cells_COMP["structure_name"] == struct, f"{ylabel}_COMP_{type}_{xlabel}"
                    ] = yr

    # %% Save
    cells_COMP.to_csv(data_root / dataset_comp)
