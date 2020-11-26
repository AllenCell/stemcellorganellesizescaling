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
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats, explain_var_compositemodels,bootstrap_linear_and_log_model
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression"])
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats, explain_var_compositemodels,bootstrap_linear_and_log_model

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
    dirs: list, dataset: Path, dataset_comp: Path, statsOUTdir: Path, COMP_flag=True, PCA_flag=True,
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
    COMP_flag=True
        Binary flag to compute statistics for compensation analysis
    PCA_flag=True
        Binary flag to compute statistics for PCA components
    """

    # %%
    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load datasets
    cells = pd.read_csv(data_root / dataset)
    if COMP_flag is True:
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

    # FS['struct_metrics'] = [
    #     "Structure volume",
    #     "Number of pieces",
    #     "Piece average",
    #     "Piece std",
    # ]

    FS['struct_metrics'] = [
        "Structure volume",
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
    if COMP_flag is True:
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
    if PCA_flag is True:
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

    # FS['struct_metrics'] = [
    #     "Structure volume",
    #     "Number of pieces",
    #     "Piece average",
    #     "Piece std",
    # ]

    FS['struct_metrics'] = [
        "Structure volume",
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

# %% function defintion for scaling stats
def scaling_stats(
    dirs: list, dataset: Path, scaleOUTdir: Path,
):
    """
    Using linear regression models, also in log-log domain, compute scaling rates. Also compute scaling curves for visualization

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    scaleOUTdir: Path
        Path to scaling statistics
    """

    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load dataset
    cells = pd.read_csv(data_root / dataset)

    # Make savedir
    save_dir = data_root / scaleOUTdir
    save_dir.mkdir(exist_ok=True)

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

    # FS['struct_metrics'] = [
    #     "Structure volume",
    #     "Number of pieces",
    #     "Piece average",
    #     "Piece std",
    # ]

    FS['struct_metrics'] = [
        "Structure volume",
    ]

    # %% Parameters
    nbins = 50  # for doubling values estimate
    growfac = 2
    nbins2 = 25  # for plotting curve
    perc_values = [5, 25, 50, 75, 95]  # for plotting curve
    type = 'Linear'


    # %% Doubling values
    x = cells['Cell volume'].to_numpy()
    xc, xbins = np.histogram(x, bins=nbins)
    xbincenter = np.zeros((nbins, 1))
    for n in range(nbins):
        xbincenter[n] = np.mean([xbins[n], xbins[n + 1]])
    ylm = 1.05 * xc.max()
    idx = np.digitize(x, xbins)

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

    cell_doubling = xbincenter[[start_bin, end_bin]]
    cell_doubling[1] = 2 * cell_doubling[0]

    print(cell_doubling)
    D = {}
    D[f"cell_doubling"] = cell_doubling
    for key in tqdm(D, 'saving all computed statistics'):
        pfile = save_dir / f"{key}.pickle"
        if pfile.is_file():
            pfile.unlink()
        with open(pfile, "wb") as f:
            pickle.dump(D[key], f)

    # %% Compute
    ScaleMat = pd.DataFrame()
    ScaleCurve = pd.DataFrame()
    cell_doubling_interval = np.linspace(cell_doubling[0], cell_doubling[1], nbins2)
    ScaleCurve["cell_doubling_interval"] = cell_doubling_interval.squeeze()
    shift = 5 * (cell_doubling_interval[1] - cell_doubling_interval[0])
    for yi, ylabel in enumerate(FS['cellnuc_metrics']):
        x = cells['Cell volume'].squeeze().to_numpy()
        y = cells[ylabel].squeeze().to_numpy()
        scaling_stats = bootstrap_linear_and_log_model(x, y, 'Cell volume', ylabel, type, cell_doubling, 'None', Nbootstrap=100)
        ScaleMat[f"{ylabel}_prc"] = scaling_stats[:,0]
        ScaleMat[f"{ylabel}_log"] = scaling_stats[:,1]
        if yi==0:
            PM = scaling_stats
        else:
            PM = np.concatenate((PM, scaling_stats), axis=0)

        x2 = np.expand_dims(x, axis=1)
        y_res = np.zeros((nbins2, 2 + len(perc_values)))
        for j, vol in enumerate(cell_doubling_interval):
            pos = np.argwhere(
                np.all(np.concatenate((x2 >= (vol - shift), x2 <= (vol + shift)), axis=1), axis=1)).astype(
                np.int).squeeze()
            if len(pos) < 1:
                y_res[j, :] = np.nan
                y_res[j, 0] = len(pos)
            else:
                y_res[j, 0] = len(pos)
                y_res[j, 1] = np.mean(y[pos])
                y_res[j, 2:(len(perc_values) + 2)] = np.percentile(y[pos], perc_values)
        ScaleCurve[(f"{ylabel}_n")] = y_res[:, 0]
        ScaleCurve[(f"{ylabel}_mean")] = y_res[:, 1]
        for perc in np.arange(len(perc_values)):
            ScaleCurve[f"{ylabel}_{perc_values[perc]}"] = y_res[:, perc + 2]

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
            scaling_stats = bootstrap_linear_and_log_model(x, y, 'Cell volume', ylabel, type, cell_doubling, struct, Nbootstrap=100)
            ScaleMat[f"{ylabel}_{struct}prc"] = scaling_stats[:, 0]
            ScaleMat[f"{ylabel}_{struct}log"] = scaling_stats[:, 1]
            PM = np.concatenate((PM, scaling_stats), axis=0)

            x2 = np.expand_dims(x, axis=1)
            y_res = np.zeros((nbins2, 2 + len(perc_values)))
            for j, vol in enumerate(cell_doubling_interval):
                pos = np.argwhere(
                    np.all(np.concatenate((x2 >= (vol - shift), x2 <= (vol + shift)), axis=1), axis=1)).astype(
                    np.int).squeeze()
                if len(pos) < 1:
                    y_res[j, :] = np.nan
                    y_res[j, 0] = len(pos)
                else:
                    y_res[j, 0] = len(pos)
                    y_res[j, 1] = np.mean(y[pos])
                    y_res[j, 2:(len(perc_values) + 2)] = np.percentile(y[pos], perc_values)
            ScaleCurve[(f"{ylabel}_{struct}_n")] = y_res[:, 0]
            ScaleCurve[(f"{ylabel}_{struct}_mean")] = y_res[:, 1]
            for perc in np.arange(len(perc_values)):
                ScaleCurve[f"{ylabel}_{struct}_{perc_values[perc]}"] = y_res[:, perc + 2]

    # %% Saving
    ScaleMat.to_csv(save_dir / "ScaleStats_20201125.csv")
    ScaleCurve.to_csv(save_dir / "ScaleCurve_20201125.csv")

# %% function defintion for sampling
# def sampling_stats(
#     dirs: list, dataset: Path, sampleOUTdir: Path,
# ):
#     """
#     Compute explained variances and scaling rates for subsampled data for generealization figure
#
#     Parameters
#     ----------
#     dirs: list
#         Lists data and plotting dir
#     dataset: Path
#         Path to CSV file with cell by feature data table
#     sampleOUTdir: Path
#         Path to sampling statistics
#     """
#
#     # Resolve directories
#     data_root = dirs[0]
#     pic_root = dirs[1]
#
#     # Load dataset
#     cells = pd.read_csv(data_root / dataset)
#
#     # Make savedir
#     save_dir = data_root / sampleOUTdir
#     save_dir.mkdir(exist_ok=True)
#
#     # %% Select feature sets
#     FS = {}
#     FS['cellnuc_metrics'] = [
#         "Cell surface area",
#         "Cell volume",
#         "Cell height",
#         "Nuclear surface area",
#         "Nuclear volume",
#         "Nucleus height",
#         "Cytoplasmic volume",
#     ]
#
#     FS['cellnuc_AV'] = ["Cell surface area", "Cell volume", "Nuclear surface area", "Nuclear volume"]
#     FS['cell_A'] = ["Cell volume", "Nuclear surface area", "Nuclear volume"]
#     FS['cell_V'] = ["Cell surface area", "Nuclear surface area", "Nuclear volume"]
#     FS['cell_AV'] = ["Nuclear surface area", "Nuclear volume"]
#     FS['nuc_A'] = ["Cell surface area", "Cell volume", "Nuclear volume"]
#     FS['nuc_V'] = ["Cell surface area", "Cell volume", "Nuclear surface area"]
#     FS['nuc_AV'] = ["Cell surface area", "Cell volume"]
#
#     FS['struct_metrics'] = [
#         "Structure volume",
#     ]
#
#     # Setup the sampling
#     sampling_vec = [5, 10, 20, 50, 100, 200, 500, 1000]
#     sampling_N = 10
#     cells['sample_all'] = True
#     structures = cells['structure_name'].unique()
#     for i, sample in enumerate(sampling_vec):
#         for j in np.arange(sampling_N):
#             cells[f"{sample}_{j}"] = False
#             index = pd.Series([])
#             for s, struct in enumerate(structures):
#                 index = index.append(cells[cells['structure_name']==struct].sample(n=sample).index.to_series())
#             cellsN = cells.loc[index]
#
#
#
#
#
#
#
#
#
#
#     # %% Part 1 pairwise stats cell and nucleus measurement
#     print('Cell and nucleus metrics')
#     D = {}
#     for xi, xlabel in enumerate(FS['cellnuc_metrics']):
#         for yi, ylabel in enumerate(FS['cellnuc_metrics']):
#             if xlabel is not ylabel:
#                 print(f"{xlabel} vs {ylabel}")
#                 x = cells[xlabel].squeeze().to_numpy()
#                 x = np.expand_dims(x, axis=1)
#                 y = cells[ylabel].squeeze().to_numpy()
#                 y = np.expand_dims(y, axis=1)
#                 D.update(calculate_pairwisestats(x, y, xlabel, ylabel, 'None'))
#     # prepare directory
#     save_dir = (data_root / statsOUTdir / 'cell_nuc_metrics')
#     save_dir.mkdir(exist_ok=True)
#     # save
#     manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
#     i = 0
#     for key in tqdm(D, 'saving all computed statistics'):
#         pfile = save_dir / f"{key}.pickle"
#         if pfile.is_file():
#             pfile.unlink()
#         with open(pfile, "wb") as f:
#             pickle.dump(D[key], f)
#         manifest.loc[i, 'pairstats'] = key
#         manifest.loc[i, 'size'] = len(D[key])
#         i += 1
#     manifest.to_csv(save_dir / 'cell_nuc_metrics_manifest.csv')
#
#     print('done')
#
#     # %% Part 2 pairwise stats cell and nucleus measurement
#     print('Cell and nucleus metrics vs structure metrics')
#     D = {}
#     for xi, xlabel in enumerate(FS['cellnuc_metrics']):
#         for yi, ylabel in enumerate(FS['struct_metrics']):
#             print(f"{xlabel} vs {ylabel}")
#             selected_structures = cells["structure_name"].unique()
#             for si, struct in enumerate(selected_structures):
#                 print(f"{struct}")
#                 x = cells.loc[cells["structure_name"] == struct, xlabel].squeeze().to_numpy()
#                 x = np.expand_dims(x, axis=1)
#                 y = cells.loc[cells["structure_name"] == struct, ylabel].squeeze().to_numpy()
#                 y = np.expand_dims(y, axis=1)
#                 D.update(calculate_pairwisestats(x, y, xlabel, ylabel, struct))
#     # prepare directory
#     save_dir = (data_root / statsOUTdir / 'cellnuc_struct_metrics')
#     save_dir.mkdir(exist_ok=True)
#     # save
#     manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
#     i = 0
#     for key in tqdm(D, 'saving all computed statistics'):
#         pfile = save_dir / f"{key}.pickle"
#         if pfile.is_file():
#             pfile.unlink()
#         with open(pfile, "wb") as f:
#             pickle.dump(D[key], f)
#         manifest.loc[i, 'pairstats'] = key
#         manifest.loc[i, 'size'] = len(D[key])
#         i += 1
#     manifest.to_csv(save_dir / 'cellnuc_struct_metrics_manifest.csv')
#
#     print('done')
