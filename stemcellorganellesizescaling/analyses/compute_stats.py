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
    # print('Cell and nucleus metrics')
    # D = {}
    # for xi, xlabel in enumerate(FS['cellnuc_metrics']):
    #     for yi, ylabel in enumerate(FS['cellnuc_metrics']):
    #         if xlabel is not ylabel:
    #             print(f"{xlabel} vs {ylabel}")
    #             x = cells[xlabel].squeeze().to_numpy()
    #             x = np.expand_dims(x, axis=1)
    #             y = cells[ylabel].squeeze().to_numpy()
    #             y = np.expand_dims(y, axis=1)
    #             D.update(calculate_pairwisestats(x, y, xlabel, ylabel, 'None'))
    # # prepare directory
    # save_dir = (data_root / statsOUTdir / 'cell_nuc_metrics')
    # save_dir.mkdir(exist_ok=True)
    # # save
    # manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    # i = 0
    # for key in tqdm(D, 'saving all computed statistics'):
    #     pfile = save_dir / f"{key}.pickle"
    #     if pfile.is_file():
    #         pfile.unlink()
    #     with open(pfile, "wb") as f:
    #         pickle.dump(D[key], f)
    #     manifest.loc[i, 'pairstats'] = key
    #     manifest.loc[i, 'size'] = len(D[key])
    #     i += 1
    # manifest.to_csv(save_dir / 'cell_nuc_metrics_manifest.csv')
    #
    # print('done')
    #
    # # %% Part 2 pairwise stats cell and nucleus measurement
    # print('Cell and nucleus metrics vs structure metrics')
    # D = {}
    # for xi, xlabel in enumerate(FS['cellnuc_metrics']):
    #     for yi, ylabel in enumerate(FS['struct_metrics']):
    #         print(f"{xlabel} vs {ylabel}")
    #         selected_structures = cells["structure_name"].unique()
    #         for si, struct in enumerate(selected_structures):
    #             print(f"{struct}")
    #             x = cells.loc[cells["structure_name"] == struct, xlabel].squeeze().to_numpy()
    #             x = np.expand_dims(x, axis=1)
    #             y = cells.loc[cells["structure_name"] == struct, ylabel].squeeze().to_numpy()
    #             y = np.expand_dims(y, axis=1)
    #             D.update(calculate_pairwisestats(x, y, xlabel, ylabel, struct))
    # # prepare directory
    # save_dir = (data_root / statsOUTdir / 'cellnuc_struct_metrics')
    # save_dir.mkdir(exist_ok=True)
    # # save
    # manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    # i = 0
    # for key in tqdm(D, 'saving all computed statistics'):
    #     pfile = save_dir / f"{key}.pickle"
    #     if pfile.is_file():
    #         pfile.unlink()
    #     with open(pfile, "wb") as f:
    #         pickle.dump(D[key], f)
    #     manifest.loc[i, 'pairstats'] = key
    #     manifest.loc[i, 'size'] = len(D[key])
    #     i += 1
    # manifest.to_csv(save_dir / 'cellnuc_struct_metrics_manifest.csv')
    #
    # print('done')
    #
    # # %% Part 3 pairwise stats compensated metrics
    # print('COMP Cell and nucleus metrics vs COMP structure metrics')
    # D = {}
    # comp_columns = list(cells_COMP.columns)
    # for xi, xlabel in enumerate(
    #     ['nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H', 'cell_metrics_AVH', 'cell_metrics_AV',
    #      'cell_metrics_H']):
    #     for zi, zlabel in enumerate(FS['cellnuc_metrics']):
    #         for ti, type in enumerate(["Linear", "Complex"]):
    #             col2 = f"{zlabel}_COMP_{type}_{xlabel}"
    #             if col2 in comp_columns:
    #                 print(col2)
    #                 for yi, ylabel in enumerate(FS['struct_metrics']):
    #                     selected_structures = cells_COMP["structure_name"].unique()
    #                     for si, struct in enumerate(selected_structures):
    #                         col1 = f"{ylabel}_COMP_{type}_{xlabel}"
    #                         x = cells_COMP.loc[cells_COMP["structure_name"] == struct, col2].squeeze().to_numpy()
    #                         x = np.expand_dims(x, axis=1)
    #                         y = cells_COMP.loc[cells_COMP["structure_name"] == struct, col1].squeeze().to_numpy()
    #                         y = np.expand_dims(y, axis=1)
    #                         D.update(calculate_pairwisestats(x, y, col2, col1, struct))
    # # prepare directory
    # save_dir = (data_root / statsOUTdir / 'cellnuc_struct_COMP_metrics')
    # save_dir.mkdir(exist_ok=True)
    # # save
    # manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
    # i = 0
    # for key in tqdm(D, 'saving all computed statistics'):
    #     pfile = save_dir / f"{key}.pickle"
    #     if pfile.is_file():
    #         pfile.unlink()
    #     with open(pfile, "wb") as f:
    #         pickle.dump(D[key], f)
    #     manifest.loc[i, 'pairstats'] = key
    #     manifest.loc[i, 'size'] = len(D[key])
    #     i += 1
    # manifest.to_csv(save_dir / 'cellnuc_struct_COMP_metrics_manifest.csv')
    #
    # print('done')

    # %% Part 4 pairwise stats cell and nucleus measurement and PCA components
    print('Cell and nucleus metrics VS PCA components')
    D = {}
    for xi, xlabel in enumerate(FS['cellnuc_metrics']):
        for yi, ylabel in enumerate(FS['pca_components']):
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
        ['nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H', 'cell_metrics_AVH', 'cell_metrics_AV',
         'cell_metrics_H']):
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
