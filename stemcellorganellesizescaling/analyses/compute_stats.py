#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys, importlib

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression"])
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats

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
    #     "Structure volume",
    #     "Number of pieces",
    #     "Piece average",
    #     "Piece std",
    #     "Piece CoV",
    #     "Piece sum",
    # ]
    FS['struct_metrics'] = [
        "Structure volume",
        "Number of pieces"
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

    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load datasets
    cells = pd.read_csv(data_root / dataset)
    cells_COMP = pd.read_csv(data_root / dataset_comp)

    # Create directory to store stats
    (data_root / statsOUTdir).mkdir(exist_ok=True)

    #







    PairStats = {}

    for f1, feature1 in enumerate(features_1):
        for f2, feature2 in enumerate(features_2):
            if feature1 is not feature2:
                print(f"{feature1} vs {feature2}")

                if struct_flag is False:
                    x = df_in[feature1].squeeze().to_numpy()
                    x = np.expand_dims(x, axis=1)
                    y = df_in[feature2].squeeze().to_numpy()
                    y = np.expand_dims(y, axis=1)

                    (
                        xi,
                        rs_vecL,
                        pred_matL,
                        rs_vecC,
                        pred_matC,
                        xii,
                        yii,
                        zii,
                        cell_dens,
                        x_ra,
                        y_ra,
                    ) = calculate_pairwisestats(x, y)

                    PairStats[f"{feature1}_{feature1}_xi"] = xi
                    PairStats[f"{feature1}_{feature2}_rs_vecL"] = rs_vecL
                    PairStats[f"{feature1}_{feature2}_pred_matL"] = pred_matL
                    PairStats[f"{feature1}_{feature2}_rs_vecC"] = rs_vecC
                    PairStats[f"{feature1}_{feature2}_pred_matC"] = pred_matC
                    PairStats[f"{feature1}_{feature2}_xii"] = xii
                    PairStats[f"{feature1}_{feature2}_yii"] = yii
                    PairStats[f"{feature1}_{feature2}_zii"] = zii
                    PairStats[f"{feature1}_{feature2}_cell_dens"] = cell_dens
                    PairStats[f"{feature1}_{feature2}_x_ra"] = x_ra
                    PairStats[f"{feature1}_{feature2}_y_ra"] = y_ra

                elif struct_flag is True:
                    selected_structures = df_in["structure_name"].unique()
                    for si, struct in enumerate(selected_structures):
                        print(f"{struct}")
                        x = (
                            df_in.loc[df_in["structure_name"] == struct, feature1]
                                .squeeze()
                                .to_numpy()
                        )
                        x = np.expand_dims(x, axis=1)
                        y = (
                            df_in.loc[df_in["structure_name"] == struct, feature2]
                                .squeeze()
                                .to_numpy()
                        )
                        y = np.expand_dims(y, axis=1)

                        (
                            xi,
                            rs_vecL,
                            pred_matL,
                            rs_vecC,
                            pred_matC,
                            xii,
                            yii,
                            zii,
                            cell_dens,
                            x_ra,
                            y_ra,
                        ) = calculate_pairwisestats(x, y)

                        PairStats[f"{feature1}_{feature1}_{struct}_xi"] = xi
                        PairStats[f"{feature1}_{feature2}_{struct}_rs_vecL"] = rs_vecL
                        PairStats[
                            f"{feature1}_{feature2}_{struct}_pred_matL"
                        ] = pred_matL
                        PairStats[f"{feature1}_{feature2}_{struct}_rs_vecC"] = rs_vecC
                        PairStats[
                            f"{feature1}_{feature2}_{struct}_pred_matC"
                        ] = pred_matC
                        PairStats[f"{feature1}_{feature2}_{struct}_xii"] = xii
                        PairStats[f"{feature1}_{feature2}_{struct}_yii"] = yii
                        PairStats[f"{feature1}_{feature2}_{struct}_zii"] = zii
                        PairStats[
                            f"{feature1}_{feature2}_{struct}_cell_dens"
                        ] = cell_dens
                        PairStats[f"{feature1}_{feature2}_{struct}_x_ra"] = x_ra
                        PairStats[f"{feature1}_{feature2}_{struct}_y_ra"] = y_ra

    return PairStats

