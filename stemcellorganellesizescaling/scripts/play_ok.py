#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from scipy.stats import gaussian_kde
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import os, platform
import sys, importlib

# Third party

# Relative

# Third party
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    organelle_scatter,
    fscatter,
    compensated_scatter,
    organelle_scatterT,
    compensated_scatter_t,
)

importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.scatter_plotting_func"]
)
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    organelle_scatter,
    fscatter,
    compensated_scatter,
    organelle_scatterT,
    compensated_scatter_t,
)

print("Libraries loaded successfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %%


def get_feature_sets(cells,):
    """
    Return feature sets for all plotting functions
    Parameters
    ----------
    cell: Pandas dataframe with cells
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    """
    # %% Feature sets
    FS = {}
    FS["cell_metrics_AVH"] = ["Cell surface area", "Cell volume", "Cell height"]
    FS["nuc_metrics_AVH"] = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
    FS["cell_metrics_AV"] = ["Cell surface area", "Cell volume"]
    FS["nuc_metrics_AV"] = ["Nuclear surface area", "Nuclear volume"]
    FS["cell_metrics_H"] = ["Cell height"]
    FS["nuc_metrics_H"] = ["Nucleus height"]
    FS["cellnuc_metrics"] = [
        "Cell surface area",
        "Cell volume",
        "Cell height",
        "Nuclear surface area",
        "Nuclear volume",
        "Nucleus height",
        "Cytoplasmic volume",
    ]
    FS["cellnuc_abbs"] = [
        "Cell area",
        "Cell vol",
        "Cell height",
        "Nuclear area",
        "Nuclear vol",
        "Nucleus height",
        "Cyto vol",
    ]
    FS["cellnuc_COMP_metrics"] = [
        "Cell surface area",
        "Cell volume",
        "Cell height",
        "Nuclear surface area",
        "Nuclear volume",
        "Nucleus height",
    ]

    FS["cellnuc_COMP_abbs"] = [
        "Cell area",
        "Cell vol",
        "Cell height",
        "Nuclear area",
        "Nuclear vol",
        "Nucleus height",
    ]

    FS["selected_structures"] = [
        "LMNB1",
        "ST6GAL1",
        "TOMM20",
        "SEC61B",
        "ATP2A2",
        "LAMP1",
        "RAB5A",
        "SLC25A17",
        "TUBA1B",
        "TJP1",
        "NUP153",
        "FBL",
        "NPM1",
        "SON",
    ]
    FS["other_structures"] = list(
        set(cells["structure_name"].unique()) - set(FS["selected_structures"])
    )
    # struct_metrics = [
    #     "Structure volume",
    #     "Number of pieces",
    #     "Piece average",
    #     "Piece std",
    #     "Piece CoV",
    #     "Piece sum",
    # ]

    FS["struct_metrics"] = [
        "Structure volume",
        "Number of pieces",
        "Piece average",
        "Piece std",
    ]
    FS["COMP_types"] = ["AVH", "AV", "H"]

    return FS


# %% Cell and nucleus scatter plots
# def cellnuc_scatter_plots(dirs: list, dataset: Path, statsIN: Path):
    """
    Plotting cell and nucleus metrics as scatter plots

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    statsIN: Path
        Path to pairwise statistics
    """

# %%
# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

# Load datasets
cells = pd.read_csv(data_root / dataset)
ps = data_root / statsIN / "cell_nuc_metrics"

# Get feature sets
FS = get_feature_sets(cells)

# %% Parameters, updated directories
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "scatter_plots_workflow"
pic_root.mkdir(exist_ok=True)
pic_root = pic_root / "cell_nuc_metrics"
pic_root.mkdir(exist_ok=True)

# %% One pair
xvec = [1]
yvec = [4]
pair1 = np.stack((xvec, yvec)).T

# %% 6 pairs
xvec = [1, 1, 6, 1, 4, 6]
yvec = [4, 6, 4, 0, 3, 3]
pair6 = np.stack((xvec, yvec)).T

# %% skip
N = 13
xvec = np.random.choice(len(FS["cellnuc_metrics"]), N)
yvec = np.random.choice(len(FS["cellnuc_metrics"]), N)
pairN = np.stack((xvec, yvec)).T

# %% 21 pairs
L = len(FS["cellnuc_metrics"])
pair21 = np.zeros((int(L * (L - 1) / 2), 2)).astype(np.int)
i = 0
for f1 in np.arange(L):
    for f2 in np.arange(L):
        if f2 > f1:
            pair21[i, :] = [f1, f2]
            i += 1

    # %% Test plot
    # plotname = "test"
    # fscatter(
    #     FS["cellnuc_metrics"],
    #     FS["cellnuc_abbs"],
    #     pair6,
    #     cells,
    #     ps,
    #     False,
    #     pic_root,
    #     f"{plotname}_plain",
    #     kde_flag=True,
    #     fourcolors_flag=False,
    #     colorpoints_flag=True,
    #     rollingavg_flag=True,
    #     ols_flag=True,
    #     N2=1000,
    #     plotcells=[]
    # )

    # %% Plot some more
    kde_flagL = [False, False, False, True, True, True, True, True, True]
    fourcolors_flagL = [False, False, False, False, True, False, False, True, False]
    colorpoints_flagL = [False, False, False, False, False, True, False, False, True]
    rollingavg_flagL = [False, True, True, False, False, False, True, True, True]
    ols_flagL = [False, False, True, False, False, False, True, True, True]
    Name = [
        "plain",
        "roll",
        "ols",
        "galaxy",
        "arch",
        "color",
        "galaxy_ro",
        "arch_ro",
        "color_ro",
    ]

    PS = {}
    PS["pair1"] = pair1
    PS["pair6"] = pair6
    PS["pair21"] = pair21

    for key in PS:
        pair = PS[key]
        for (
            i,
            (
                kde_flag,
                fourcolors_flag,
                colorpoints_flag,
                rollingavg_flag,
                ols_flag,
                name,
            ),
        ) in enumerate(
            zip(
                kde_flagL,
                fourcolors_flagL,
                colorpoints_flagL,
                rollingavg_flagL,
                ols_flagL,
                Name,
            )
        ):
            plotname = f"{key}_{name}"
            print(plotname)
            fscatter(
                FS["cellnuc_metrics"],
                FS["cellnuc_abbs"],
                pair,
                cells,
                ps,
                True,
                pic_root,
                plotname,
                kde_flag=kde_flag,
                fourcolors_flag=fourcolors_flag,
                colorpoints_flag=colorpoints_flag,
                rollingavg_flag=rollingavg_flag,
                ols_flag=ols_flag,
                N2=1000,
                plotcells=[],
            )


# %% Organelle scatter plots
def organelle_scatter_plots(dirs: list, dataset: Path, statsIN: Path):
    """
    Plotting cell and nucleus metrics vs organelle metrics

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    statsIN: Path
        Path to pairwise statistics
    """

    # %%
    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load datasets
    cells = pd.read_csv(data_root / dataset)
    ps = data_root / statsIN / "cellnuc_struct_metrics"

    # Get feature sets
    FS = get_feature_sets(cells)

    # %% Parameters, updated directories
    plt.rcParams.update({"font.size": 12})
    pic_root = pic_root / "scatter_plots_workflow"
    pic_root.mkdir(exist_ok=True)
    pic_root = pic_root / "cellnuc_struct_metrics"
    pic_root.mkdir(exist_ok=True)

    # %% Test plot
    # plotname = "test"
    # organelle_scatter(
    #     FS["cellnuc_metrics"],
    #     FS["cellnuc_abbs"],
    #     FS["selected_structures"],
    #     "Structure volume",
    #     cells,
    #     ps,
    #     False,
    #     pic_root,
    #     plotname,
    #     kde_flag=True,
    #     fourcolors_flag=False,
    #     colorpoints_flag=True,
    #     rollingavg_flag=True,
    #     ols_flag=True,
    #     N2=100,
    #     plotcells=[],
    # )

    # %% Plot some more
    kde_flagL = [False, False, False, True, True, True, True, True, True]
    fourcolors_flagL = [False, False, False, False, True, False, False, True, False]
    colorpoints_flagL = [False, False, False, False, False, True, False, False, True]
    rollingavg_flagL = [False, True, True, False, False, False, True, True, True]
    ols_flagL = [False, False, True, False, False, False, True, True, True]
    Name = [
        "plain",
        "roll",
        "ols",
        "galaxy",
        "arch",
        "color",
        "galaxy_ro",
        "arch_ro",
        "color_ro",
    ]

    for i in np.arange(2):
        if i == 0:
            sel_struct = FS["selected_structures"]
            key = "sel"
        elif i == 1:
            sel_struct = FS["other_structures"]
            key = "other"
        for sm, struct_metric in enumerate(FS["struct_metrics"]):
            for (
                j,
                (
                    kde_flag,
                    fourcolors_flag,
                    colorpoints_flag,
                    rollingavg_flag,
                    ols_flag,
                    name,
                ),
            ) in enumerate(
                zip(
                    kde_flagL,
                    fourcolors_flagL,
                    colorpoints_flagL,
                    rollingavg_flagL,
                    ols_flagL,
                    Name,
                )
            ):
                plotname = f"{key}_{struct_metric}_{name}"
                print(plotname)
                organelle_scatter(
                    FS["cellnuc_metrics"],
                    FS["cellnuc_abbs"],
                    sel_struct,
                    struct_metric,
                    cells,
                    ps,
                    True,
                    pic_root,
                    plotname,
                    kde_flag=kde_flag,
                    fourcolors_flag=fourcolors_flag,
                    colorpoints_flag=colorpoints_flag,
                    rollingavg_flag=rollingavg_flag,
                    ols_flag=ols_flag,
                    N2=100,
                    plotcells=[],
                )


# %% Organelle scatter plots
def organelle_compensated_scatter_plots(
    dirs: list, dataset: Path, dataset_comp: Path, statsIN: Path
):
    """
    Plotting cell and nucleus metrics vs organelle metrics

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        Path to CSV file with cell by feature data table
    dataset_comp: Path
        Path to CSV file with compensated feature values
    statsIN: Path
        Path to pairwise statistics
    """

    # %%
    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load datasets
    cells = pd.read_csv(data_root / dataset)
    cells_COMP = pd.read_csv(data_root / dataset_comp)
    ps = data_root / statsIN / "cellnuc_struct_COMP_metrics"

    # Get feature sets
    FS = get_feature_sets(cells)

    # %% Parameters, updated directories
    plt.rcParams.update({"font.size": 12})
    pic_root = pic_root / "scatter_plots_workflow"
    pic_root.mkdir(exist_ok=True)
    pic_root = pic_root / "cellnuc_struct_COMP_metrics"
    pic_root.mkdir(exist_ok=True)

    # %% Test plot
    # plotname = "test"
    # compensated_scatter(
    #     FS["cellnuc_COMP_metrics"],
    #     FS["cellnuc_COMP_abbs"],
    #     FS["selected_structures"],
    #     "AVH",
    #     'Linear',
    #     "Structure volume",
    #     cells_COMP,
    #     ps,
    #     False,
    #     pic_root,
    #     plotname,
    #     kde_flag=True,
    #     fourcolors_flag=False,
    #     colorpoints_flag=True,
    #     rollingavg_flag=True,
    #     ols_flag=True,
    #     N2=100,
    # )

    # %% Plot some more
    kde_flagL = [False, False, False, True, True, True, True, True, True]
    fourcolors_flagL = [False, False, False, False, True, False, False, True, False]
    colorpoints_flagL = [False, False, False, False, False, True, False, False, True]
    rollingavg_flagL = [False, True, True, False, False, False, True, True, True]
    ols_flagL = [False, False, True, False, False, False, True, True, True]
    Name = [
        "plain",
        "roll",
        "ols",
        "galaxy",
        "arch",
        "color",
        "galaxy_ro",
        "arch_ro",
        "color_ro",
    ]

    for c, comp_type, in enumerate(FS["COMP_types"]):
        for ti, lin_type in enumerate(["Linear", "Complex"]):
            for i in np.arange(2):
                if i == 0:
                    sel_struct = FS["selected_structures"]
                    key = "sel"
                elif i == 1:
                    sel_struct = FS["other_structures"]
                    key = "other"
                for sm, struct_metric in enumerate(FS["struct_metrics"]):
                    for (
                        j,
                        (
                            kde_flag,
                            fourcolors_flag,
                            colorpoints_flag,
                            rollingavg_flag,
                            ols_flag,
                            name,
                        ),
                    ) in enumerate(
                        zip(
                            kde_flagL,
                            fourcolors_flagL,
                            colorpoints_flagL,
                            rollingavg_flagL,
                            ols_flagL,
                            Name,
                        )
                    ):
                        plotname = (
                            f"{key}_{struct_metric}_{lin_type}_{comp_type}_{name}"
                        )
                        print(plotname)

                        compensated_scatter(
                            FS["cellnuc_COMP_metrics"],
                            FS["cellnuc_COMP_abbs"],
                            FS["selected_structures"],
                            comp_type,
                            lin_type,
                            struct_metric,
                            cells_COMP,
                            ps,
                            True,
                            pic_root,
                            plotname,
                            kde_flag=kde_flag,
                            fourcolors_flag=fourcolors_flag,
                            colorpoints_flag=colorpoints_flag,
                            rollingavg_flag=rollingavg_flag,
                            ols_flag=ols_flag,
                            N2=100,
                        )


# %% Extra
#%%
# plotname = "x"
# ps = data_root / statsIN / "cellnuc_struct_metrics"
# pic_rootT = pic_root / "forpres"
# pic_rootT.mkdir(exist_ok=True)
# organelle_scatterT(
#     FS["cellnuc_metrics"],
#     FS["cellnuc_abbs"],
#     ['ST6GAL1','SON'],
#     "Structure volume",
#     cells,
#     ps,
#     True,
#     pic_rootT,
#     plotname,
#     kde_flag=True,
#     fourcolors_flag=False,
#     colorpoints_flag=True,
#     rollingavg_flag=True,
#     ols_flag=True,
#     N2=100,
#     plotcells=bcells,
# )
#
# #%%
# plotname = "x_c"
# pic_rootT = pic_root / "forpres"
# pic_rootT.mkdir(exist_ok=True)
# ps = data_root / statsIN / "cellnuc_struct_COMP_metrics"
# compensated_scatter_t(
#     FS["cellnuc_COMP_metrics"],
#     FS["cellnuc_COMP_abbs"],
#     ['ST6GAL1','SON'],
#     "AVH",
#     'Linear',
#     "Structure volume",
#     cells_COMP,
#     ps,
#     True,
#     pic_rootT,
#     plotname,
#     kde_flag=True,
#     fourcolors_flag=False,
#     colorpoints_flag=True,
#     rollingavg_flag=True,
#     ols_flag=True,
#     N2=100,
# )
