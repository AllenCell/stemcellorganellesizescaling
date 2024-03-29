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


def initial_parsing(
    dirs: list,
    dataset: Path,
    old_dataset: Path,
    piecedir: Path,
    pca_dataset: Path,
    dataset_snippet: Path,
    dataset_filtered: Path,
):
    """
    Parses large table to get size scaling data table which contains only a subset of features

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        absolute Path to original data table csv file
    dataset_old: Path
        absolute Path to old data table csv file
    piecedir: Path
        absolute Path (folder) to original piece stats
    pca_dataset: Path
        absolute Path to original data table csv file with the shape mode coefficients
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
        #DNA_MEM_PC1 - missing
        keepcolumns = [
            "CellId",
            "structure_name",
            "MEM_roundness_surface_area",
            "MEM_shape_volume",
            "NUC_roundness_surface_area",
            "NUC_shape_volume",
            "STR_connectivity_cc",
            "STR_shape_volume",
            "MEM_position_depth",
            "MEM_position_height",
            "MEM_position_width",
            "NUC_position_depth",
            "NUC_position_height",
            "NUC_position_width",
            "WorkflowId",
            "meta_imaging_mode",
        ]
        cells = cells[keepcolumns]

        # Missing:
        # 'DNA_MEM_UMAP1', 'DNA_MEM_UMAP2'

        # %%
        # "meta_fov_image_date"
        # Load old dataset to get time in there
        old_cells = pd.read_csv(old_dataset)
        cells = cells.merge(old_cells[['CellId','meta_fov_image_date']],how='left',on='CellId')

        # Load PCA components
        pca_cells = pd.read_csv(pca_dataset)
        pca_cols = [f'NUC_MEM_PC{x+1}' for x in np.arange(8)]
        cells = cells.merge(pca_cells[['CellId',*pca_cols]],how='left',on='CellId')

# %% Rename columns
        cells = cells.rename(
            columns={
                "MEM_roundness_surface_area": "Cell surface area",
                "MEM_shape_volume": "Cell volume",
                "NUC_roundness_surface_area": "Nuclear surface area",
                "NUC_shape_volume": "Nuclear volume",
                "STR_connectivity_cc": "Number of pieces",
                "STR_shape_volume": "Structure volume",
                "MEM_position_depth": "Cell height",
                "MEM_position_height": "Cell xbox",
                "MEM_position_width": "Cell ybox",
                "NUC_position_depth": "Nucleus height",
                "NUC_position_height": "Nucleus xbox",
                "NUC_position_width": "Nucleus ybox",
                "meta_fov_image_date": "ImageDate",
            }
        )

        # %% Add a column
        cells["Cytoplasmic volume"] = cells["Cell volume"] - cells["Nuclear volume"]

        # # %% Adding feature pieces
        # paths = Path(piecedir).glob("**/*.csv")
        # cells["Piece average"] = np.nan
        # cells["Piece max"] = np.nan
        # cells["Piece min"] = np.nan
        # cells["Piece std"] = np.nan
        # cells["Piece sum"] = np.nan
        # cells.set_index("CellId", drop=False, inplace=True)
        # for csvf in paths:
        #     print(csvf)
        #     pieces = pd.read_csv(csvf)
        #     keepcolumns = [
        #         "CellId",
        #         "str_shape_volume_pcc_avg",
        #         "str_shape_volume_pcc_max",
        #         "str_shape_volume_pcc_min",
        #         "str_shape_volume_pcc_std",
        #         "str_shape_volume_pcc_sum",
        #     ]
        #     pieces = pieces[keepcolumns]
        #     pieces = pieces.rename(
        #         columns={
        #             "str_shape_volume_pcc_avg": "Piece average",
        #             "str_shape_volume_pcc_max": "Piece max",
        #             "str_shape_volume_pcc_min": "Piece min",
        #             "str_shape_volume_pcc_std": "Piece std",
        #             "str_shape_volume_pcc_sum": "Piece sum",
        #         }
        #     )
        #     pieces.set_index("CellId", drop=False, inplace=True)
        #     cells.update(pieces)

        # %% Post-processing and checking
        # sv = cells["Structure volume"].to_numpy()
        # ps = cells["Piece sum"].to_numpy()
        # sn = cells["structure_name"].to_numpy()
        # pos = np.argwhere(np.divide(abs(ps - sv), sv) > 0)
        # print(f"{len(pos)} mismatches in {np.unique(sn[pos])}")
        # posS = np.argwhere(np.divide(abs(ps - sv), sv) > 0.01)
        # print(f"{len(posS)} larger than 1%")
        # posT = np.argwhere(np.divide(abs(ps - sv), sv) > 0.1)
        # print(f"{len(posT)} larger than 10%")
        # # cells.drop(labels='Unnamed: 0', axis=1, inplace=True)
        # cells.reset_index(drop=True, inplace=True)
        # print(np.any(cells.isnull()))
        # cells.loc[cells["Piece std"].isnull(), "Piece std"] = 0
        # print(np.any(cells.isnull()))

        # %% Save
        cells.to_csv(data_root / dataset_filtered)

    else:
        print("Can only be run on Linux machine at AICS")


def diagnostic_violins(
    dirs: list, dataset: Path,
):
    """
    Creates a bunch of diagnostic plots for the cleaned dataset

    Parameters
    ----------
    dirs: list
        Lists data and plotting dir
    dataset: Path
        absolute Path to original data table csv file
    """

    # Resolve directories
    data_root = dirs[0]
    pic_root = dirs[1]

    # Load dataset
    cells = pd.read_csv(data_root / dataset)

    # Drop cells with imagedata
    print(len(cells))
    cells.dropna(axis=0,subset=['ImageDate'],inplace=True)
    print(len(cells))

    # %% Parameters, updated directories
    save_flag = 1  # save plot (1) or show on screen (0)
    pic_root = pic_root / "diagnostic_violins"
    pic_root.mkdir(exist_ok=True)

    # %% Time vs. structure
    timestr = cells["ImageDate"]
    time = np.zeros((len(timestr), 1))
    for i, val in tqdm(enumerate(timestr)):
        date_time_obj = datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
        # time[i] = int(date_time_obj.strftime("%Y%m%d%H%M%S"))
        time[i] = int(date_time_obj.timestamp())
    cells["int_acquisition_time"] = time

    # %% Plot time
    fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
    axes.hist(cells["int_acquisition_time"], bins=200)
    locs, labels = plt.xticks()
    for i, val in enumerate(locs):
        date_time_obj = datetime.fromtimestamp(val)
        labels[i] = date_time_obj.strftime("%b%y")
    plt.xticks(locs, labels)

    axes.set_title("Cells over time")
    axes.grid(True, which="major", axis="y")
    axes.set_axisbelow(True)

    if save_flag:
        plot_save_path = pic_root / "HISTOGRAM_CellsOverTime.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Order of structures and FOVs
    table = pd.pivot_table(
        cells, index="structure_name", values="int_acquisition_time", aggfunc="min"
    )
    table = table.sort_values(by=["int_acquisition_time"])
    sortedStructures = table.index.values

    # %% Plot structures over time
    fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
    sns.violinplot(
        y="structure_name",
        x="int_acquisition_time",
        data=cells,
        ax=axes,
        order=sortedStructures,
    )
    locs, labels = plt.xticks()
    for i, val in enumerate(locs):
        date_time_obj = datetime.fromtimestamp(val)
        labels[i] = date_time_obj.strftime("%b%y")
    plt.xticks(locs, labels)

    axes.set_title("Structures over time")
    axes.grid(True, which="major", axis="both")
    axes.set_axisbelow(True)
    axes.set_ylabel(None)
    axes.set_xlabel(None)

    if save_flag:
        plot_save_path = pic_root / "VIOLIN_structure_vs_time.png"
        plt.savefig(plot_save_path, format="png", dpi=300)
        plt.close()
    else:
        plt.show()

    # %% Bars with numbers of cells for each of the structures
    table = pd.pivot_table(cells, index="structure_name", aggfunc="size")
    table = table.reindex(sortedStructures)
    fig, axes = plt.subplots(figsize=(10, 5), dpi=100)

    table.plot.barh(ax=axes)
    # x_pos = range(len(table))
    # plt.barh(x_pos, table)
    # plt.yticks(x_pos, table.keys())

    for j, val in enumerate(table):
        axes.text(
            val,
            j,
            str(val),
            ha="right",
            va="center",
            color="white",
            size=6,
            weight="bold",
        )

    axes.set_title("Number of cells per structure")
    axes.set_ylabel(None)
    axes.grid(True, which="major", axis="x")
    axes.set_axisbelow(True)
    axes.invert_yaxis()

    if save_flag:
        plot_save_path = pic_root / "BAR_StructureCounts.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Stacked bars comparing stuctures and imaging mode
    table = pd.pivot_table(
        cells, index="structure_name", columns="WorkflowId", aggfunc="size"
    )
    table = table.reindex(sortedStructures)
    fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
    table.plot.barh(stacked=True, ax=axes)

    axes.set_ylabel(None)
    axes.set_title("Structures and Image Mode")
    axes.grid(True, which="major", axis="x")
    axes.set_axisbelow(True)
    axes.invert_yaxis()

    if save_flag:
        plot_save_path = pic_root / "BAR_StructureVsImageMode.png"
        plt.savefig(plot_save_path, format="png", dpi=1000)
        plt.close()
    else:
        plt.show()

    # %% Plot structure over FOVids
    # Still missing:
    # 'DNA_MEM_UMAP1', 'DNA_MEM_UMAP2',
    # DNA_MEM_PC1

    selected_metrics = [
        "Cell surface area",
        "Cell volume",
        "Nuclear surface area",
        "Nuclear volume",
        "Cytoplasmic volume",
        "Number of pieces",
        "Structure volume",
        "Cell height",
        "Cell xbox",
        "Cell ybox",
        "Nucleus height",
        "Nucleus xbox",
        "Nucleus ybox",
        "NUC_MEM_PC1",
        "NUC_MEM_PC2",
        "NUC_MEM_PC3",
        "NUC_MEM_PC4",
        "NUC_MEM_PC5",
        "NUC_MEM_PC6",
        "NUC_MEM_PC7",
        "NUC_MEM_PC8",
    ]

    for i, metric in enumerate(selected_metrics):

        fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
        sns.violinplot(
            y="structure_name",
            x=metric,
            color="black",
            data=cells,
            scale="width",
            ax=axes,
            order=sortedStructures,
        )

        axes.set_title(f"{metric} across cell lines")
        axes.grid(True, which="major", axis="both")
        axes.set_axisbelow(True)
        axes.set_ylabel(None)
        axes.set_xlabel(None)

        if save_flag:
            plot_save_path = pic_root / f"VIOLIN_{metric}.png"
            plt.savefig(plot_save_path, format="png", dpi=300)
            plt.close()
        else:
            plt.show()


def outlier_removal(
    dirs: list, dataset: Path, dataset_clean: Path, dataset_outliers: Path,
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
    dataset_outliers: Path
        Path to outlier annotation CSV (same number of cells as dataset)
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
    data_root_extra = data_root / "outlier_removal"
    data_root_extra.mkdir(exist_ok=True)

    # %% Threshold for determing outliers
    cell_dens_th_CN = 1e-20  # for cell-nucleus metrics across all cells
    cell_dens_th_S = 1e-10  # for structure volume metrics

    ####### Remove outliers ########

    # %% Remove cells that lack a Structure Volume value
    print(np.any(cells.isnull()))
    cells_ao = cells[["CellId", "structure_name"]].copy()
    cells_ao["Outlier annotation"] = "Keep"
    print(cells.shape)
    CellIds_remove = (
        cells.loc[cells["Structure volume"].isnull(), "CellId"].squeeze().to_numpy()
    )
    cells_ao.loc[
        cells_ao["CellId"].isin(CellIds_remove), "Outlier annotation"
    ] = "Missing structure volume"
    cells = cells.drop(cells[cells["CellId"].isin(CellIds_remove)].index)
    cells.reset_index(drop=True)
    print(
        f"Removing {len(CellIds_remove)} cells that lack a Structure Volume measurement value"
    )
    print(cells.shape)
    print(np.any(cells.isnull()))

    # %%
    print("FIX LINE BELOW")
    # cells["Piece std"] = cells["Piece std"].replace(np.nan, 0)

    # %% Feature set for cell and nuclear features
    cellnuc_metrics = [
        "Cell surface area",
        "Cell volume",
        "Cell height",
        "Nuclear surface area",
        "Nuclear volume",
        "Nucleus height",
        "Cytoplasmic volume",
    ]
    cellnuc_abbs = [
        "Cell area",
        "Cell vol",
        "Cell height",
        "Nuc area",
        "Nuc vol",
        "Nuc height",
        "Cyto vol",
    ]
    struct_metrics = ["Structure volume"]

    # %% All metrics including height
    L = len(cellnuc_metrics)
    pairs = np.zeros((int(L * (L - 1) / 2), 2)).astype(np.int)
    i = 0
    for f1 in np.arange(L):
        for f2 in np.arange(L):
            if f2 > f1:
                pairs[i, :] = [f1, f2]
                i += 1

    # %% The typical six scatter plots
    xvec = [1, 1, 6, 1, 4, 6]
    yvec = [4, 6, 4, 0, 3, 3]
    pairs2 = np.stack((xvec, yvec)).T

    # %% Just one
    xvec = [1]
    yvec = [4]
    pairs1 = np.stack((xvec, yvec)).T

    # %% Parameters
    nbins = 100
    N = 10000
    fac = 1000
    Rounds = 5

    # %% For all pairs compute densities
    remove_cells = cells["CellId"].to_frame().copy()
    for i, xy_pair in tqdm(enumerate(pairs), "Enumerate pairs of metrics"):

        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        print(f"{metricX} vs {metricY}")

        # data
        x = cells[metricX].to_numpy() / fac
        y = cells[metricY].to_numpy() / fac
        # x = cells[metricX].sample(1000,random_state = 1117).to_numpy() / fac
        # y = cells[metricY].sample(1000,random_state = 1117).to_numpy() / fac

        # density estimate, repeat because of probabilistic nature of density estimate used here
        for r in np.arange(Rounds):
            remove_cells[f"{metricX} vs {metricY}_{r}"] = np.nan
            print(f"Round {r + 1} of {Rounds}")
            rs = int(r)
            xS, yS = resample(
                x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
            )
            k = gaussian_kde(np.vstack([xS, yS]))
            cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
            cell_dens = cell_dens / np.sum(cell_dens)
            remove_cells.loc[
                remove_cells.index[np.arange(len(cell_dens))],
                f"{metricX} vs {metricY}_{r}",
            ] = cell_dens

    remove_cells.to_csv(data_root_extra / "cell_nucleus.csv")
    # remove_cells = pd.read_csv(data_root_extra / 'cell_nucleus.csv')

    # %% Summarize across repeats
    remove_cells_summary = cells["CellId"].to_frame().copy()
    for i, xy_pair in enumerate(pairs):
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        print(f"{metricX} vs {metricY}")
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        filter_col = [
            col for col in remove_cells if col.startswith(f"{metricX} vs {metricY}")
        ]
        x = remove_cells[filter_col].to_numpy()
        pos = np.argwhere(np.any(x < cell_dens_th_CN, axis=1))
        y = x[pos, :].squeeze()

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        xr = np.log(x.flatten())
        xr = np.delete(xr, np.argwhere(np.isinf(xr)))
        axs[0].hist(xr, bins=100)
        axs[0].set_title(f"Histogram of cell probabilities (log scale)")
        axs[0].set_yscale("log")
        im = axs[1].imshow(np.log(y), aspect="auto")
        plt.colorbar(im)
        axs[1].set_title(f"Heatmap with low probability cells (log scale)")

        if save_flag:
            plot_save_path = pic_root / f"{metricX} vs {metricY}_cellswithlowprobs.png"
            plt.savefig(plot_save_path, format="png", dpi=1000)
            plt.close()
        else:
            plt.show()

        remove_cells_summary[f"{metricX} vs {metricY}"] = np.median(x, axis=1)

    # %% Identify cells to be removed
    CellIds_remove_dict = {}
    CellIds_remove = np.empty(0, dtype=int)
    for i, xy_pair in enumerate(pairs):
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        CellIds_remove_dict[f"{metricX} vs {metricY}"] = np.argwhere(
            remove_cells_summary[f"{metricX} vs {metricY}"].to_numpy() < cell_dens_th_CN
        )
        CellIds_remove = np.union1d(
            CellIds_remove, CellIds_remove_dict[f"{metricX} vs {metricY}"]
        )
        print(len(CellIds_remove))

    # %% Plot and remove outliers
    plotname = "CellNucleus"
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        pic_root,
        f"{plotname}_6_org_fine",
        0.5,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        pic_root,
        f"{plotname}_6_org_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        pic_root,
        f"{plotname}_6_outliers",
        2,
        CellIds_remove_dict,
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        pic_root,
        f"{plotname}_21_org_fine",
        0.5,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        pic_root,
        f"{plotname}_21_org_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        pic_root,
        f"{plotname}_21_outliers",
        2,
        CellIds_remove_dict,
    )
    print(cells.shape)
    CellIds_remove = (
        cells.loc[cells.index[CellIds_remove], "CellId"].squeeze().to_numpy()
    )
    cells_ao.loc[
        cells_ao["CellId"].isin(CellIds_remove), "Outlier annotation"
    ] = "Abnormal cell or nuclear metric"
    cells = cells.drop(cells.index[cells["CellId"].isin(CellIds_remove)])
    print(
        f"Removing {len(CellIds_remove)} cells due to abnormal cell or nuclear metric"
    )
    print(cells.shape)
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        pic_root,
        f"{plotname}_6_clean_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        pic_root,
        f"{plotname}_6_clean_fine",
        0.5,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        pic_root,
        f"{plotname}_21_clean_thick",
        2,
        [],
    )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs,
        cells,
        True,
        pic_root,
        f"{plotname}_21_clean_fine",
        0.5,
        [],
    )

    # %% Feature sets for structures
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
    selected_structures_org = [
        "Nuclear envelope",
        "Golgi",
        "Mitochondria",
        "ER",
        "ER",
        "Lysosome",
        "Endosomes",
        "Peroxisomes",
        "Microtubules",
        "Tight junctions",
        "NPC",
        "Nucleolus F",
        "Nucleolus G",
        "SON",
    ]
    selected_structures_cat = [
        "Major organelle",
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
        "Nuclear",
    ]
    structure_metric = "Structure volume"

    # %% Parameters
    nbins = 100
    N = 1000
    fac = 1000
    Rounds = 5

    # %% For all pairs compute densities
    remove_cells = cells["CellId"].to_frame().copy()
    for xm, metric in tqdm(enumerate(selected_metrics), "Iterating metrics"):
        for ys, struct in tqdm(enumerate(selected_structures), "and structures"):

            # data
            x = (
                cells.loc[cells["structure_name"] == struct, [metric]]
                .squeeze()
                .to_numpy()
                / fac
            )
            y = (
                cells.loc[cells["structure_name"] == struct, [structure_metric]]
                .squeeze()
                .to_numpy()
                / fac
            )

            # density estimate, repeat because of probabilistic nature of density estimate used here
            for r in np.arange(Rounds):
                if ys == 0:
                    remove_cells[f"{metric} vs {structure_metric}_{r}"] = np.nan
                # print(f"Round {r+1} of {Rounds}")
                rs = int(r)
                xS, yS = resample(
                    x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
                )
                k = gaussian_kde(np.vstack([xS, yS]))
                cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
                cell_dens = cell_dens / np.sum(cell_dens)
                remove_cells.loc[
                    cells["structure_name"] == struct,
                    f"{metric} vs {structure_metric}_{r}",
                ] = cell_dens

    remove_cells.to_csv(data_root_extra / "structures.csv")
    # remove_cells = pd.read_csv(data_root_extra / 'structures.csv')

    # %% Summarize across repeats
    remove_cells_summary = cells["CellId"].to_frame().copy()
    for xm, metric in enumerate(selected_metrics):
        print(metric)

        filter_col = [
            col
            for col in remove_cells
            if col.startswith(f"{metric} vs {structure_metric}")
        ]
        x = remove_cells[filter_col].to_numpy()
        pos = np.argwhere(np.any(x < cell_dens_th_S, axis=1))
        y = x[pos, :].squeeze()

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        xr = np.log(x.flatten())
        xr = np.delete(xr, np.argwhere(np.isinf(xr)))
        axs[0].hist(xr, bins=100)
        axs[0].set_title(f"Histogram of cell probabilities (log scale)")
        axs[0].set_yscale("log")
        im = axs[1].imshow(np.log(y), aspect="auto")
        plt.colorbar(im)
        axs[1].set_title(f"Heatmap with low probability cells (log scale)")

        if save_flag:
            plot_save_path = (
                pic_root / f"{metric} vs {structure_metric}_cellswithlowprobs.png"
            )
            plt.savefig(plot_save_path, format="png", dpi=1000)
            plt.close()
        else:
            plt.show()

        remove_cells_summary[f"{metric} vs {structure_metric}"] = np.median(x, axis=1)

    # %% Identify cells to be removed
    CellIds_remove_dict = {}
    CellIds_remove = np.empty(0, dtype=int)
    for xm, metric in enumerate(selected_metrics):
        print(metric)
        CellIds_remove_dict[f"{metric} vs {structure_metric}"] = np.argwhere(
            remove_cells_summary[f"{metric} vs {structure_metric}"].to_numpy()
            < cell_dens_th_S
        )
        CellIds_remove = np.union1d(
            CellIds_remove, CellIds_remove_dict[f"{metric} vs {structure_metric}"]
        )
        print(len(CellIds_remove))

    # %% Plot and remove outliers
    plotname = "Structures"
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_1_org_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_2_org_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_1_org_thick",
        2,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_2_org_thick",
        2,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_1_outliers",
        2,
        CellIds_remove_dict,
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_2_outliers",
        2,
        CellIds_remove_dict,
    )
    print(cells.shape)
    CellIds_remove = (
        cells.loc[cells.index[CellIds_remove], "CellId"].squeeze().to_numpy()
    )
    cells_ao.loc[
        cells_ao["CellId"].isin(CellIds_remove), "Outlier annotation"
    ] = "Abnormal structure volume metrics"
    cells = cells.drop(cells.index[cells["CellId"].isin(CellIds_remove)])
    print(f"Removing {len(CellIds_remove)} cells due to structure volume metrics")
    print(cells.shape)
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_1_clean_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_2_clean_fine",
        0.5,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_1_clean_thick",
        2,
        [],
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        pic_root,
        f"{plotname}_2_clean_thick",
        2,
        [],
    )

    # %% Saving
    cells.to_csv(data_root / dataset_clean)
    cells_ao.to_csv(data_root / dataset_outliers)

    # %% Final diagnostic plot
    cells = pd.read_csv(data_root / dataset)
    CellIds_remove_dict = {}

    for i, xy_pair in enumerate(pairs):
        metricX = cellnuc_metrics[xy_pair[0]]
        metricY = cellnuc_metrics[xy_pair[1]]
        CellIds_remove_dict[f"{metricX} vs {metricY}"] = np.argwhere(
            (
                cells_ao["Outlier annotation"] == "Abnormal cell or nuclear metric"
            ).to_numpy()
        )
    oplot(
        cellnuc_metrics,
        cellnuc_abbs,
        pairs2,
        cells,
        True,
        pic_root,
        f"Check_cellnucleus",
        2,
        CellIds_remove_dict,
    )

    CellIds_remove_dict = {}
    for xm, metric in enumerate(selected_metrics):
        CellIds_remove_dict[f"{metric} vs {structure_metric}"] = np.argwhere(
            (
                (cells_ao["Outlier annotation"] == "Abnormal structure volume metrics")
                | (cells_ao["Outlier annotation"] == "Abnormal cell or nuclear metric")
            ).to_numpy()
        )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[0:7],
        structure_metric,
        cells,
        True,
        pic_root,
        f"Check_structures_1",
        2,
        CellIds_remove_dict,
    )
    splot(
        selected_metrics,
        selected_metrics_abb,
        selected_structures[7:14],
        structure_metric,
        cells,
        True,
        pic_root,
        f"Check_structures_2",
        2,
        CellIds_remove_dict,
    )
