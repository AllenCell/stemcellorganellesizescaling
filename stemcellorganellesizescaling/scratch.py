#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import sys, importlib
import os, platform
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Third party

# Relative
from stemcellorganellesizescaling.analyses.data_prep import (
    outlier_removal,
    initial_parsing,
    diagnostic_violins,
)

importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.data_prep"])
from stemcellorganellesizescaling.analyses.data_prep import (
    outlier_removal,
    initial_parsing,
    diagnostic_violins,
)

from stemcellorganellesizescaling.analyses.compute_stats import (
    compensate,
    pairwisestats,
    compositemodels_explainedvariance,
    scaling_stats,
)

importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.compute_stats"])
from stemcellorganellesizescaling.analyses.compute_stats import (
    compensate,
    pairwisestats,
    compositemodels_explainedvariance,
    scaling_stats,
)

from stemcellorganellesizescaling.analyses.scatter_plots import (
    cellnuc_scatter_plots,
    organelle_scatter_plots,
    organelle_compensated_scatter_plots,
)

importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.scatter_plots"])
from stemcellorganellesizescaling.analyses.scatter_plots import (
    cellnuc_scatter_plots,
    organelle_scatter_plots,
    organelle_compensated_scatter_plots,
)

print("Libraries loaded succesfully")

#%% Directories
if platform.system() == "Windows":
    data_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/tmp/checkoldvsnew_20211119/")
elif platform.system() == "Linux":
    data_root = Path(
        "/allen/aics/modeling/theok/Projects/Data/scoss/Data/Oct2021/"
    )

# #%% Define sampling numbers
# tableNEW = '/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/preprocessing/manifest.csv'
# tableOLD = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cell_shape_variation/local_staging_beta/shapemode/manifest.csv"
# tableSNIPOLD = "Manifest_snippet_OLD.csv"
# tableSNIPNEW = "Manifest_snippet_NEW.csv"
# cellsOLD = pd.read_csv(tableOLD)
# print(len(cellsOLD))
# cellsOLD.sample(n=10).to_csv(data_root / tableSNIPOLD)
# cellsNEW = pd.read_csv(tableNEW)
# print(len(cellsNEW))
# cellsNEW.sample(n=10).to_csv(data_root / tableSNIPNEW)
# print(len(set(cellsOLD.CellId).intersection(set(cellsNEW.CellId))))
# # import pdb
# # pdb.set_trace()

# %% Looking at the new shape mode table
# PCA_in = '/allen/aics/assay-dev/MicroscopyOtherData/Viana/forTheo/variance/final_dataset/manifest_shape_modes.csv'
# PCA_snip = 'manifest_shape_modes_sample.csv'
# PCA = pd.read_csv(PCA_in)
# print(len(PCA))
# PCA.sample(n=10).to_csv(data_root / PCA_snip)

# cells_path = Path("/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cell_shape_variation/local_staging_beta/shapemode/manifest.csv")
# cells = pd.read_csv(cells_path)
# cells.sample(n=10).to_csv(Path("/allen/aics/modeling/VariancePlayground/manifests/VAR_snip.csv"))

# %%
OLD_in = data_root / 'SizeScaling_20201102.csv'
OLD = pd.read_csv(OLD_in)
OLD = OLD.rename(
    columns={
        "DNA_MEM_PC1": "NUC_MEM_PC1",
        "DNA_MEM_PC2": "NUC_MEM_PC2",
        "DNA_MEM_PC3": "NUC_MEM_PC3",
        "DNA_MEM_PC4": "NUC_MEM_PC4",
        "DNA_MEM_PC5": "NUC_MEM_PC5",
        "DNA_MEM_PC6": "NUC_MEM_PC6",
        "DNA_MEM_PC7": "NUC_MEM_PC7",
        "DNA_MEM_PC8": "NUC_MEM_PC8",

    }
)
NEW_in = data_root / 'SizeScaling_20211101.csv'
NEW = pd.read_csv(NEW_in)

# %%
CELLS = NEW.merge(OLD,how='inner',on='CellId',suffixes=('_new','_old'))

# %%
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

# %%
save_flag=True

for i, metric in enumerate(selected_metrics):

    fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
    axes.scatter(CELLS[f'{metric}_old'],CELLS[f'{metric}_new'],1)


    axes.set_title(f"{metric}")
    axes.grid(True, which="major", axis="both")
    axes.set_axisbelow(True)
    axes.set_ylabel('NEW')
    axes.set_xlabel('OLD')


    if save_flag:
        plot_save_path = data_root / f"{metric}.png"
        plt.savefig(plot_save_path, format="png", dpi=300)
        plt.close()
    else:
        plt.show()

