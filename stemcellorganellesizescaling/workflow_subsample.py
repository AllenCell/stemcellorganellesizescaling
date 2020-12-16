#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import sys, importlib
import os, platform
from shutil import copyfile
import pandas as pd

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
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %% Define sampling numbers
samplevec = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 1500]
repeats = 3
for s, sample in enumerate(samplevec):
    for r in range(0, repeats):

        datastr = f"{sample}_{r}"
        #%% Directories
        if platform.system() == "Windows":
            data_root = Path(f"E:/DA/Data/scoss/Data/Subsample_Dec2020mesh/{datastr}")
            pic_root = Path(f"E:/DA/Data/scoss/Pics/Subsample_Dec2020mesh/{datastr}")
            org_root = Path("E:/DA/Data/scoss/Data/Dec2020mesh")
        elif platform.system() == "Linux":
            data_root = Path(
                f"/allen/aics/modeling/theok/Projects/Data/scoss/Data/Subsample_Dec2020mesh/{datastr}"
            )
            pic_root = Path(
                f"/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Subsample_Dec2020mesh/{datastr}"
            )
            org_root = Path(
                "/allen/aics/modeling/theok/Projects/Data/scoss/Data/Dec2020mesh"
            )
        print(data_root)
        data_root.mkdir(exist_ok=True)
        pic_root.mkdir(exist_ok=True)
        # Load dataset
        cells = pd.read_csv((org_root / "SizeScaling_20201215.csv"))
        # Sample
        structures = cells["structure_name"].unique()
        index = pd.Series([])
        for s, struct in enumerate(structures):
            index = index.append(
                cells[cells["structure_name"] == struct]
                .sample(n=sample)
                .index.to_series()
            )
        cells = cells.loc[index]
        cells.to_csv(data_root / "SizeScaling_20201215.csv")
        # Directories
        dirs = []
        dirs.append(data_root)
        dirs.append(pic_root)

        #%% Computing statistics - Pairwise statistics
        print(
            "##################### Computing statistics - Pairwise statistics #####################"
        )
        tableIN = "SizeScaling_20201215.csv"
        table_compIN = "SizeScaling_20201215_comp.csv"
        statsOUTdir = "Stats_20201215"
        pairwisestats(
            dirs,
            tableIN,
            table_compIN,
            statsOUTdir,
            COMP_flag=False,
            PCA_flag=False,
            SubSample_flag=True,
        )

        # #%% Computing statistics - Explained variance of composite models
        print(
            "##################### Computing statistics - Composite models #####################"
        )
        tableIN = "SizeScaling_20201215.csv"
        statsOUTdir = "Stats_20201215"
        compositemodels_explainedvariance(dirs, tableIN, statsOUTdir)

        # #%% Computing statistics - Scaling statistics
        print(
            "##################### Computing statistics - Scaling statistics #####################"
        )
        tableIN = "SizeScaling_20201215.csv"
        statsOUTdir = "Stats_20201215"
        scaling_stats(dirs, tableIN, statsOUTdir)
