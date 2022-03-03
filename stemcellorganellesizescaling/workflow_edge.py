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


# Edge and non-edge cell manifests
# edge: /allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_edge_cells_midpoint_expanded/preprocessing/manifest.csv
# control: /allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance_edges/preprocessing/manifest.csv

if platform.system() == "Windows":
    edge = 'Z:/modeling/theok/Projects/Data/scoss/Data/edgemanifests/manifest_edge.csv'
    nonedge = 'Z:/modeling/theok/Projects/Data/scoss/Data/edgemanifests/manifest_nonedge.csv'
elif platform.system() == "Linux":
    edge = '/allen/aics/modeling/theok/Projects/Data/scoss/Data/edgemanifests/manifest_edge.csv'
    nonedge = '/allen/aics/modeling/theok/Projects/Data/scoss/Data/edgemanifests/manifest_nonedge.csv'

#%% Define sampling numbers
samplevec = ['edge','non-edge']
for s, sample in enumerate(samplevec):

    datastr = f"{sample}"
    #%% Directories
    if platform.system() == "Windows":
        data_root = Path(f"Z:/modeling/theok/Projects/Data/scoss/Data/Subsample_Oct2021/{datastr}")
        pic_root = Path(f"Z:/modeling/theok/Projects/Data/scoss/Data/Subsample_Oct2021/{datastr}")
        org_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Dec2020")
    elif platform.system() == "Linux":
        data_root = Path(
            f"/allen/aics/modeling/theok/Projects/Data/scoss/Data/Subsample_Oct2021/{datastr}"
        )
        pic_root = Path(
            f"/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Subsample_Oct2021/{datastr}"
        )
        org_root = Path(
            "/allen/aics/modeling/theok/Projects/Data/scoss/Data/Oct2021"
        )
    print(data_root)
    data_root.mkdir(exist_ok=True)
    pic_root.mkdir(exist_ok=True)
    # Load dataset
    cells = pd.read_csv((org_root / "SizeScaling_20211101.csv"))
    # Load (non)edge manifest
    if sample == 'edge':
        em = pd.read_csv(edge)
    elif sample == 'non-edge':
        em = pd.read_csv(nonedge)

    cells = cells.merge(em, how='inner', on='CellId', suffixes=[None, '_c'])
    print(len(cells))
    cells.to_csv(data_root / "SizeScaling_20211101.csv")
    # Directories
    dirs = []
    dirs.append(data_root)
    dirs.append(pic_root)


    #%% Data preparation - Diagnostic violins
    print('##################### Data preparation - Diagnostic violins #####################')
    tableIN = "SizeScaling_20211101.csv"
    diagnostic_violins(dirs, tableIN)

    #%% Computing statistics - Pairwise statistics
    print(
        "##################### Computing statistics - Pairwise statistics #####################"
    )
    tableIN = "SizeScaling_20211101.csv"
    table_compIN = "SizeScaling_20211101_comp.csv"
    statsOUTdir = "Stats_20211101"
    pairwisestats(
        dirs,
        tableIN,
        table_compIN,
        statsOUTdir,
        COMP_flag=False,
        PCA_flag=False,
        SubSample_flag=False,
    )

    # #%% Computing statistics - Explained variance of composite models
    print(
        "##################### Computing statistics - Composite models #####################"
    )
    tableIN = "SizeScaling_20211101.csv"
    statsOUTdir = "Stats_20211101"
    compositemodels_explainedvariance(dirs, tableIN, statsOUTdir)

    # #%% Computing statistics - Scaling statistics
    print(
        "##################### Computing statistics - Scaling statistics #####################"
    )
    tableIN = "SizeScaling_20211101.csv"
    statsOUTdir = "Stats_20211101"
    scaling_stats(dirs, tableIN, statsOUTdir)
