#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import sys, importlib
import os, platform

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

#%% Directories
if platform.system() == "Windows":
    data_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Oct2021/")
    pic_root = Path("Z:/modeling/theok/Projects/Data/scoss/Pics/Oct2021/")
elif platform.system() == "Linux":
    data_root = Path(
        "/allen/aics/modeling/theok/Projects/Data/scoss/Data/Oct2021/"
    )
    pic_root = Path(
        "/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Oct2021/"
    )
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

#%% Data preparation - Initial Parsing
# print('##################### Data preparation - Initial Parsing #####################')
# #current version 2021-11-19_1:25pm
# tableIN = '/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cvapipe_analysis/local_staging_variance/preprocessing/manifest.csv'
# old_tableIN = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cell_shape_variation/local_staging_beta/shapemode/manifest.csv"
# featIN = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/resources/qcb/data-raw/production/feature"
# #current version 2021-11-19_1:44pm
# ShapeModeIN = '/allen/aics/assay-dev/MicroscopyOtherData/Viana/forTheo/variance/final_dataset/manifest_shape_modes.csv'
# tableSNIP = "Manifest_snippet_20211101.csv"
# tableOUT = "SizeScaling_20211101.csv"
# initial_parsing(dirs, tableIN, old_tableIN, featIN, ShapeModeIN, tableSNIP, tableOUT)

#%% Data preparation - Outlier Removal
# print('##################### Outlier removal is done more upstream and typically not run anymore as part of the size scaling workflow #####################'
# print('##################### Data preparation - Outlier Removal #####################')
# tableIN = "SizeScaling_20201006.csv"
# tableOUT = "SizeScaling_20201006_clean.csv"
# tableOUTL = "SizeScaling_20201006_outliers.csv"
# outlier_removal(dirs, tableIN, tableOUT, tableOUTL)

#%% Data preparation - Diagnostic violins
# print('##################### Data preparation - Diagnostic violins #####################')
# tableIN = "SizeScaling_20211101.csv"
# diagnostic_violins(dirs, tableIN)

#%% Computing statistics - Compensation analysis
# print('##################### Computing statistics - Compensation analysis #####################')
# tableIN = "SizeScaling_20211101.csv"
# tableOUT = "SizeScaling_20211101_comp.csv"
# compensate(dirs, tableIN, tableOUT)

#%% Computing statistics - Pairwise statistics
print('##################### Computing statistics - Pairwise statistics #####################')
tableIN = "SizeScaling_20211101.csv"
table_compIN = "SizeScaling_20211101_comp.csv"
statsOUTdir = "Stats_20211101"
pairwisestats(dirs, tableIN, table_compIN, statsOUTdir, COMP_flag=False)

#%% Computing statistics - Explained variance of composite models
# print('##################### Computing statistics - Composite models #####################')
# tableIN = "SizeScaling_20211101.csv"
# statsOUTdir = "Stats_20211101"
# compositemodels_explainedvariance(dirs, tableIN, statsOUTdir)

#%% Computing statistics - Scaling statistics
# print(
#     "##################### Computing statistics - Scaling statistics #####################"
# )
# tableIN = "SizeScaling_20211101.csv"
# scaleOUTdir = "Scale_20211101"
# scaling_stats(dirs, tableIN, scaleOUTdir)

#%% Plotting scatterplots - Cell and nuclear metrics
# print('##################### Plotting scatterplots - Cell and nuclear metrics #####################')
# tableIN = "SizeScaling_20211101.csv"
# statsIN = "Stats_20211101"
# cellnuc_scatter_plots(dirs, tableIN, statsIN)

#%% Plotting scatterplots - Organelle scatter plots
# print('##################### Plotting scatterplots - Organelle scatter plots #####################')
# tableIN = "SizeScaling_20211101.csv"
# statsIN = "Stats_20211101"
# organelle_scatter_plots(dirs, tableIN, statsIN)

#%% Plotting scatterplots - Organelle scatter plots
# print('##################### Plotting scatterplots - Compensated organelle scatter plots #####################')
# tableIN = "SizeScaling_20211101.csv"
# table_compIN = "SizeScaling_20211101_comp.csv"
# statsIN = "Stats_20211101"
# organelle_compensated_scatter_plots(dirs, tableIN, table_compIN, statsIN)

#%% Downsampling analysis
# print(
#     "##################### Plotting scatterplots - Compensated organelle scatter plots #####################"
# )
# tableIN = "SizeScaling_20211101.csv"
# sampleOUTdir = "Sample_20211101"
# sampling_stats(dirs, tableIN, sampleOUTdir)
