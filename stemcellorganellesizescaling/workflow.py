#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import sys, importlib
import os, platform

# Third party

# Relative
from stemcellorganellesizescaling.analyses.data_prep import outlier_removal, initial_parsing, diagnostic_violins
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.data_prep"])
from stemcellorganellesizescaling.analyses.data_prep import outlier_removal, initial_parsing, diagnostic_violins

from stemcellorganellesizescaling.analyses.compute_stats import compensate
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.compute_stats"])
from stemcellorganellesizescaling.analyses.compute_stats import compensate, pairwisestats

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/")
    pic_root = Path("E:/DA/Data/scoss/Pics/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

#%% Data preparation - Initial Parsing
# print('##################### Data preparation - Initial Parsing #####################')
# tableIN = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cell_shape_variation/local_staging_PRODUCTION/expand/manifest.csv"
# tableSNIP = "Manifest_snippet_202010112.csv"
# tableOUT = "SizeScaling_20201012.csv"
# initial_parsing(dirs, tableIN, tableSNIP, tableOUT)

# #%% Data preparation - Outlier Removal
# print('##################### Data preparation - Outlier Removal #####################')
tableIN = "SizeScaling_20201006.csv"
tableOUT = "SizeScaling_20201006_clean.csv"
tableOUTL = "SizeScaling_20201006_outliers.csv"
outlier_removal(dirs, tableIN, tableOUT, tableOUTL)

#%% Data preparation - Diagnostic violins
print('##################### Data preparation - Diagnostic violins #####################')
tableIN = "SizeScaling_20201012.csv"
diagnostic_violins(dirs, tableIN)

#%% Computing statistics
# tableIN = "SizeScaling_20201006_clean.csv"
# tableOUT = "SizeScaling_20201006_comp.csv"
# compensate(dirs, tableIN, tableOUT)

#%% Computing statistics
# tableIN = "SizeScaling_20201006_clean.csv"
# table_compIN = "SizeScaling_20201006_comp.csv"
# statsOUTdir = "Stats_20201006"
# pairwisestats(dirs, tableIN, table_compIN, statsOUTdir)
