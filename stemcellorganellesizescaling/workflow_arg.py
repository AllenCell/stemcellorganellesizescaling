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
from stemcellorganellesizescaling.analyses.data_prep import outlier_removal, initial_parsing, diagnostic_violins
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.data_prep"])
from stemcellorganellesizescaling.analyses.data_prep import outlier_removal, initial_parsing, diagnostic_violins

from stemcellorganellesizescaling.analyses.compute_stats import compensate, pairwisestats, compositemodels_explainedvariance, scaling_stats
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.compute_stats"])
from stemcellorganellesizescaling.analyses.compute_stats import compensate, pairwisestats, compositemodels_explainedvariance, scaling_stats

from stemcellorganellesizescaling.analyses.scatter_plots import cellnuc_scatter_plots, organelle_scatter_plots, organelle_compensated_scatter_plots
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.scatter_plots"])
from stemcellorganellesizescaling.analyses.scatter_plots import cellnuc_scatter_plots, organelle_scatter_plots, organelle_compensated_scatter_plots

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

def workflow_arg(n_sample,n_try):

    datastr = f"{n_sample} {n_try}"
    #%% Directories
    if platform.system() == "Windows":
        data_root = Path(f"E:/DA/Data/scoss/Data/Subsample_Nov2020/{datastr}")
        pic_root =  Path(f"E:/DA/Data/scoss/Pics/Subsample_Nov2020/{datastr}")
        org_root =  Path('E:/DA/Data/scoss/Data/Nov2020')
    elif platform.system() == "Linux":
        data_root = Path(f"/allen/aics/modeling/theok/Projects/Data/scoss/Data/Subsample_Nov2020/{datastr}")
        pic_root =  Path(f"/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Subsample_Nov2020/{datastr}")
        org_root =  Path('/allen/aics/modeling/theok/Projects/Data/scoss/Data/Nov2020')
    print(data_root)
    data_root.mkdir(exist_ok=True)
    pic_root.mkdir(exist_ok=True)
    # Load dataset
    cells = pd.read_csv((org_root / 'SizeScaling_20201102.csv'))
    # Sample
    structures = cells['structure_name'].unique()
    index = pd.Series([])
    for s, struct in enumerate(structures):
        index = index.append(cells[cells['structure_name'] == struct].sample(n=int(n_sample)).index.to_series())
    cells = cells.loc[index]
    cells.to_csv(data_root / 'SizeScaling_20201102.csv')
    # Directories
    dirs = []
    dirs.append(data_root)
    dirs.append(pic_root)

    #%% Data preparation - Initial Parsing
    # print('##################### Data preparation - Initial Parsing #####################')
    # tableIN = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cell_shape_variation/local_staging/shapemode/manifest.csv"
    # featIN = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/resources/qcb/data-raw/production/feature"
    # tableSNIP = "Manifest_snippet_20201102.csv"
    # tableOUT = "SizeScaling_20201102.csv"
    # initial_parsing(dirs, tableIN, featIN, tableSNIP, tableOUT)

    #%% Data preparation - Outlier Removal
    # print('##################### Outlier removal is done more upstream and typically not run anymore as part of the size scaling workflow #####################'
    # print('##################### Data preparation - Outlier Removal #####################')
    # tableIN = "SizeScaling_20201006.csv"
    # tableOUT = "SizeScaling_20201006_clean.csv"
    # tableOUTL = "SizeScaling_20201006_outliers.csv"
    # outlier_removal(dirs, tableIN, tableOUT, tableOUTL)

    # #%% Data preparation - Diagnostic violins
    # # print('##################### Data preparation - Diagnostic violins #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # diagnostic_violins(dirs, tableIN)

    # #%% Computing statistics - Compensation analysis
    # print('##################### Computing statistics - Compensation analysis #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # tableOUT = "SizeScaling_20201102_comp.csv"
    # compensate(dirs, tableIN, tableOUT)
    #
    #%% Computing statistics - Pairwise statistics
    # print('##################### Computing statistics - Pairwise statistics #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # table_compIN = "SizeScaling_20201102_comp.csv"
    # statsOUTdir = "Stats_20201102"
    # pairwisestats(dirs, tableIN, table_compIN, statsOUTdir, COMP_flag=False, PCA_flag=False)
    #
    # #%% Computing statistics - Explained variance of composite models
    # print('##################### Computing statistics - Composite models #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # statsOUTdir = "Stats_20201102"
    # compositemodels_explainedvariance(dirs, tableIN, statsOUTdir)
    #
    # #%% Computing statistics - Scaling statistics
    # print('##################### Computing statistics - Scaling statistics #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # statsOUTdir = "Stats_20201102"
    # scaling_stats(dirs, tableIN, statsOUTdir)
    #
    # #%% Plotting scatterplots - Cell and nuclear metrics
    # print('##################### Plotting scatterplots - Cell and nuclear metrics #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # statsIN = "Stats_20201102"
    # cellnuc_scatter_plots(dirs, tableIN, statsIN)
    #
    # #%% Plotting scatterplots - Organelle scatter plots
    # print('##################### Plotting scatterplots - Organelle scatter plots #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # statsIN = "Stats_20201102"
    # organelle_scatter_plots(dirs, tableIN, statsIN)
    #
    # #%% Plotting scatterplots - Organelle scatter plots
    # print('##################### Plotting scatterplots - Compensated organelle scatter plots #####################')
    # tableIN = "SizeScaling_20201102.csv"
    # table_compIN = "SizeScaling_20201102_comp.csv"
    # statsIN = "Stats_20201102"
    # organelle_compensated_scatter_plots(dirs, tableIN, table_compIN, statsIN)

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    workflow_arg(*sys.argv[1:])
