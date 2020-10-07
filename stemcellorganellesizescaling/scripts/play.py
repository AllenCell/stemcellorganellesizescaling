#%%

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import datetime
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib import cm

# Third party

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

#%% Directories
data_root = Path("E:/DA/Data/scoss/Data/")
pic_root = Path("E:/DA/Data/scoss/Pics/")
#%% Data preparation
dataset = 'Manifest_snippet_202010106.csv'
dataset_clean = 'SizeScaling_20200828_clean.csv'

# %% Load
cells = pd.read_csv(data_root / dataset)

#%% Check out columns, remove a couple
keepcolumns = ['CellId', 'structure_name', 'mem_roundness_surface_area_lcc', 'mem_shape_volume_lcc', 'dna_roundness_surface_area_lcc',
               'dna_shape_volume_lcc', 'str_connectivity_number_cc', 'str_shape_volume',
               'mem_position_depth_lcc', 'mem_position_height_lcc', 'mem_position_width_lcc',
               'dna_position_depth_lcc', 'dna_position_height_lcc', 'dna_position_width_lcc',
               'DNA_MEM_PC1', 'DNA_MEM_PC2', 'DNA_MEM_PC3', 'DNA_MEM_PC4',
               'DNA_MEM_PC5', 'DNA_MEM_PC6', 'DNA_MEM_PC7', 'DNA_MEM_PC8']
cells = cells[keepcolumns]

# Missing:
#  'WorkflowId', 'meta_fov_image_date',
# 'DNA_MEM_UMAP1', 'DNA_MEM_UMAP2'

#%% Rename columns
cells = cells.rename(columns={
    'mem_roundness_surface_area_lcc': 'Cell surface area',
    'mem_shape_volume_lcc': 'Cell volume',
    'dna_roundness_surface_area_lcc': 'Nuclear surface area',
    'dna_shape_volume_lcc': 'Nuclear volume',
    'str_connectivity_number_cc': 'Number of pieces',
    'str_shape_volume': 'Structure volume',
    'str_shape_volume_lcc': 'Structure volume alt',
    'mem_position_depth_lcc': 'Cell height',
    'mem_position_height_lcc': 'Cell xbox',
    'mem_position_width_lcc': 'Cell ybox',
    'dna_position_depth_lcc': 'Nucleus height',
    'dna_position_height_lcc': 'Nucleus xbox',
    'dna_position_width_lcc': 'Nucleus ybox'
})

# Missing:
# 'meta_fov_image_date': 'ImageDate'

#%% Add a column
cells['Cytoplasmic volume'] =  cells['Cell volume']-cells['Nuclear volume']

