#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
import sys, importlib
from tqdm import tqdm

# Third party
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    organelle_scatter,
    fscatter,
    compensated_scatter,
    organelle_scatterT,
    compensated_scatter_t
)

importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.scatter_plotting_func"]
)
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    organelle_scatter,
    fscatter,
    compensated_scatter,
    organelle_scatterT,
    compensated_scatter_t
)

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020/")
    pic_root = Path("E:/DA/Data/scoss/Pics/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201102.csv"
tableIN_alt = "SizeScaling_20201102_alt.csv"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())
cells_alt = pd.read_csv(data_root / tableIN_alt)
np.any(cells_alt.isnull())



# %% Add feature pieces
pieces_root = 'E:/DA/Data/scoss/Data/PieceStats_20201012'
paths = Path(pieces_root).glob('**/*.csv')
cells['Piece average'] = np.nan
cells['Piece max'] = np.nan
cells['Piece min'] = np.nan
cells['Piece std'] = np.nan
cells['Piece sum'] = np.nan
cells.set_index('CellId',drop=False,inplace=True)
for csvf in paths:
    print(csvf)
    pieces = pd.read_csv(csvf)
    keepcolumns = [
        "CellId",
        "str_shape_volume_pcc_avg",
        "str_shape_volume_pcc_max",
        "str_shape_volume_pcc_min",
        "str_shape_volume_pcc_std",
        "str_shape_volume_pcc_sum",
        ]
    pieces = pieces[keepcolumns]
    pieces = pieces.rename(
            columns={
                "str_shape_volume_pcc_avg": "Piece average",
                "str_shape_volume_pcc_max": "Piece max",
                "str_shape_volume_pcc_min": "Piece min",
                "str_shape_volume_pcc_std": "Piece std",
                "str_shape_volume_pcc_sum": "Piece sum",
            }
        )
    pieces.set_index('CellId',drop=False,inplace=True)
    cells.update(pieces)

# %% Post-processing and checking
sv = cells['Structure volume'].to_numpy()
ps = cells['Piece sum'].to_numpy()
sn = cells['structure_name'].to_numpy()
pos = np.argwhere(np.divide(abs(ps-sv),sv)>0)
print(f"{len(pos)} mismatches in {np.unique(sn[pos])}")
posS = np.argwhere(np.divide(abs(ps-sv),sv)>0.01)
print(f"{len(posS)} larger than 1%")
posT = np.argwhere(np.divide(abs(ps-sv),sv)>0.1)
print(f"{len(posT)} larger than 10%")
cells.drop(labels='Unnamed: 0',axis=1,inplace=True)
cells.reset_index(drop=True,inplace=True)
print(np.any(cells.isnull()))
cells.loc[cells['Piece std'].isnull(),'Piece std'] = 0
print(np.any(cells.isnull()))


# %%







# %%
1+1
# Missing:
# 'DNA_MEM_UMAP1', 'DNA_MEM_UMAP2'

    # %% Rename columns
    cells = cells.rename(
        columns={
            "mem_roundness_surface_area_lcc": "Cell surface area",
            "mem_shape_volume_lcc": "Cell volume",
            "dna_roundness_surface_area_lcc": "Nuclear surface area",
            "dna_shape_volume_lcc": "Nuclear volume",
            "str_connectivity_number_cc": "Number of pieces",
            "str_shape_volume": "Structure volume",
            "str_shape_volume_lcc": "Structure volume alt",
            "mem_position_depth_lcc": "Cell height",
            "mem_position_height_lcc": "Cell xbox",
            "mem_position_width_lcc": "Cell ybox",
            "dna_position_depth_lcc": "Nucleus height",
            "dna_position_height_lcc": "Nucleus xbox",
            "dna_position_width_lcc": "Nucleus ybox",
            "meta_fov_image_date": "ImageDate",
        }
    )






    x = pieces['CellId']
    for i, ci in tqdm(enumerate(x),f"{csvf.stem}"):
        pos = np.argwhere(y.isin([ci]).to_numpy())
        if len(pos) > 1:
            1/0
        elif len(pos)==1:
            if cells.loc[pos[0], ['Piece average']].isnull() is False:
                print('Overwriting')
            cells.loc[pos[0], ['Piece average']] = pieces.loc[i, ['str_shape_volume_pcc_avg']].to_numpy()
            cells.loc[pos[0], ['Piece max']] = pieces.loc[i, ['str_shape_volume_pcc_max']].to_numpy()
            cells.loc[pos[0], ['Piece min']] = pieces.loc[i, ['str_shape_volume_pcc_min']].to_numpy()
            cells.loc[pos[0], ['Piece std']] = pieces.loc[i, ['str_shape_volume_pcc_std']].to_numpy()
            cells.loc[pos[0], ['Piece sum']] = pieces.loc[i, ['str_shape_volume_pcc_sum']].to_numpy()
            if (csvf.stem!='LMNB1') and (csvf.stem!='H2B'):
                val1 = cells.loc[pos[0], ['Piece sum']].to_numpy()
                val2 = cells.loc[pos[0], ['Structure volume']].to_numpy()
                if ((np.abs(val1-val2))/val2)>0.01:
                    1/0

# %%
