# %% Load libraries
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from shutil import copyfile

print("Libraries loaded successfully")

# %% Arrange directories
data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/Jan2021/")
data_rootC = data_root / "celltiffs"
data_rootC.mkdir(exist_ok=True)

#%% Data preparation - Initial Parsing
print('##################### Data preparation - Initial Parsing #####################')
tableIN = "/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cell_shape_variation/local_staging/shapemode/manifest.csv"
dataset = data_root / tableIN
# Load dataset
file = Path(data_root / 'ExampleDataSet_20210111.csv')
if file.exists():
    cells = pd.read_csv(file)
else:
    cells = pd.read_csv(dataset)
    #  Check out columns, keep a couple
    keepcolumns = [
        "CellId",
        "structure_name",
        "mem_roundness_surface_area_lcc",
        "mem_shape_volume_lcc",
        "dna_roundness_surface_area_lcc",
        "dna_shape_volume_lcc",
        "str_connectivity_number_cc",
        "str_shape_volume",
        "mem_position_depth_lcc",
        "mem_position_height_lcc",
        "mem_position_width_lcc",
        "dna_position_depth_lcc",
        "dna_position_height_lcc",
        "dna_position_width_lcc",
        "WorkflowId",
        "meta_fov_image_date",
        "meta_imaging_mode",
        "crop_raw",
        "crop_seg",
        "name_dict",
        "scale_micron",
        "edge_flag",
        "FOVId",
        "fov_path",
        "fov_seg_path",
        "struct_seg_path",
    ]
    cells = cells[keepcolumns]
    #  Rename columns
    cells = cells.rename(
        columns={
            "mem_roundness_surface_area_lcc": "Cell surface area",
            "mem_shape_volume_lcc": "Cell volume",
            "dna_roundness_surface_area_lcc": "Nuclear surface area",
            "dna_shape_volume_lcc": "Nuclear volume",
            "str_connectivity_number_cc": "Number of pieces",
            "str_shape_volume": "Structure volume",
            "mem_position_depth_lcc": "Cell height",
            "mem_position_height_lcc": "Cell xbox",
            "mem_position_width_lcc": "Cell ybox",
            "dna_position_depth_lcc": "Nucleus height",
            "dna_position_height_lcc": "Nucleus xbox",
            "dna_position_width_lcc": "Nucleus ybox",
            "meta_fov_image_date": "ImageDate",
        }
    )
    # Add a column
    cells["Cytoplasmic volume"] = cells["Cell volume"] - cells["Nuclear volume"]

    #%% Sample a few cells per structure
    n_sample = 3
    structures = cells['structure_name'].unique()
    index = pd.Series([])
    for s, struct in enumerate(structures):
        index = index.append(cells[cells['structure_name'] == struct].sample(n=int(n_sample)).index.to_series())
    cells = cells.loc[index]
    cells.to_csv(data_root / 'ExampleDataSet_20210111.csv')

# %% Copy the tiff files
for index, row in tqdm(cells.iterrows(),'Copying tiff files'):
    src1 = row['crop_raw']
    src2 = row['crop_seg']
    dst1 = data_rootC / f"{row['structure_name']} {row['CellId']} raw.tif"
    dst2 = data_rootC / f"{row['structure_name']} {row['CellId']} seg.tif"
    copyfile(src1, dst1)
    copyfile(src2, dst2)

