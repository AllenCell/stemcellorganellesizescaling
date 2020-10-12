#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import os, platform

# Third party
# from stemcellorganellesizescaling.analyses.data_prep import outlier_removal, initial_parsing
# from stemcellorganellesizescaling.analyses.compute_stats import compensate
# importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.compute_stats"])
# from stemcellorganellesizescaling.analyses.compute_stats import compensate, pairwisestats
# Relative

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

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

dataset = "SizeScaling_20201012.csv"

# Load dataset
cells = pd.read_csv(data_root / dataset)
np.any(cells.isnull())

# Remove outliers
# %% Parameters, updated directories
save_flag = 1  # save plot (1) or show on screen (0)
pic_root = pic_root / "diagnostic violins"
pic_root.mkdir(exist_ok=True)

# %%
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

#%% Bars with numbers of cells for each of the structures
table = pd.pivot_table(cells, index="structure_name", aggfunc="size")
table = table.reindex(sortedStructures)
fig, axes = plt.subplots(figsize=(10, 5), dpi=100)

table.plot.barh(ax=axes)
# x_pos = range(len(table))
# plt.barh(x_pos, table)
# plt.yticks(x_pos, table.keys())

for j, val in enumerate(table):
    axes.text(
        val, j, str(val), ha="right", va="center", color="white", size=6, weight="bold"
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

#%% Stacked bars comparing stuctures and imaging mode
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
# 'DNA_MEM_UMAP1', 'DNA_MEM_UMAP2', 'Piece average', 'Piece max', 'Piece min',
#        'Piece std', 'Piece sum'

selected_metrics = ['Cell surface area', 'Cell volume', 'Nuclear surface area',
       'Nuclear volume', 'Cytoplasmic volume', 'Number of pieces', 'Structure volume', 'Cell height',
       'Cell xbox', 'Cell ybox', 'Nucleus height', 'Nucleus xbox',
       'Nucleus ybox','DNA_MEM_PC1', 'DNA_MEM_PC2',
       'DNA_MEM_PC3', 'DNA_MEM_PC4', 'DNA_MEM_PC5', 'DNA_MEM_PC6',
       'DNA_MEM_PC7', 'DNA_MEM_PC8',
       ]

for i, metric in enumerate(selected_metrics):

    fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
    sns.violinplot(
        y="structure_name", x=metric, color='black', data=cells, scale='width', ax=axes, order=sortedStructures
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
