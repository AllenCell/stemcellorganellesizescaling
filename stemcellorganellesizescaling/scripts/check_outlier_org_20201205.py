#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cbook, colors as mcolors
import numpy as np
import matplotlib
import statsmodels.api as sm
from sklearn.utils import resample
import datetime
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from matplotlib import cm
import pickle
import seaborn as sns
import os, platform
import sys, importlib

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter

importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.scatter_plotting_func"]
)
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################
#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/")
    pic_root = Path("E:/DA/Data/scoss/Pics/")
elif platform.system() == "Linux":
    1/0
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

save_flag = 0
plt.rcParams.update({"font.size": 10})
plt.rcParams['svg.fonttype'] = 'none'

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201006.csv"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
print(np.any(cells.isnull()))

# %%
CellIds_remove = cells.loc[cells["Structure volume"].isnull(), 'CellId'].squeeze().to_numpy()

# %% Bars with numbers of cells for each of the structures
fs = 10
plt.rcParams.update({"font.size": fs})
OutlierType = cells['Outlier'].unique()
fig, axes = plt.subplots(2,2, figsize=(16, 9), dpi=100)


for i, ot in enumerate(OutlierType):
    if i==0:
        x=0
        y=0
    elif i==1:
        x=0
        y=1
    elif i==2:
        x =1
        y=0
    elif i==3:
        x=1
        y=1
    table = pd.pivot_table(cells.loc[cells['Outlier']==ot], index="structure_name", aggfunc="size")
    table.plot.barh(ax=axes[x,y])

    for j, val in enumerate(table):
        axes[x,y].text(
            val, j, str(val), ha="left", va="center", color="black", size=fs, weight="bold"
        )

    axes[x,y].set_title(f"Outlier type: {ot} (n= {sum(table)})")
    axes[x,y].set_ylabel(None)
    # axes[x,y].grid(True, which="major", axis="x")
    axes[x,y].set_axisbelow(True)
    axes[x,y].invert_yaxis()

plt.show()
