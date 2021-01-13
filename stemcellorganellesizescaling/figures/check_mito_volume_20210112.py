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
    data_root = Path("E:/DA/Data/scoss/Data/Dec2020/")
    ann_root = Path("E:/DA/Data/scoss/Data/Nov2020/")
    pic_root = Path("E:/DA/Data/scoss/Pics/Dec2020/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/Dec2020/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Dec2020/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

save_flag = 0
plt.rcParams.update({"font.size": 8})
plt.rcParams["svg.fonttype"] = "none"

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201215.csv"
table_compIN = "SizeScaling_20201215_comp.csv"
statsIN = "Stats_20201215"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
print(np.any(cells.isnull()))
# cells_COMP = pd.read_csv(data_root / table_compIN)
# print(np.any(cells_COMP.isnull()))
structures = pd.read_csv(ann_root / "annotation" / "structure_annotated_20201113.csv")
ScaleMat = pd.read_csv(data_root / 'Scale_20201215' / "ScaleStats_20201125.csv")
ScaleCurve = pd.read_csv(data_root / 'Scale_20201215' / "ScaleCurve_20201125.csv")
# ScaleMat = pd.read_csv(data_root / "growing" / "ScaleStats_20201118.csv")
# ScaleCurve = pd.read_csv(data_root / "growing" / "ScaleCurve_20201124.csv")


# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result


# %% Feature sets
HM = {}
HM["cellnuc_heatmap"] = [
    "Cell volume",
    "Cell surface area",
    "Nuclear volume",
    "Nuclear surface area",
    "Cytoplasmic volume",
    ]

# %%
facX = 1 / ((0.108333) ** 3)
stats_root = data_root / statsIN / "cellnuc_struct_metrics"
metricX = HM["cellnuc_heatmap"][0]
metricY = "TOMM20"
structure_metric = 'Structure volume'
xii = loadps(stats_root, f"{metricX}_{structure_metric}_{metricY}_xii") / facX
pred_yL = (
    loadps(stats_root, f"{metricX}_{structure_metric}_{metricY}_pred_matL")
    / facX
)
y0 = np.interp(1160,xii[:,0],pred_yL)
y1 = np.interp(2320,xii[:,0],pred_yL)
y0 = np.round(y0)
y1 = np.round(y1)

print(y0)
print(y1)
print((y1-y0)/y0)

