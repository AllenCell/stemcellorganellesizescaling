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


print("Libraries loaded succesfully")
if platform.system() == "Linux":
    1 / 0
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result


# %% Feature sets
FS = {}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Nuclear surface area",
    "Nuclear volume",
    "Cytoplasmic volume",
]

FS["struct_metrics"] = ["Structure volume"]
FS["COMP_types"] = ["AVH", "AV", "H"]

# %% Select columns for heatmap
HM = {}
HM["cellnuc_heatmap"] = [
    "Cell volume",
    "Cell surface area",
    "Nuclear volume",
    "Nuclear surface area",
    "Cytoplasmic volume",
]

HM["cellnuc_heatmap_RES_metrics"] = [
    "cell_AV",
    "cell_V",
    "cell_A",
    "nuc_AV",
    "nuc_V",
    "nuc_A",
]
HM["cellnuc_heatmap_RES_abbs"] = [
    "Cell v+a",
    "Cell vol",
    "Cell area",
    "Nuc v+a",
    "Nuc vol",
    "Nuc area",
]

HM["struct_heatmap_metrics"] = "Structure volume"
HM["COMP_type"] = "AV"
HM["LIN_type"] = "Linear"

# %% Annotation
ann_root = Path("E:/DA/Data/scoss/Data/Nov2020/annotation")
structures = pd.read_csv(ann_root / "structure_annotated_20201113.csv")

# %%
PlotMat = pd.DataFrame()
data_root = Path(f"E:/DA/Data/scoss/Data/Dec2020/")

# %% Load dataset

tableIN = "SizeScaling_20201215.csv"
table_compIN = "SizeScaling_20201215_comp.csv"
statsIN = "Stats_20201215"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
ScaleMat = pd.read_csv(data_root / "Scale_20201215" / "ScaleStats_20201125.csv")
ScaleCurve = pd.read_csv(data_root / "Scale_20201215" / "ScaleCurve_20201125.csv")


# %% Start dataframe
CompMat = pd.DataFrame()

# %% Part 1 pairwise stats cell and nucleus measurement
ps = data_root / statsIN / "cell_nuc_metrics"
counter = -1
for xi, xlabel in enumerate(FS["cellnuc_metrics"]):
    for yi, ylabel in enumerate(FS["cellnuc_metrics"]):
        if xi > yi:
            counter = counter + 1
            CompMat.loc[counter, "Type"] = f"CellNuc"
            CompMat.loc[counter, "Pair"] = f"{xlabel}_{ylabel}"
            # linear
            val = loadps(ps, f"{xlabel}_{ylabel}_rs_vecL")
            pred_yL = loadps(ps, f"{xlabel}_{ylabel}_pred_matL")
            cmin = 100 * np.percentile(val, [50])
            cmin_min = 100 * np.percentile(val, [2.5])
            cmin_max = 100 * np.percentile(val, [97.5])
            CompMat.loc[counter, "Lin"] = cmin
            CompMat.loc[counter, "Lin_min"] = cmin_min
            CompMat.loc[counter, "Lin_max"] = cmin_max
            # complex
            val = loadps(ps, f"{xlabel}_{ylabel}_rs_vecC")
            pred_yL = loadps(ps, f"{xlabel}_{ylabel}_pred_matC")
            cmin = 100 * np.percentile(val, [50])
            cmin_min = 100 * np.percentile(val, [2.5])
            cmin_max = 100 * np.percentile(val, [97.5])
            CompMat.loc[counter, "Com"] = cmin
            CompMat.loc[counter, "Com_min"] = cmin_min
            CompMat.loc[counter, "Com_max"] = cmin_max

# %% Part 2 pairwise stats cell and nucleus measurement
ps = data_root / statsIN / "cellnuc_struct_metrics"
for xi, xlabel in enumerate(FS["cellnuc_metrics"]):
    for yi, ylabel in enumerate(FS["struct_metrics"]):
        for si, pack in enumerate(zip(structures["Gene"], structures["Structure"])):
            struct = pack[0]
            organelle = pack[1]
            counter = counter + 1
            CompMat.loc[counter, "Type"] = f"Struct"
            CompMat.loc[counter, "Pair"] = f"{xlabel}_{organelle}"
            # linear
            val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecL")
            pred_yL = loadps(ps, f"{xlabel}_{ylabel}_{struct}_pred_matL")
            cmin = 100 * np.percentile(val, [50])
            cmin_min = 100 * np.percentile(val, [2.5])
            cmin_max = 100 * np.percentile(val, [97.5])
            CompMat.loc[counter, "Lin"] = cmin
            CompMat.loc[counter, "Lin_min"] = cmin_min
            CompMat.loc[counter, "Lin_max"] = cmin_max
            # complex
            val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecC")
            pred_yL = loadps(ps, f"{xlabel}_{ylabel}_{struct}_pred_matC")
            cmin = 100 * np.percentile(val, [50])
            cmin_min = 100 * np.percentile(val, [2.5])
            cmin_max = 100 * np.percentile(val, [97.5])
            CompMat.loc[counter, "Com"] = cmin
            CompMat.loc[counter, "Com_min"] = cmin_min
            CompMat.loc[counter, "Com_max"] = cmin_max

# %% Composite and unique variance
ps = data_root / statsIN / "struct_composite_metrics"
for xi, xlabel in enumerate(
    ["cellnuc_AV", "cell_A", "cell_V", "cell_AV", "nuc_A", "nuc_V", "nuc_AV"]
):
    for yi, ylabel in enumerate(FS["struct_metrics"]):
        for si, pack in enumerate(zip(structures["Gene"], structures["Structure"])):
            struct = pack[0]
            organelle = pack[1]
            counter = counter + 1
            if xlabel == "cellnuc_AV":
                CompMat.loc[counter, "Type"] = f"Composite"
            else:
                CompMat.loc[counter, "Type"] = f"UniqueVariance"
            CompMat.loc[counter, "Pair"] = f"{xlabel}_{organelle}"
            # linear
            valA = loadps(ps, f"cellnuc_AV_{ylabel}_{struct}_rs_vecL")
            val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecL")
            if xlabel == "cellnuc_AV":
                cmin = 100 * np.percentile(val, [50])
                cmin_min = 100 * np.percentile(val, [2.5])
                cmin_max = 100 * np.percentile(val, [97.5])
                CompMat.loc[counter, "Lin"] = cmin
                CompMat.loc[counter, "Lin_min"] = cmin_min
                CompMat.loc[counter, "Lin_max"] = cmin_max
            else:
                cmin = 100 * (np.percentile(valA, [50]) - np.percentile(val, [50]))
                cmin_min = 100 * (
                    np.percentile(valA, [50]) - np.percentile(val, [97.5])
                )
                cmin_max = 100 * (np.percentile(valA, [50]) - np.percentile(val, [2.5]))
                CompMat.loc[counter, "Lin"] = cmin
                CompMat.loc[counter, "Lin_min"] = cmin_min
                CompMat.loc[counter, "Lin_max"] = cmin_max
            # complex
            valA = loadps(ps, f"cellnuc_AV_{ylabel}_{struct}_rs_vecC")
            val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecC")
            if xlabel == "cellnuc_AV":
                cmin = 100 * np.percentile(val, [50])
                cmin_min = 100 * np.percentile(val, [2.5])
                cmin_max = 100 * np.percentile(val, [97.5])
                CompMat.loc[counter, "Com"] = cmin
                CompMat.loc[counter, "Com_min"] = cmin_min
                CompMat.loc[counter, "Com_max"] = cmin_max
            else:
                cmin = 100 * (np.percentile(valA, [50]) - np.percentile(val, [50]))
                cmin_min = 100 * (
                    np.percentile(valA, [50]) - np.percentile(val, [97.5])
                )
                cmin_max = 100 * (np.percentile(valA, [50]) - np.percentile(val, [2.5]))
                CompMat.loc[counter, "Com"] = cmin
                CompMat.loc[counter, "Com_min"] = cmin_min
                CompMat.loc[counter, "Com_max"] = cmin_max

# %%
data_rootT = data_root / "supplementalfiguredata"
data_rootT.mkdir(exist_ok=True)
CompMat.to_csv(data_rootT / "LinCom_20201217.csv")
