# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
import sys, importlib
from skimage.morphology import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
import vtk
from aicsshparam import shtools
from tqdm import tqdm
import statsmodels.api as sm

# Third party

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/Nov2020")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Nov2020")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201102.csv"
table_compIN = "SizeScaling_20201102_comp.csv"
statsIN = "Stats_20201102"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())
cells_COMP = pd.read_csv(data_root / table_compIN)
np.any(cells_COMP.isnull())

# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "scatter_plots"
pic_root.mkdir(exist_ok=True)

# %% Feature sets
FS = {}
FS["cell_metrics_AVH"] = ["Cell surface area", "Cell volume", "Cell height"]
FS["nuc_metrics_AVH"] = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
FS["cell_metrics_AV"] = ["Cell surface area", "Cell volume"]
FS["nuc_metrics_AV"] = ["Nuclear surface area", "Nuclear volume"]
FS["cell_metrics_H"] = ["Cell height"]
FS["nuc_metrics_H"] = ["Nucleus height"]
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    "Cytoplasmic volume",
]
FS["cellnuc_abbs"] = [
    "Cell area",
    "Cell vol",
    "Cell height",
    "Nuclear area",
    "Nuclear vol",
    "Nucleus height",
    "Cyto vol",
]
FS["cellnuc_COMP_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
]

FS["cellnuc_COMP_abbs"] = [
    "Cell area",
    "Cell vol",
    "Cell height",
    "Nuclear area",
    "Nuclear vol",
    "Nucleus height",
]

FS["selected_structures"] = [
    "LMNB1",
    "ST6GAL1",
    "TOMM20",
    "SEC61B",
    "ATP2A2",
    "LAMP1",
    "RAB5A",
    "SLC25A17",
    "TUBA1B",
    "TJP1",
    "NUP153",
    "FBL",
    "NPM1",
    "SON",
]
FS["other_structures"] = list(
    set(cells["structure_name"].unique()) - set(FS["selected_structures"])
)
# struct_metrics = [
#     "Structure volume",
#     "Number of pieces",
#     "Piece average",
#     "Piece std",
#     "Piece CoV",
#     "Piece sum",
# ]
FS["struct_metrics"] = ["Structure volume", "Number of pieces"]
FS["COMP_types"] = ["AVH","AV","H"]


#%%
x = cells.loc[cells['structure_name']=='LMNB1','Cell volume']
y = cells.loc[cells['structure_name']=='LMNB1','Structure volume']


#%%
x = cells.loc[cells['structure_name']=='LMNB1','Nuclear surface area'].to_numpy()
y = cells.loc[cells['structure_name']=='LMNB1','Structure volume'].to_numpy()
xL = x.copy()
xL = sm.add_constant(xL)
modelL = sm.OLS(y, xL)
fittedmodelL = modelL.fit()
rsL = fittedmodelL.rsquared
yL = fittedmodelL.predict(xL)
print(rsL)

# fig, ax = plt.subplots(figsize=(16, 9))
# ax.plot(x,y,'r.',markersize=5)
# ax.plot(x,yL,'b.',markersize=1)
# plt.show()

#%%
struct = 'LAMP1'
addvar = 'Cell surface area'
basemodel = FS["nuc_metrics_AV"]
x1 = cells.loc[cells['structure_name']==struct,basemodel].to_numpy()
x2 = cells.loc[cells['structure_name']==struct,[*basemodel, addvar]].to_numpy()
y = cells.loc[cells['structure_name']==struct,'Structure volume'].to_numpy()
xL1 = x1.copy()
xL1 = sm.add_constant(xL1)
modelL = sm.OLS(y, xL1)
fittedmodelL = modelL.fit()
rsL1 = fittedmodelL.rsquared
xL2 = x2.copy()
xL2 = sm.add_constant(xL2)
modelL = sm.OLS(y, xL2)
fittedmodelL = modelL.fit()
rsL2 = fittedmodelL.rsquared
print(rsL1)
print(rsL2)
print((rsL2-rsL1)/(1-rsL1))




