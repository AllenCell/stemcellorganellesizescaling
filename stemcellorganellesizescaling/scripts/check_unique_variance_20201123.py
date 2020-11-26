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
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())

# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "scatter_plots"
pic_root.mkdir(exist_ok=True)

#%%
struct = 'RAB5A'
basemodel = ['Cell volume', 'Cell surface area','Nuclear volume', 'Nuclear surface area']
unmodel = ['Cell volume', 'Cell surface area', 'Nuclear surface area']
x1 = cells.loc[cells['structure_name']==struct,basemodel].to_numpy()
x2 = cells.loc[cells['structure_name']==struct,unmodel].to_numpy()
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
print(round(100*rsL1))
print(round(100*rsL2))
print(round(100*(rsL1-rsL2)))

# %%
