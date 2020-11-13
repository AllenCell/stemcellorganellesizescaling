# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
from matplotlib.colors import ListedColormap
import pickle
import statsmodels.api as sm

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/Nov2020")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Nov2020/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

# Load dataset
tableIN = "SizeScaling_20201102.csv"
table_compIN = "SizeScaling_20201102_comp.csv"
statsIN = "Stats_20201102"
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())
cells_COMP = pd.read_csv(data_root / table_compIN)
np.any(cells_COMP.isnull())

# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "plotwithlog"
pic_root.mkdir(exist_ok=True)

# %%
struct = 'LMNB1'

# %%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
x = cells.loc[cells['structure_name']==struct,'Cell volume'].to_numpy()
y = cells.loc[cells['structure_name']==struct,'Structure volume'].to_numpy()
x_lin = x.copy()
x_lin = sm.add_constant(x_lin)
model_lin = sm.OLS(y, x_lin)
fittedmodel_lin = model_lin.fit()
rs1 = fittedmodel_lin.rsquared
ax1.plot(x,y,'r.',markersize=5)
xC = [np.amin(x), np.amax(x)]
xC_lin = sm.add_constant(xC)
yC = fittedmodel_lin.predict(xC_lin)
xD = [1e6, 2e6]
xD_lin = sm.add_constant(xD)
yD = fittedmodel_lin.predict(xD_lin)
sf = np.round(yD[1]/yD[0],2)
ax1.plot(xC,yC,'b')
ax1.set_xlabel('Cell Vol')
ax1.set_ylabel(f"{struct} vol")
ax1.set_title(f"rs={np.round(rs1,2)}     sf={sf}")
ax1.grid()

x = np.log2(cells.loc[cells['structure_name']==struct,'Cell volume'].to_numpy())
y = np.log2(cells.loc[cells['structure_name']==struct,'Structure volume'].to_numpy())
x_lin = x.copy()
x_lin = sm.add_constant(x_lin)
model_lin = sm.OLS(y, x_lin)
fittedmodel_lin = model_lin.fit()
rs1 = fittedmodel_lin.rsquared
ax2.plot(x,y,'r.',markersize=5)
xC = [np.amin(x), np.amax(x)]
xC_lin = sm.add_constant(xC)
yC = fittedmodel_lin.predict(xC_lin)
xD = [1e6, 2e6]
xD_lin = sm.add_constant(xD)
yD = fittedmodel_lin.predict(xD_lin)
sf = np.round(fittedmodel_lin.params[1],2)
ax2.plot(xC,yC,'b')
ax2.set_xlabel('Cell Vol')
ax1.set_ylabel(f"{struct} vol")
ax2.set_title(f"rs={np.round(rs1,2)}     sf={sf}")
ax2.grid()
plt.show()

