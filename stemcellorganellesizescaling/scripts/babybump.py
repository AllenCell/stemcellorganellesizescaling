# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
import sys, importlib

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

tableIN = "SizeScaling_20201012.csv"
table_compIN = "SizeScaling_20201012_comp.csv"
statsIN = "Stats_20201012"
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


# %%
xvec = [6]
yvec = [4]
pair1 = np.stack((xvec, yvec)).T

# %%
xvec = [1, 1, 6, 1, 4, 6]
yvec = [4, 6, 4, 0, 3, 3]
pair6 = np.stack((xvec, yvec)).T

# %%
N = 13
xvec = np.random.choice(len(FS["cellnuc_metrics"]), N)
yvec = np.random.choice(len(FS["cellnuc_metrics"]), N)
pairN = np.stack((xvec, yvec)).T

# %%
L = len(FS["cellnuc_metrics"])
pair21 = np.zeros((int(L * (L - 1) / 2), 2)).astype(np.int)
i = 0
for f1 in np.arange(L):
    for f2 in np.arange(L):
        if f2 > f1:
            pair21[i, :] = [f1, f2]
            i += 1


#%%
plotname = "test"
ps = data_root / statsIN / "cell_nuc_metrics"
fscatter(
    FS["cellnuc_metrics"],
    FS["cellnuc_abbs"],
    pair1,
    cells,
    ps,
    False,
    pic_root,
    f"{plotname}_plain",
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
)

#%%
fac = 1000
x = cells['Cytoplasmic volume']/fac
y = cells['Nuclear volume']/fac
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x,y,'r.',markersize=.5)
ax.grid()
plt.show()

#%%
import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y, x)
fittedmodel = model.fit()
fittedmodel.params
b = fittedmodel.params[0]
a = fittedmodel.params[1]

# %%
x = cells['Cytoplasmic volume']/fac
y = cells['Nuclear volume']/fac
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x,y,'r.',markersize=.5)
offset = -120
ax.plot([min(x), max(x)],[offset+b+a*min(x), offset+b+a*max(x)])
ax.grid()
cond = ((y<(offset+b+a*x)) & (x>500) & (x<1500) & (y>100) & (y<350))
ax.plot(x[cond],y[cond],'.',markersize=.5,color='deepskyblue')
plt.show()
babycells = cells.loc[cond,'CellId'].values
len(babycells)
# %%
bcells = pd.DataFrame()
bcells['CellId'] = babycells
bcells.to_csv(data_root / 'babycells' / 'babybump_20201016.csv')

# %% Save part without babycells and only babycells
cells.loc[~cond].to_csv(data_root / 'babycells' / 'SizeScaling_20201012_nobaby.csv')
cells.loc[cond].to_csv(data_root / 'babycells' / 'SizeScaling_20201012_onlybaby.csv')
