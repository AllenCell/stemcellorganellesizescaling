# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pickle
import locale
locale.setlocale(locale.LC_ALL, '')
from scipy import interpolate
import os, platform

from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import organelle_scatter


print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################


# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result

#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020/")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020/")
elif platform.system() == "Linux":

    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/2020/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/2020/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

save_flag = 0

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201102.csv"
statsIN = "Stats_20201102"
ps = data_root / statsIN / "cellnuc_struct_metrics"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
print(np.any(cells.isnull()))
structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201113.csv")

# %% Feature sets
FS = {}
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




# %% Parameters, updated directories
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "scatter_plots_4Julie"
pic_root.mkdir(exist_ok=True)

# %% Test plot
for i in np.arange(8):
    print(i)
    bv = i*2
    ev = bv+2
    print(f"{bv}:{ev}")

    plotname = f"jp{i}"
    organelle_scatter(
        FS["cellnuc_metrics"][1:2],
        FS["cellnuc_abbs"][1:2],
        structures['Gene'][bv:ev].to_numpy(),
        "Structure volume",
        cells,
        ps,
        True,
        pic_root,
        plotname,
        kde_flag=True,
        fourcolors_flag=False,
        colorpoints_flag=True,
        rollingavg_flag=True,
        ols_flag=True,
        N2=100,
        plotcells=[],
    )

