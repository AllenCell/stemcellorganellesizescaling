#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
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

# Third party

# Relative
from organelle_size_scaling.utils import regression_20200907

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
print(np.any(cells.isnull()))
cells_COMP = pd.read_csv(data_root / table_compIN)
print(np.any(cells_COMP.isnull()))
structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201014.csv")
Grow = pd.read_csv(data_root / 'growing' / "Growthstats_20201012.csv")
print(np.any(cells_COMP.isnull()))

# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result

# %% measurements
w1 = -1
w2 = 0.01
w3 = 0.05
w4 = 0.02
w5 = 0.01
w6 = 0.01
w7 = 0.01
w8 = 0.01
w9 = 0.01
w10 = 0.03

x1 = 0.2
x2s = 0.05
w1 = w6+x2s
x2 = x1+w2+x1
x3s = 0.03
x3 = (1-(w1+x1+w2+x1+w3+x3s+w4+x3s+w5))/2
x4 = 0.2
x5 = 0.03
x6 = 0.2
x7 = 0.2
x8 = 1-w6-x4-w7-x5-w8-x6-w9-x7-w10-w5
xa =x2


h1 = 0.02
h2 = 0.02
h3 = 0.03
h4 = 0.04
h5 = 0.02

y1 =  0.2
ya = 0.05
y2s = 0.05
y2 = 0.2
y3s = 0.03
y3 = ((h1+y1+ya+y2s+y2)-(h1+y3s+h2+y3s))/2
y4 = 0.3
y5 = 1-(h1+y1+ya+y2s+y2+h3+y4+h5)
y6 = 1-(h1+y1+ya+y2s+y2+h4+h5)
yh = h1+y1+ya+y2s+y2

# %%layout
fig = plt.figure(figsize=(12, 12))

# GrowCell
axGrowCell = fig.add_axes([w1, h1, x1, y1])
axGrowCell.text(0.5,0.5,'GrowCell',horizontalalignment='center')
axGrowCell.set_xticks([]),axGrowCell.set_yticks([])

# GrowNuc
axGrowNuc = fig.add_axes([w1+x1+w2, h1, x1, y1])
axGrowNuc.text(0.5,0.5,'GrowNuc',horizontalalignment='center')
axGrowNuc.set_xticks([]),axGrowNuc.set_yticks([])

# Transition
axTransition = fig.add_axes([w1, h1+y1, xa, ya])
axTransition.text(0.5,0.5,'Transition',horizontalalignment='center')
axTransition.set_xticks([]),axTransition.set_yticks([])

# Grow
axGrow = fig.add_axes([w6+x2s, h1+y1+ya+y2s, x2, y2])
axGrow.text(0.5,0.5,'Grow',horizontalalignment='center')
axGrow.set_xticks([]),axGrow.set_yticks([])

# Grow bottom
axGrowB = fig.add_axes([w6+x2s, h1+y1+ya, x2, y2s])
axGrowB.text(0.5,0.5,'GrowB',horizontalalignment='center')
axGrowB.set_xticks([]),axGrowB.set_yticks([])

# Grow side
axGrowS = fig.add_axes([w6, h1+y1+ya+y2s, x2s, y2])
axGrowS.text(0.5,0.5,'GrowS',horizontalalignment='center')
axGrowS.set_xticks([]),axGrowS.set_yticks([])

# Scale3
axScale3 = fig.add_axes([w1+x1+w2+x1+w3+x3s, h1+y3s, x3, y3])
axScale3.text(0.5,0.5,'Scale3',horizontalalignment='center')
axScale3.set_xticks([]),axScale3.set_yticks([])

# Scale3 side
axScale3S = fig.add_axes([w1+x1+w2+x1+w3, h1+y3s, x3s, y3])
axScale3S.text(0.5,0.5,'Scale3S',horizontalalignment='center')
axScale3S.set_xticks([]),axScale3S.set_yticks([])

# Scale3 bottom
axScale3B = fig.add_axes([w1+x1+w2+x1+w3+x3s, h1, x3, y3s])
axScale3B.text(0.5,0.5,'Scale3B',horizontalalignment='center')
axScale3B.set_xticks([]),axScale3B.set_yticks([])

# Scale4
axScale4 = fig.add_axes([w1+x1+w2+x1+w3+x3s+x3+w4+x3s, h1+y3s, x3, y3])
axScale4.text(0.5,0.5,'Scale4',horizontalalignment='center')
axScale4.set_xticks([]),axScale4.set_yticks([])

# Scale4 side
axScale4S = fig.add_axes([w1+x1+w2+x1+w3+x3s+x3+w4, h1+y3s, x3s, y3])
axScale4S.text(0.5,0.5,'Scale4S',horizontalalignment='center')
axScale4S.set_xticks([]),axScale4S.set_yticks([])

# Scale4 bottom
axScale4B = fig.add_axes([w1+x1+w2+x1+w3+x3s+x3+w4+x3s, h1, x3, y3s])
axScale4B.text(0.5,0.5,'Scale4B',horizontalalignment='center')
axScale4B.set_xticks([]),axScale4B.set_yticks([])

# Scale1
axScale1 = fig.add_axes([w1+x1+w2+x1+w3+x3s, h1+y3s+y3+h2+y3s, x3, y3])
axScale1.text(0.5,0.5,'Scale1',horizontalalignment='center')
axScale1.set_xticks([]),axScale1.set_yticks([])

# Scale1 side
axScale1S = fig.add_axes([w1+x1+w2+x1+w3, h1+y3s+y3+h2+y3s, x3s, y3])
axScale1S.text(0.5,0.5,'Scale1S',horizontalalignment='center')
axScale1S.set_xticks([]),axScale1S.set_yticks([])

# Scale1 bottom
axScale1B = fig.add_axes([w1+x1+w2+x1+w3+x3s, h1+y3s+y3+h2, x3, y3s])
axScale1B.text(0.5,0.5,'Scale1B',horizontalalignment='center')
axScale1B.set_xticks([]),axScale1B.set_yticks([])

# Scale2
axScale2 = fig.add_axes([w1+x1+w2+x1+w3+x3s+x3+w4+x3s, h1+y3s+y3+h2+y3s, x3, y3])
axScale2.text(0.5,0.5,'Scale2',horizontalalignment='center')
axScale2.set_xticks([]),axScale2.set_yticks([])

# Scale2 side
axScale2S = fig.add_axes([w1+x1+w2+x1+w3+x3s+x3+w4, h1+y3s+y3+h2+y3s, x3s, y3])
axScale2S.text(0.5,0.5,'Scale2S',horizontalalignment='center')
axScale2S.set_xticks([]),axScale2S.set_yticks([])

# Scale2 bottom
axScale2B = fig.add_axes([w1+x1+w2+x1+w3+x3s+x3+w4+x3s, h1+y3s+y3+h2, x3, y3s])
axScale2B.text(0.5,0.5,'Scale2B',horizontalalignment='center')
axScale2B.set_xticks([]),axScale2B.set_yticks([])

# Annotation
axAnn = fig.add_axes([w6, yh+h3, x4, y4])
axAnn.text(0.5,0.5,'Ann',horizontalalignment='center')
axAnn.set_xticks([]),axAnn.set_yticks([])

# Organelle Growth rates
axOrgGrow = fig.add_axes([w6+x4+w7, yh+h3, x5, y4])
axOrgGrow.text(0.5,0.5,'OrgGrow',horizontalalignment='center')
axOrgGrow.set_xticks([]),axOrgGrow.set_yticks([])

# Cell Growth rates
axCellGrow = fig.add_axes([w6+x4+w7, yh+h3+y4, x5, y5])
axCellGrow.text(0.5,0.5,'CellGrow',horizontalalignment='center')
axCellGrow.set_xticks([]),axCellGrow.set_yticks([])

# Organelle Variance rates
axOrgVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3, x6, y4])
axOrgVar.text(0.5,0.5,'OrgVar',horizontalalignment='center')
axOrgVar.set_xticks([]),axOrgVar.set_yticks([])

# Cell Variance rates
axCellVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3+y4, x6, y5])
axCellVar.text(0.5,0.5,'CellVar',horizontalalignment='center')
axCellVar.set_xticks([]),axCellVar.set_yticks([])

# Cell Comp rates
axCompVar= fig.add_axes([w6+x4+w7+x5+w8+x6+w9, yh+h3, x7, y4])
axCompVar.text(0.5,0.5,'CompVar',horizontalalignment='center')
axCompVar.set_xticks([]),axCompVar.set_yticks([])

# GrowVarS
axGrowVarS= fig.add_axes([w6+x4+w7+x5+w8+x6+w9+x7+w10, yh+h4, x8, y6])
axGrowVarS.text(0.5,0.5,'GrowVarS',horizontalalignment='center')
axGrowVarS.set_xticks([]),axGrowVarS.set_yticks([])

# plot_save_path = pic_root / f"test.png"
# plt.savefig(plot_save_path, format="png", dpi=1000)
# plt.close()
plt.show()

# %%


