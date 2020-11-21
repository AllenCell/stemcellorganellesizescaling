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

# # Resolve directories
# data_root = dirs[0]
# pic_root = dirs[1]
#
# tableIN = "SizeScaling_20201012.csv"
# table_compIN = "SizeScaling_20201012_comp.csv"
# statsIN = "Stats_20201012"
# # Load dataset
# cells = pd.read_csv(data_root / tableIN)
# print(np.any(cells.isnull()))
# cells_COMP = pd.read_csv(data_root / table_compIN)
# print(np.any(cells_COMP.isnull()))
# structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201014.csv")
# Grow = pd.read_csv(data_root / 'growing' / "Growthstats_20201012.csv")
# print(np.any(cells_COMP.isnull()))


# %% measurements
w1 = -1
w2 = 0
w3 = 0.01
w4 = 0.02
w5 = 0.01
w6 = 0.01
w7 = 0.01
w8 = 0.01
w9 = 0.01
w10 = 0.03
w11 = -0.02
w12 = 0.02
w13 = 0.05
w14 = 0.01

x3s = 0.03
x8s = 0.05
x8 = 0.4
x3 = (1-(w10+x8+x8s+x3s+w4+x3s+w5))/2
x4 = 0.13
x4r = 0.013
x4l = x4-x4r
x5 = 0.03
x6 = 0.2
x7 = 0.2
x7l = x7/6
x7r = x7-x7l
# x8 = 1-w6-x4-w7-x5-w8-x6-w9-x7-w12-x7-w10-x8s-w5
x9 = x7l+w12+x7r-w11+w9
x10 = x7r
x2s = 0.03
xw = w6+x4+w7+x5+w8+x6+w9+x7l+w12+x7r
x1 = 1-xw-w13-w14


h1 = 0.03
h2 = 0.02
h3 = 0.03
h4 = 0.01
h5 = 0.005
h6 = 0.015
h7 = 0.035
h8 = 0.02
h9 = 0.015

y3s = 0.03
y6s = 0.05
y6 = 0.4
y3 = ((h4+y6+y6s)-(y3s+h2+y3s))/2
y4 = 0.3
# y6 = 1-(h1+y1+ya+y2s+y2+h4+h5+y6s)
yh = h4+y6+y6s
y5 = 1-(yh+y4+h5+h3)
y7 = (y5+h5-h6-h7-h8-h9)/3
y1 =  0.2
ya = 0.03
y2s = 0.03
y2 = 1-yh-h3-y1-ya-y2s-h5


# %%layout
fs = 10
fig = plt.figure(figsize=(12, 12))
plt.rcParams.update({"font.size": fs})

# Scale4
axScale4 = fig.add_axes([w3+x3s, y3s, x3, y3])
axScale4.text(0.5,0.5,'Scale4',horizontalalignment='center')
axScale4.set_xticks([]),axScale4.set_yticks([])
# Scale4 side
axScale4S = fig.add_axes([w3, y3s, x3s, y3])
axScale4S.text(0.5,0.5,'Scale4S',horizontalalignment='center')
axScale4S.set_xticks([]),axScale4S.set_yticks([])
# Scale4 bottom
axScale4B = fig.add_axes([w3+x3s, 0, x3, y3s])
axScale4B.text(0.5,0.5,'Scale4B',horizontalalignment='center')
axScale4B.set_xticks([]),axScale4B.set_yticks([])

# Scale5
axScale5 = fig.add_axes([w3+x3s+x3+x3s+w4, y3s, x3, y3])
axScale5.text(0.5,0.5,'Scale5',horizontalalignment='center')
axScale5.set_xticks([]),axScale5.set_yticks([])
# Scale5 side
axScale5S = fig.add_axes([w3+x3+x3s+w4, y3s, x3s, y3])
axScale5S.text(0.5,0.5,'Scale5S',horizontalalignment='center')
axScale5S.set_xticks([]),axScale5S.set_yticks([])
# Scale5 bottom
axScale5B = fig.add_axes([w3+x3s+x3+x3s+w4, 0, x3, y3s])
axScale5B.text(0.5,0.5,'Scale5B',horizontalalignment='center')
axScale5B.set_xticks([]),axScale5B.set_yticks([])

# Scale1
axScale1 = fig.add_axes([w3+x3s, y3s+y3+h2+y3s, x3, y3])
axScale1.text(0.5,0.5,'Scale1',horizontalalignment='center')
axScale1.set_xticks([]),axScale1.set_yticks([])
# Scale1 side
axScale1S = fig.add_axes([w3, y3s+y3+h2+y3s, x3s, y3])
axScale1S.text(0.5,0.5,'Scale1S',horizontalalignment='center')
axScale1S.set_xticks([]),axScale1S.set_yticks([])
# Scale1 bottom
axScale1B = fig.add_axes([w3+x3s, 0+y3+h2+y3s, x3, y3s])
axScale1B.text(0.5,0.5,'Scale1B',horizontalalignment='center')
axScale1B.set_xticks([]),axScale1B.set_yticks([])

# Scale2
axScale2 = fig.add_axes([w3+x3s+x3+x3s+w4, y3s+y3+h2+y3s, x3, y3])
axScale2.text(0.5,0.5,'Scale2',horizontalalignment='center')
axScale2.set_xticks([]),axScale2.set_yticks([])
# Scale2 side
axScale2S = fig.add_axes([w3+x3+x3s+w4, y3s+y3+h2+y3s, x3s, y3])
axScale2S.text(0.5,0.5,'Scale2S',horizontalalignment='center')
axScale2S.set_xticks([]),axScale2S.set_yticks([])
# Scale2 bottom
axScale2B = fig.add_axes([w3+x3s+x3+x3s+w4, 0+y3+h2+y3s, x3, y3s])
axScale2B.text(0.5,0.5,'Scale2B',horizontalalignment='center')
axScale2B.set_xticks([]),axScale2B.set_yticks([])

# # GrowVarS side
axGrowVarSS= fig.add_axes([w3+x3s+x3+w4+x3s+x3+w10+x8, h4, x8s, y6])
axGrowVarSS.text(0.5,0.5,'GrowVarS',horizontalalignment='center')
axGrowVarSS.set_xticks([]),axGrowVarSS.set_yticks([])
# GrowVarS
axGrowVarS= fig.add_axes([w3+x3s+x3+w4+x3s+x3+w10, h4, x8, y6])
axGrowVarS.text(0.5,0.5,'GrowVarSS',horizontalalignment='center')
axGrowVarS.set_xticks([]),axGrowVarS.set_yticks([])
# GrowVarS bottom
axGrowVarSB= fig.add_axes([w3+x3s+x3+w4+x3s+x3+w10, h4+y6, x8, y6s])
axGrowVarSB.text(0.5,0.5,'axGrowVarSB',horizontalalignment='center')
axGrowVarSB.set_xticks([]),axGrowVarSB.set_yticks([])

# Annotation
axAnn = fig.add_axes([w6+x4l, yh+h3, x4r, y4])
axAnn.text(0.5,0.5,'axAnn',horizontalalignment='center')
axAnn.set_xticks([]),axAnn.set_yticks([])

# Organelle Growth rates
axOrgGrow = fig.add_axes([w6+x4+w7, yh+h3, x5, y4])
axOrgGrow.text(0.5,0.5,'axOrgGrow',horizontalalignment='center')
axOrgGrow.set_xticks([]),axOrgGrow.set_yticks([])

# Cell Growth rates
axCellGrow = fig.add_axes([w6+x4+w7, yh+h3+y4, x5, y5])
axCellGrow.text(0.5,0.5,'axCellGrow',horizontalalignment='center')
axCellGrow.set_xticks([]),axCellGrow.set_yticks([])

# Organelle Variance rates
axOrgVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3, x6, y4])
axOrgVar.text(0.5,0.5,'axOrgVar',horizontalalignment='center')
axOrgVar.set_xticks([]),axOrgVar.set_yticks([])

# Cell Variance rates
axCellVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3+y4, x6, y5])
axCellVar.text(0.5,0.5,'axCellVar',horizontalalignment='center')
axCellVar.set_xticks([]),axCellVar.set_yticks([])

# Composite model All rates
axAllVarC= fig.add_axes([w6+x4+w7+x5+w8+x6+w9, yh+h3, x7l, y4])
axAllVarC.text(0.5,0.5,'axAllVarC',horizontalalignment='center')
axAllVarC.set_xticks([]),axAllVarC.set_yticks([])

# Composite model unique rates
axCompVarC= fig.add_axes([w6+x4+w7+x5+w8+x6+w9+x7l+w12, yh+h3, x7r, y4])
axCompVarC.text(0.5,0.5,'axCompVarC',horizontalalignment='center')
axCompVarC.set_xticks([]),axCompVarC.set_yticks([])

axUniVarBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w9+x7l+w12, yh+h3+y4+h6, x10, y7])
axUniVarBar.text(0.5,0.5,'axUniVarBar',horizontalalignment='center')
axUniVarBar.set_xticks([]),axUniVarBar.set_yticks([])

axExpVarBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w11, yh+h3+y4+h6+y7+h7, x9, y7])
axExpVarBar.text(0.5,0.5,'axExpVarBar',horizontalalignment='center')
axExpVarBar.set_xticks([]),axExpVarBar.set_yticks([])

axGrowBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w11, yh+h3+y4+h6+y7+h7+y7+h8, x9, y7])
axGrowBar.text(0.5,0.5,'axGrowBar',horizontalalignment='center')
axGrowBar.set_xticks([]),axGrowBar.set_yticks([])

# GrowCell
axGrowCell = fig.add_axes([xw+w13, yh+h1, x1, y1])
axGrowCell.text(0.5,0.5,'GrowCell',horizontalalignment='center')
axGrowCell.set_xticks([]),axGrowCell.set_yticks([])

# Grow
axGrow = fig.add_axes([xw+w13, yh+h1+y1+ya+y2s, x1, y2])
axGrow.text(0.5,0.5,'Grow',horizontalalignment='center')
axGrow.set_xticks([]),axGrow.set_yticks([])
# Grow bottom
axGrowB = fig.add_axes([xw+w13, yh+h1+y1+ya, x1, y2s])
axGrowB.text(0.5,0.5,'GrowB',horizontalalignment='center')
axGrowB.set_xticks([]),axGrowB.set_yticks([])
# Grow side
axGrowS = fig.add_axes([xw+w13-x2s, yh+h1+y1+ya+y2s, x2s, y2])
axGrowS.text(0.5,0.5,'GrowS',horizontalalignment='center')
axGrowS.set_xticks([]),axGrowS.set_yticks([])

# Transition
axTransition = fig.add_axes([xw+w13, yh+h1+y1, x1, ya])
axTransition.text(0.5,0.5,'Transition',horizontalalignment='center')
axTransition.set_xticks([]),axTransition.set_yticks([])


plot_save_path = pic_root / f"figure_v8.png"
plt.savefig(plot_save_path, format="png", dpi=1000)
plt.close()




