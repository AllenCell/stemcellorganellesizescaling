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

# %% measurements
w1 = 0.05
w2 = 0.1
w3 = 0.05
w4 = 0.1
w5 = 0.1

x1 = 0.3
x2 = 1-w1-x1-w2-w3
x3 = 0.3
x3w = 0.05
x4 = 0.25
x5 = 1-w1-x3-w4-x4-w5-w3


h1 = 0.05
h2 = 0.1
h3 = 0.05
y1 = 0.4
y2 = 1-h1-y1-h2-h3
y2h = 0.05

# %%layout
fs = 10
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({"font.size": fs})

# SIM
axSim = fig.add_axes([w1, h1, x1, y1])
axSim.text(0.5,0.5,'axSim',horizontalalignment='center')
axSim.set_xticks([]),axSim.set_yticks([])

# PCA
axPCA = fig.add_axes([w1+x1+w2, h1, x2, y1])
axPCA.text(0.5,0.5,'axPCA',horizontalalignment='center')
axPCA.set_xticks([]),axPCA.set_yticks([])

# Nuc
axNuc = fig.add_axes([w1, h1+y1+h2, x3, y2])
axNuc.text(0.5,0.5,'axNuc',horizontalalignment='center')
axNuc.set_xticks([]),axNuc.set_yticks([])
# Nuc side
axNucS = fig.add_axes([w1-x3w, h1+y1+h2, x3w, y2])
axNucS.text(0.5,0.5,'axNucS',horizontalalignment='center')
axNucS.set_xticks([]),axNucS.set_yticks([])
# Nuc bottom
axNucB = fig.add_axes([w1, h1+y1+h2-y2h, x3, y2h])
axNucB.text(0.5,0.5,'axNucB',horizontalalignment='center')
axNucB.set_xticks([]),axNucB.set_yticks([])

# Lin
axLin = fig.add_axes([w1+x3+w4, h1+y1+h2, x4, y2])
axLin.text(0.5,0.5,'axLin',horizontalalignment='center')
axLin.set_xticks([]),axLin.set_yticks([])

# Scale
axScale = fig.add_axes([w1+x3+w4+x4+w5, h1+y1+h2, x5, y1])
axScale.text(0.5,0.5,'axScale',horizontalalignment='center')
axScale.set_xticks([]),axScale.set_yticks([])

# plot_save_path = pic_root / f"figure_v8.png"
# plt.savefig(plot_save_path, format="png", dpi=1000)
# plt.close()

plt.show()

# %%


